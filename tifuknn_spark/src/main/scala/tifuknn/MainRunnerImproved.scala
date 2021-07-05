/*
* One issue with the state call back function is that the incoming baskets have no orders.
* how do we make sure the baskets are updated in their correct orders?
* Option 1: use a timestamp column, does spark then take care of the ordering?
* Option 2: Make sure we only send a basket session at a time, essentially solving from the data provider side.
*
* Steps:
* 1. Define input, state and output, (in our case, output is the same as state)
* 2. Defin function to update state, the callback function, it has to take 3 inputs (input, iterator, state), the return
* depends on the output in step 1
* 3.
* References
* 1. https://databricks.com/blog/2017/10/17/arbitrary-stateful-processing-in-apache-sparks-structured-streaming.html
* 2. https://databricks.com/session/deep-dive-into-stateful-stream-processing-in-structured-streaming
*
*
 */

// Note: in this version all history needs to be stored. The implementation is not completed yet.
// In the V2 version, we will keep the full history in a separate static DF and join the stream when needed.
package tifuknn


import org.apache.spark.sql.streaming._
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, SparkSession}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}

import java.io.File
import com.github.tototoshi.csv._

import java.nio.file.{Files, Paths}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

object MainRunnerImproved {
  val cleanUpState = true
  val decrementFromStart = true
  val randomDecrement = false
  // setting some hyper parameters
  val groupSize = 2
  val rb = 0.9
  val rg = 0.7
  val basePath: String = "eval_results/tafang";
  val userVectorPath: String = "user_vector"
  val dataPath =
    "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/jsondata/tafang"
  val vocabPath =
    "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/jsondata/tafang_vocab.csv"
  var batchOutputevalFile: String = "";
  var stateUpdateEvalFile: String = "";

  val checkpointLocation: String = "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/checkpoint_dec"
  val spark: SparkSession = SparkSession
    .builder
    .appName("TIFUKNN")
    .master("local[*]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("WARN")
  // see: https://jaceklaskowski.gitbooks.io/spark-structured-streaming/content/spark-sql-streaming-properties.html
  //default is 10, this will trigger a snapshot every 3 (2+1) deltas
  spark.conf.set("spark.sql.streaming.stateStore.minDeltasForSnapshot", 10)
  //default is 100, this will keep the last 3 (2+1) commits and offsets and deltas
  spark.conf.set("spark.sql.streaming.minBatchesToRetain", 10)
  //default is 60s, this will trigger the maintenance task every 10 seconds
  spark.conf.set("spark.sql.streaming.stateStore.maintenanceInterval", "10s")
  // dont scan when there is no data.
  //  spark.conf.set("spark.sql.streaming.noDataMicroBatches.enabled", "false")
  // the default is 200 that uis used for shuffling data for joins and aggregations, which might sometimes
  // turn unnecessary, because you will have to wait for 200 tasks to finish.
  // notice that in my local computer, a small shuffle partition significantly speeds up...
  // notice that the state in checkpoint root dir, which follows state/<opearatorId>/<partitionId>/<1-N>.delta
  // also stores only 1 parition instead of 200!
  // for why the parition number：
  // https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-performance-tuning-groupBy-aggregation.html
  spark.sqlContext.setConf("spark.sql.shuffle.partitions", "1")

  import spark.implicits._

  implicit val sqlCtx: SQLContext = spark.sqlContext


  // vocab as parameterless method to conform to uniform access principle, see PinScala chapter 10
  var vocabLength: Int = -1 // record the length of vocabulary
  def vocab: Array[String] = {
    val vocabDF = spark.read.option("header", "true").csv(vocabPath)
    val vocabArray = vocabDF.select("itemId").as[String].collect()
    vocabLength = vocabArray.length
    vocabArray
  }

  // this is a parameterless method is scala, which can be accessed like a field
  def basketsOneHotDF: DataFrame = {
    val rawBasketsDF = spark.readStream
      .option("maxFilesPerTrigger", 1) // add maxFilesPerTrigger to limit max files
      .schema(basketSchema)
      .json(dataPath)
    val cvm = new CountVectorizerModel(vocab).setInputCol("basket").setOutputCol("basketOneHot")
    // toArr is a function of type Vector => Array[Double], its value is _.toArray
    val toArr: Vector => Array[Double] = _.toArray
    val toArrUdf = udf(toArr)
    val basketsOneHotDF = cvm
      .transform(rawBasketsDF)
      .drop("basket")
      .withColumn("basketOneHot", toArrUdf(col("basketOneHot")))
    basketsOneHotDF
  }

  def getInitState(customerId: Long): UserState = {
    UserState(
      customerId = customerId,
      rg = rg,
      rb = rb,
      countOfBaskets = 0,
      countOfGroups = 1,
      groupSize = groupSize,
      userVector = Array.ofDim[Double](vocabLength), // zero array of vocab size
      allBasketGroups = Array(Array.empty), // zero array of vocab size
      allGroupVectors = Array(Array.ofDim[Double](vocabLength))
    )
  }

  case class UserState(
                        customerId: Long,
                        rg: Double, // group decay rate for calculating user vector
                        rb: Double, // basket decay rate for calculating group vector
                        groupSize: Long,
                        countOfBaskets: Long,
                        countOfGroups: Long,
                        allBasketGroups: Array[Array[InputBasketRow]], // group composition encoded in the structure
                        allGroupVectors: Array[Array[Double]],
                        userVector: Array[Double]
                      )


  def main(args: Array[String]): Unit = {
    if (cleanUpState) {
      println(s"Cleaning up state directory $checkpointLocation")
      FileUtils.cleanDirectory(new File(checkpointLocation))
    }
    Files.createDirectories(Paths.get(s"${basePath}"))

    // convert DF to DS for creating streaming query
    val basketsTypedStream: Dataset[InputBasketRow] = basketsOneHotDF.as[InputBasketRow]
    basketsTypedStream.printSchema()
    // we have a state per customer because of the groupByKey operation
    // mapGroupsWithState returns a dataset[T] where T is the return type of updateState function
    // NoTimeout is by default, instead of .mapGroupsWithState(GroupStateTimeout.NoTimeout)(updateStateCallback)
    val userStream: Dataset[UserState] = basketsTypedStream
      .groupByKey(_.customerId)
      .mapGroupsWithState(GroupStateTimeout.NoTimeout)(updateStateCallback)

    val query = userStream
      .writeStream
      .foreachBatch(printAndSaveOutput _)
      .outputMode("update") // complete and append mode not supportted for this query
      .option("checkpointLocation", checkpointLocation)
      .start()
    query.awaitTermination()
  }

  def findIndex(customerId: Long, orderId: Long, currentState: UserState): Seq[Int] = {
    // we try to find the location of the basket to be deleted in the full basket history
    // todo: better to build an idex from (customerId, orderID) to index location
    var i: Int = -1 // group index
    var j: Int = -1 // within index
    val randomI = scala.util.Random.nextInt(currentState.allBasketGroups.length) // an random group index
    val currentBasketGroups = currentState.allBasketGroups
    for ((groupBaskets, i_) <- currentBasketGroups.zipWithIndex) {
      for ((basket, j_) <- groupBaskets.zipWithIndex)
        if (customerId == basket.customerId && orderId == basket.orderId) {
          i = i_
          j = j_
          if (randomDecrement && i == randomI) {return Seq(i, j)} // if we want to decrement from a random group loc.
          if (decrementFromStart){return Seq(i, j)}
        }
    }
    Seq(i, j)
  }

  def updateStateCallback(customerId: Long,
                          newBaskets: Iterator[InputBasketRow],
                          groupState: GroupState[UserState]): UserState = {
    /*
    The user state is returned as streaming query output, which is later on stored for recommendation usage.
     */
    // step 1 get previous state
    println(f"---------Started updating state------------")
    val startTime = System.nanoTime
    var state: UserState = if (groupState.exists) {
      // get old state if existing
      groupState.get
    } else {
      // initialize the state at beginning
      getInitState(customerId)
    }
    // step 2 update the input state with new data
    for (basket <- newBaskets) {
      state = updateUserStateWithNewBasket(state, basket)
      // TODO: why we have to update the old state here??
      // update: it seems to be necessary, if I comment out this, the states are not updated.
      // question is I have already returned the updated state, why I have to update the old state again?
      // answer: check the docstring for update function, it seems that the update is required, while the
      // returned state is not used for the purpose of updating state, but it is needed for the output of the query
      groupState.update(state)
    }

    val endTime = System.nanoTime
    val duration = (endTime - startTime) / scala.math.pow(10, 6)
    if (stateUpdateEvalFile == "") {
      val ts = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm").format(LocalDateTime.now)
      stateUpdateEvalFile = s"${basePath}/state_update_time_${ts}.csv"
      val file = new File(stateUpdateEvalFile)
      val writer = CSVWriter.open(file, append = true)
      writer.writeRow(List("customerId", "ms", "groupSize", "rg", "rb", "userVectorFirstEle", "lastGroupVectorFirstEle"))
      writer.writeRow(List(
        customerId,
        duration,
        state.groupSize,
        state.rg,
        state.rb,
        state.userVector(0),
        state.allGroupVectors.last(0),
      ))
    } else {
      val file = new File(stateUpdateEvalFile)
      val writer = CSVWriter.open(file, append = true)
      writer.writeRow(List(
        customerId,
        duration,
        state.groupSize,
        state.rg,
        state.rb,
        state.userVector(0),
        state.allGroupVectors.last(0),
      ))
    }
    println(f"Time(ms) to update state: ${duration}")
    println(f"---------Finished updating state------------")
    // step 3 return the updated state
    state
  }

  def updateUserStateWithNewBasket(currentState: UserState, input: InputBasketRow): UserState = {
    println("------Started processing new basket-------")
    val currentLastGroupVector = currentState.allGroupVectors.last
    val currentLastGroupSize = currentState.allBasketGroups.last.length
    val currentUserVector = currentState.userVector
    val currentCountOfGroups = currentState.countOfGroups
    val groupSize = currentState.groupSize
    val rb = currentState.rb
    val rg = currentState.rg
    val isDeletion = input.isDeletion

    if (!isDeletion) {
      // the incremental case
      val newCountOfBaskets = currentState.countOfBaskets + 1
      if (groupSize == currentLastGroupSize) {
        // scenario 1: create a new group
        println(s"Incremental 1: creating new group...")
        val newLastGroupOfBaskets = Array(input)
        val newCountOfGroups = currentCountOfGroups + 1
        val newLastGroupVector = input.basketOneHot
        val newUserVector =
          (currentUserVector.map(_ * rg * currentCountOfGroups / newCountOfGroups),
            newLastGroupVector.map(_ / newCountOfGroups)).zipped.map(_ + _)
        // update state
        val newState = currentState.copy(
          countOfBaskets = newCountOfBaskets,
          countOfGroups = newCountOfGroups,
          userVector = newUserVector,
          allGroupVectors = currentState.allGroupVectors ++ Seq(newLastGroupVector),
          allBasketGroups = currentState.allBasketGroups ++ Seq(newLastGroupOfBaskets)
        )
        println("------Finished processing new basket-------")
        newState
      } else {
        println(s"Incremental 2: updating last group...")
        // scenario 2: the group length is less than group size, last group will be updated
        val newLastGroupOfBaskets = currentState.allBasketGroups.last ++ Seq(input) // add the new basket to last group
        val newLastGroupSize = currentLastGroupSize + 1
        val newLastGroupVector =
          (currentLastGroupVector.map(_ * rb * currentLastGroupSize / newLastGroupSize),
            input.basketOneHot.map(_ / newLastGroupSize)).zipped.map(_ + _)
        val newUserVector = (currentUserVector,
          newLastGroupVector.map(_ / currentCountOfGroups),
          currentLastGroupVector.map(_ * (-1.0) / currentCountOfGroups)).zipped.map(_ + _ + _)
        // update state
        val newState = currentState.copy(
          countOfBaskets = newCountOfBaskets,
          userVector = newUserVector,
          allGroupVectors = currentState.allGroupVectors.updated(currentState.allGroupVectors.length - 1, newLastGroupVector),
          allBasketGroups = currentState.allBasketGroups.updated(currentState.allBasketGroups.length - 1, newLastGroupOfBaskets)
        )
        println("------Finished processing new basket-------")
        newState
      }

    }
    else {
      // the decremental case
      // step 1 get the customerId and orderID to identify where to delete the basket
      val customerIdToDelete = input.customerId
      val orderIdToDelete = input.orderId
      val currentBasketGroups = currentState.allBasketGroups
      // we try to find the location of the basket to be deleted in the full basket history
      val index = findIndex(customerIdToDelete, orderIdToDelete, currentState)
      val i = index.head
      val j = index.last
      println(f"we found the basket to delete at group index $i, and basket index $j")
      if (i < 0 && j < 0) return currentState // the basket not found, nothing needs to be updated
      val lengthOfGroup = currentBasketGroups(i).length
      if (lengthOfGroup > 1) {
        println("Decremental 1: removing basket within a group, updating group vector then updating user vector")
        // scenario 1: we have more than one basket to delete within a group, group vector is updated
        val newCountOfBaskets = currentState.countOfBaskets - 1
        // remove the basket from group
        // now lets update group vector
        val baskeGroupUnderImpact: Array[Array[Double]] = currentState.allBasketGroups(i).map(x => x.basketOneHot)
        val groupVectorUnderImpact = currentState.allGroupVectors(i)
        val followsBasketVectors = baskeGroupUnderImpact.drop(j)
        val shiftLeftVectors = followsBasketVectors.tail :+ Array.ofDim[Double](vocabLength)
        //        val diff = sum(shiftLeftVectors, followsGroupVectors)
        val diff = myArraySubtract(shiftLeftVectors, followsBasketVectors)
        // this creats the decayed weights, careful with the index [rg^(tau -i),..., rg, 0]
        val decayWeights: Array[Double] =
          (lengthOfGroup - 1 - j until -1 by -1)
            .map(x => scala.math.pow(currentState.rb, x))
            .toArray
        val mutiplied: Array[Array[Double]] = myArrayMultiply(diff, decayWeights)
        val dotProduct: Array[Double] = myArraySum(mutiplied)
        // now the updated group vector
        val newGroupVector = myMultiply(
          mySum(myMultiply(groupVectorUnderImpact, lengthOfGroup), dotProduct),
          1.0 / ((lengthOfGroup - 1) * rb)
        )
        // update the new group vector
        val newAllGroupVectors = currentState.allGroupVectors
        newAllGroupVectors(i) = newGroupVector

        // update the user vector
        val groupDiff = mySum(myMultiply(groupVectorUnderImpact, -1.0), newGroupVector)
        val newUserVector = mySum(currentUserVector, myMultiply(groupDiff,
          scala.math.pow(rg, currentCountOfGroups - i - 1) / currentCountOfGroups))
        // remove the basket group history
        val newAllBasketGroups = currentState.allBasketGroups // copy the value for later update
        newAllBasketGroups(i) = newAllBasketGroups(i).patch(j, Nil, 1)

        // only need to update the fields needed
        val newState = currentState.copy(
          countOfBaskets = newCountOfBaskets,
          allGroupVectors = newAllGroupVectors,
          allBasketGroups = newAllBasketGroups,
          userVector = newUserVector,
        )
        println("------Finished processing new basket-------")
        newState
      }
      else {
        println("Decremental 2: removing group vector then updating user vector")
        // scenario 2: the group contains only 1 basket, we are going to remove this group vector
        // update counts
        val newCountOfBaskets = currentState.countOfBaskets - 1
        val newCountOfGroups = currentState.countOfGroups - 1
        if (newCountOfBaskets == 0) {
          getInitState(customerIdToDelete)
        }
        else {
          // update group vectors
          val newAllGroupVectors = currentState.allGroupVectors.patch(i, Nil, 1) //remove the group vector at i
          val newAllBasketGroups = currentState.allBasketGroups.patch(i, Nil, 1) // remove the group baskets at i
          // update user vector finally according to Eq. 12
          // 1. get all the sebsequent group vectors
          val followsGroupVectors = currentState.allGroupVectors.drop(i)

          // 2. create the first order diff of the subsequent group vector, the D(gi,...g_n) part in equation
          // ie from [x1， x2, x3] to [x2-x1, x3-x2, -x3]
          // this shifts by 1 to left (tail operation) and append 0 vector
          // [x2, x3, 0]
          val shiftLeftVectors = followsGroupVectors.tail :+ Array.ofDim[Double](vocabLength)
          //        val diff = sum(shiftLeftVectors, followsGroupVectors)
          val diff = myArraySubtract(shiftLeftVectors, followsGroupVectors)
          // this creats the decayed weights, careful with the index [rg^(tau -i),..., rg, 0]
          val decayWeights: Array[Double] =
            (newCountOfGroups - i until -1 by -1)
              .map(x => scala.math.pow(currentState.rg, x))
              .toArray
          val mutiplied: Array[Array[Double]] = myArrayMultiply(diff, decayWeights)
          val dotProduct: Array[Double] = myArraySum(mutiplied)
          //          var newUserVector = Array.ofDim[Double](vocabLength)
          var newUserVector = myMultiply(
            mySum(myMultiply(currentUserVector, currentState.countOfGroups), dotProduct),
            1.0 / (newCountOfGroups * rg)
          )

          //          def logArray(input: Array[Double]): Array[Double] = {
          //            input.map(x => scala.math.log(x))
          //          }
          //
          //          def expArray(input: Array[Double]): Array[Double] = {
          //            input.map(x => scala.math.exp(x))
          //          }

          //          val logVar = scala.math.log(newCountOfGroups * rg)
          //          var newUserVector = expArray(
          //            mySubstract(
          //              logArray(mySum(myMultiply(currentUserVector, currentState.countOfGroups), dotProduct)),
          //              Array.fill[Double](vocabLength)(logVar)
          //            )
          //          )
          // perhaps this is more numerically stable, not sure.
          //          val newUserVectorAlt = mySum(myMultiply(currentUserVector, currentState.countOfGroups/(newCountOfGroups * rg)),
          //          myMultiply(dotProduct, 1.0/(newCountOfGroups * rg)))


          // the vector value must be less than 1, give some slack for error
          //          assert(newUserVector(0) <= 1.1)
          if (newUserVector.forall(scala.math.abs(_) >= 20)) {
            println(s"user vector values are larger than 1.05..., camp down to 1.0 mmanually...")
            newUserVector = Array.fill[Double](vocabLength)(1.0)
          }
          // only need to update the fields needed
          val newState = currentState.copy(
            countOfBaskets = newCountOfBaskets,
            countOfGroups = newCountOfGroups,
            allGroupVectors = newAllGroupVectors,
            allBasketGroups = newAllBasketGroups,
            userVector = newUserVector,
          )
          println("------Finished processing new basket-------")
          newState
        }

      }


    }
  }


  def printAndSaveOutput(streamOutputData: Dataset[UserState], batchId: Long): Unit = {
    /*
    This function allows us to exucute a function on the output of each micro-batch of streaming query.
    In our case, it is the updated state containing the user vector.
    We first print out the output data, we then save it to some place
     */

    println(s"------Started outputing micro-batch with batch id $batchId --------")
    // we should measure time taken here, because the bicro batch is only lazily triggered
    // by output action here

    val startTime = System.nanoTime
    // persist the df to avoid multiple output of streaming calculations
    streamOutputData.persist()
    streamOutputData.select("customerId", "rg", "rb", "groupSize", "countOfBaskets", "countOfGroups", "userVector")
      .show(true)
    val endTime = System.nanoTime
    val duration = (endTime - startTime) / scala.math.pow(10, 6)
    val state = streamOutputData.take(1)(0)
    if (batchOutputevalFile == "") {
      val ts = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm").format(LocalDateTime.now)
      batchOutputevalFile = s"${basePath}/output_batch_time_${ts}.csv"
      val file = new File(batchOutputevalFile)
      val writer = CSVWriter.open(file, append = true)
      writer.writeRow(List("batchId", "ms", "groupSize", "rg", "rb", "userVectorFirstEle", "lastGroupVectorFirstEle"))
      writer.writeRow(List(
        batchId,
        duration,
        state.groupSize,
        state.rg,
        state.rb,
        state.userVector(0),
        state.allGroupVectors.last(0),
      ))
    } else {
      val file = new File(batchOutputevalFile)
      val writer = CSVWriter.open(file, append = true)
      writer.writeRow(List(
        batchId,
        duration,
        state.groupSize,
        state.rg,
        state.rb,
        state.userVector(0),
        state.allGroupVectors.last(0),
      ))
    }
    // we also want to record the user vector
    val customerId = state.customerId
    // create directory if not exists
    Files.createDirectories(Paths.get(s"${basePath}/${userVectorPath}"))
    val userVectorFile = s"${basePath}/${userVectorPath}/customer_${customerId}.csv"
    val file = new File(userVectorFile)
    val writer = CSVWriter.open(file, append = false)
    writer.writeRow(List("customerId", "groupSize", "rg", "rb","countOfBaskets", "countOfGroups", "userVector"))
    writer.writeRow(List(
      state.customerId,
      state.groupSize,
      state.rg,
      state.rb,
      state.countOfBaskets,
      state.countOfGroups,
      state.userVector.mkString("[", ",", "]"), // convert it to format [1,1,1]
    ))
    writer.close()




    println(s"Time(ms) to output micro-batch: $duration")
    println(s"------Finished outputing micro-batch------")

  }

}

