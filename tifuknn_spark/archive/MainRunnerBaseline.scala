/*
In this version, we use the whole basket history to recompute group vector and hence user vector everytime when a basket
is decremented.
 */
package tifuknn

//import breeze.linalg.sum

import com.github.tototoshi.csv._
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.streaming._
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, SparkSession}

import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

object MainRunnerBaseline {
  val cleanUpState = true
  // setting some hyper parameters
  val groupSize = 2
  val rb = 0.9
  val rg = 0.7
  val basePath: String = "eval_results/baseline_inc_baskets10";
  val dataPath =
    "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/jsondata/baskets10"
  val vocabPath =
    "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/jsondata/vocabulary10.csv"
  val checkpointLocation: String = "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/checkpoint_dec_recompute"
  var batchOutputevalFile: String = "";
  var stateUpdateEvalFile: String = "";
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
                        var countOfBaskets: Long,
                        var countOfGroups: Long,
                        var allBasketGroups: Array[Array[InputBasketRow]], // group composition encoded in the structure
                        var allGroupVectors: Array[Array[Double]],
                        var userVector: Array[Double]
                      )


  def main(args: Array[String]): Unit = {
    if (cleanUpState) {
      println(s"Cleaning up state directory $checkpointLocation")
      FileUtils.cleanDirectory(new File(checkpointLocation))
    }
    // the default is 200 that uis used for shuffling data for joins and aggregations, which might sometimes
    // turn unnecessary, because you will have to wait for 200 tasks to finish.
    // notice that in my local computer, a small shuffle partition significantly speeds up...
    // notice that the state in checkpoint root dir, which follows state/<opearatorId>/<partitionId>/<1-N>.delta
    // also stores only 1 parition instead of 200!
    // for why the parition number：
    // https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-performance-tuning-groupBy-aggregation.html
    spark.sqlContext.setConf("spark.sql.shuffle.partitions", "1")

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
    val currentBasketGroups = currentState.allBasketGroups
    for ((groupBaskets, i_) <- currentBasketGroups.zipWithIndex) {
      for ((basket, j_) <- groupBaskets.zipWithIndex)
        if (customerId == basket.customerId && orderId == basket.orderId) {
          i = i_
          j = j_
          //          return Seq(i, j) // we find the first match then break out, or comment it to find the last match
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

  def computeDecayedMean(input: Array[Array[Double]], r: Double): Array[Double] = {
    val decayWeights: Array[Double] =
      (input.length - 1 until -1 by -1)
        .map(x => scala.math.pow(r, x))
        .toArray
    // recompute the group vector for all baskets within the group
    val mutiplied: Array[Array[Double]] = myArrayMultiply(input, decayWeights)
    val dotProduct: Array[Double] = myArraySum(mutiplied)
    val decayedMean: Array[Double] = myMultiply(dotProduct, 1.0 / input.length)
    decayedMean
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
        val newAllGroupVectors = currentState.allGroupVectors ++ Array(input.basketOneHot)
        // recompute the new user vector
        val newUserVector = computeDecayedMean(newAllGroupVectors, rg)
        // update state
        val newAllBasketGroups = currentState.allBasketGroups ++ Array(Array(input))
        val newCountOfGroups = currentCountOfGroups + 1
        val newState = currentState.copy(
          countOfBaskets = newCountOfBaskets,
          countOfGroups = newCountOfGroups,
          userVector = newUserVector,
          allGroupVectors = newAllGroupVectors,
          allBasketGroups = newAllBasketGroups
        )
        println("------Finished processing new basket-------")
        newState
      } else {
        println(s"Incremental 2: updating last group...")
        // scenario 2: the group length is less than group size, last group will be updated
        val newLastGroupOfBaskets = currentState.allBasketGroups.last ++ Array(input)
        val basketsGroupUnderImpact: Array[Array[Double]] = newLastGroupOfBaskets.map(x => x.basketOneHot) // add the new basket to last group
        val newLastGroupSize = currentLastGroupSize + 1
        // compute updated group vector
        val newLastGroupVector = computeDecayedMean(basketsGroupUnderImpact, rb)

        val newAllGroupVectors = currentState.allGroupVectors.updated(currentState.allGroupVectors.length - 1, newLastGroupVector)
        val newAllBasketGroups = currentState.allBasketGroups.updated(currentState.allBasketGroups.length - 1, newLastGroupOfBaskets)
        // compute new user vector
        val newUserVector = computeDecayedMean(newAllGroupVectors, rg)
        // update state
        val newState = currentState.copy(
          countOfBaskets = newCountOfBaskets,
          userVector = newUserVector,
          allGroupVectors = newAllGroupVectors,
          allBasketGroups = newAllBasketGroups
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
        /*
        * Let's first update the group vector for the target basket
        */
        // remove the basket from group
        val basketsGroupUnderImpact: Array[Array[Double]] = currentState.allBasketGroups(i).map(x => x.basketOneHot)
        // remove the target basket
        val newBasketsGroupUnderImpact = basketsGroupUnderImpact.patch(j, Nil, 1)
        // decay weights, create a decay weights array [r^(length -1), ..., r^0]
        val newGroupVector: Array[Double] = computeDecayedMean(newBasketsGroupUnderImpact, rb)

        // update the new group vector
        val newAllGroupVectors = currentState.allGroupVectors
        newAllGroupVectors(i) = newGroupVector
        /*
        * Let's then update the user vector with the updated group vector included
        */
        val newUserVector: Array[Double] = computeDecayedMean(newAllGroupVectors, rg)

        val newAllBasketGroups = currentState.allBasketGroups
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

          val newUserVector = computeDecayedMean(newAllGroupVectors, rg)
          // perhaps this is more numerically stable, not sure.
          //          val newUserVectorAlt = mySum(myMultiply(currentUserVector, currentState.countOfGroups/(newCountOfGroups * rg)),
          //          myMultiply(dotProduct, 1.0/(newCountOfGroups * rg)))


          // the vector value must be less than 1, give some slack for error
          //          assert(newUserVector(0) <= 1.1)

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
    streamOutputData.select("rg", "rb", "groupSize", "countOfBaskets", "countOfGroups", "userVector").show(false)
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

    println(s"Time(ms) to output micro-batch: $duration")
    println(s"------Finished outputing micro-batch------")

  }

}

