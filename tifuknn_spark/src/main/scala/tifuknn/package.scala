import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, BooleanType, FloatType, LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Encoder, Encoders, SQLContext, SparkSession}

// package object <package_name> is a way to share defnitions within a package
// it is often named "package.scala"
// see https://stackoverflow.com/questions/3400734/package-objects
package object tifuknn {


  // input structure after one hot encoding
  case class InputBasketRow(
                             customerId: Long,
                             orderId: Long,
                             basketOneHot: Array[Double],
                             isDeletion: Boolean
                           )

  //define data structures
  // schema for the json input data
  val basketSchema = new StructType(Array(
    StructField("customerId", LongType),
    StructField("orderId", LongType),
    StructField("basket", ArrayType(StringType)),
    StructField("isDeletion", BooleanType)
  ))

  //val historyScehma = new StructType(Array(
  //    StructField("customerId", LongType),
  //    StructField("allBasketsGroups", ArrayType(ArrayType(InputBasketRow))),
  //    StructField("allGroupVectors", ArrayType(ArrayType(FloatType)))
  //  ))

  val hist = new StructType()
    .add("customerId", LongType)
    .add("allBasketsGroups", ArrayType(
      ArrayType(
        new StructType()
          .add("customerId", LongType)
          .add("orderId", LongType)
          .add("basketOneHot", ArrayType(FloatType))
          .add("isDeletion", BooleanType)
      )
    ))


  implicit val basketEncoder: Encoder[InputBasketRow] = Encoders.product[InputBasketRow]

  def mySum(x: Array[Double], y: Array[Double]): Array[Double] = {
    x.zip(y).map(ele => ele._1 + ele._2)
  }

  def mySumDecimal(x: Array[BigDecimal], y: Array[BigDecimal]): Array[BigDecimal] = {
    x.zip(y).map(ele => ele._1 + ele._2)
  }

  def mySubstract(x: Array[Double], y: Array[Double]): Array[Double] = {
    x.zip(y).map(ele => ele._1 - ele._2)
  }

  def myMultiply(x: Array[Double], y: Double): Array[Double] = {
    x.map(x => x * y)
  }
  def myMultiplyDecimal(x: Array[BigDecimal], y: BigDecimal): Array[BigDecimal] = {
    x.map(x => x * y)
  }

  def myArrayMultiply(x: Array[Array[Double]], y: Array[Double]): Array[Array[Double]] = {
    x.zip(y).map(ele => myMultiply(ele._1, ele._2))
  }

  def myArrayMultiplyDecimal(x: Array[Array[BigDecimal]], y: Array[BigDecimal]): Array[Array[BigDecimal]] = {
    x.zip(y).map(ele => myMultiplyDecimal(ele._1, ele._2))
  }

  def myArraySubtract(x: Array[Array[Double]], y: Array[Array[Double]]): Array[Array[Double]] = {
    x.zip(y).map(ele => mySum(ele._1, myMultiply(ele._2, -1.0)))
  }

  def myArraySubtractDecimal(x: Array[Array[BigDecimal]], y: Array[Array[BigDecimal]]): Array[Array[BigDecimal]] = {
    x.zip(y).map(ele => mySumDecimal(ele._1, myMultiplyDecimal(ele._2,  BigDecimal("-1.0"))))
  }

  def myArraySum(x: Array[Array[Double]]): Array[Double] = {
    var temp = Array.ofDim[Double](x(0).length)
    for (ele <- x) {
      temp = mySum(temp, ele)
    }
    temp
  }

  def myArraySumDecimal(x: Array[Array[BigDecimal]]): Array[BigDecimal] = {
    var temp = Array.fill[BigDecimal](x(0).length)(0)
    for (ele <- x) {
      temp = mySumDecimal(temp, ele)
    }
    temp
  }

}

