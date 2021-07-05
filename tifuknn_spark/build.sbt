name := "tifuknn"

version := "1.0"

scalaVersion := "2.12.8"
val sparkVersion = "2.4.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)
libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.3.7"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % Test


initialCommands in console := s"""
val sc = new org.apache.spark.SparkContext("local", "shell")
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val spark = org.apache.spark.sql.SparkSession
  .builder
  .appName("TIFUKNN")
  .master("local[*]")
  .getOrCreate()
"""