package com.yurekce.sparkproject.config
import org.apache.spark.sql.SparkSession

object SparkConfig {
  def createSession(): SparkSession = {
    SparkSession.builder()
      .appName("HowLoveFlies")
      .master("local[8]") // monster has 8
      .config("spark.driver.memory", "3g")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.4")
      .getOrCreate()
  }
}