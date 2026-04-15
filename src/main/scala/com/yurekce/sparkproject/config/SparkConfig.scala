package com.yurekce.sparkproject.config
import org.apache.spark.sql.SparkSession

object SparkConfig {
  def createSession(): SparkSession = {
    SparkSession.builder()
      .appName("HowLoveFlies")
      .master("local[7]") // monster has 8
      .config("spark.driver.memory", "12g")
      .config("spark.driver.maxResultSize", "10g")
      .config("spark.sql.shuffle.partitions", "10")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.4")
      .getOrCreate()
  }
}