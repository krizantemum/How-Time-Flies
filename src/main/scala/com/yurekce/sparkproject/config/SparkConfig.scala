package com.yurekce.sparkproject.config
import org.apache.spark.sql.SparkSession

object SparkConfig {
  def createSession(): SparkSession = {
    SparkSession.builder()
      .appName("HowLoveFlies")
      .master("local[7]") // monster has 8
      .config("spark.driver.memory", "12g")
      .config("spark.driver.maxResultSize", "10g")
      .config("spark.sql.shuffle.partitions", "14")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "2000M")
      .config("spark.memory.fraction", "0.6")
      .config("spark.memory.storageFraction", "0.3")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:6.3.3")
      .getOrCreate()
  }
}