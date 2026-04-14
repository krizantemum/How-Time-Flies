package com.yurekce.sparkproject
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataLoader {

  def load(spark: SparkSession, path: String): DataFrame = {
    spark.read
      .option("header", "true")
      .option("multiline", "true")
      .option("inferSchema", "true")
      .option("mode", "PERMISSIVE")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("encoding", "UTF-8")
      .csv(path)
  }
}
