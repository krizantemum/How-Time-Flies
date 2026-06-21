package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, col, count}

object DurationAnalyzer {

  def durationAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .withColumn("duration_sec", col("duration_ms") / 1000)
      .groupBy("year")
      .agg(
        avg("duration_sec").as("avg_duration_sec"),
        count("duration_sec").as("num_songs")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/durationData")

    processedDf.show(1000, truncate = false)
  }
}