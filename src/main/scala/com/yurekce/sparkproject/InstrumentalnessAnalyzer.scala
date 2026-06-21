package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, count}

object InstrumentalnessAnalyzer {

  def instrumentalnessAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .agg(
        avg("instrumentalness").as("avg_instrumentalness"),
        count("instrumentalness").as("num_songs")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/instrumentalnessData")

    processedDf.show(1000, truncate = false)
  }
}