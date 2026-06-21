package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, count}

object DanceabilityAnalyzer {

  def danceabilityAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .agg(
        avg("danceability").as("avg_danceability"),
        count("danceability").as("num_songs")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/danceabilityData")

    processedDf.show(1000, truncate = false)
  }
}
