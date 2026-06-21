package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, count}

object SpeechinessAnalyzer {

  def speechinessAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .agg(
        avg("speechiness").as("avg_speechiness"),
        count("speechiness").as("num_songs")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/speechinessData")

    processedDf.show(1000, truncate = false)
  }
}