package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, count}

object TempoAnalyzer {
  def TempoAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .agg(
        avg("tempo").as("avg_tempo"),
        count("tempo").as("num_songs")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/tempoData")

    processedDf.show(1000, truncate = false)
  }

}
