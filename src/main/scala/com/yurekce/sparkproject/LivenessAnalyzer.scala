package com.yurekce.sparkproject
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, count}

object LivenessAnalyzer {

  def livenessAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .agg(
        avg("liveness").as("avg_liveness"),
        count("liveness").as("num_songs")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/livenessData")

    processedDf.show(1000, truncate = false)
  }
}
