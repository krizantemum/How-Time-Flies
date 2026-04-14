package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame

object DanceabilityAnalyzer {

  def danceabilityAverageByYear(df: DataFrame): DataFrame = {
    df
      .groupBy("year")
      .avg("danceability")
      .withColumnRenamed("avg(danceability)", "avg_danceability")
      .join(
        df.groupBy("year").count().withColumnRenamed("count", "num_songs"),
        Seq("year")
      )
      .orderBy("year")
  }
}
