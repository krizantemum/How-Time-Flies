package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame

object ValenceAnalyzer {

  def valenceAverageByYear(df: DataFrame): DataFrame = {
    df
      .groupBy("year")
      .avg("valence")
      .withColumnRenamed("avg(valence)", "avg_valence") // rename for clarity
      .join(
        df.groupBy("year").count().withColumnRenamed("count", "num_songs"),
        Seq("year")
      )
      .orderBy("year")
  }

}
