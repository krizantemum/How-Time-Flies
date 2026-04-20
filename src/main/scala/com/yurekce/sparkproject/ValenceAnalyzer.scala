package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame

object ValenceAnalyzer {

  def valenceAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .avg("valence")
      .withColumnRenamed("avg(valence)", "avg_valence")
      .join(
        df.groupBy("year").count().withColumnRenamed("count", "num_songs"),
        Seq("year")
      )
      .orderBy("year")

    processedDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/valenceData")

    processedDf.show(truncate = false)
  }

}
