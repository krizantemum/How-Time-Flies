package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame

object TempoAnalyzer {
  def TempoAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .avg("tempo")
      .withColumnRenamed("avg(tempo)", "avg_tempo")
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
      .save("csvFiles/tempoData")

    processedDf.show(truncate = false)
  }

}
