package com.yurekce.sparkproject
import org.apache.spark.sql.DataFrame

object LivenessAnalyzer {

  def livenessAverageByYear(df: DataFrame): Unit = {
    val processedDf = df
      .groupBy("year")
      .avg("liveness")
      .withColumnRenamed("avg(liveness)", "avg_liveness")
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
      .save("csvFiles/livenessData")

    processedDf.show(truncate = false)
  }
}
