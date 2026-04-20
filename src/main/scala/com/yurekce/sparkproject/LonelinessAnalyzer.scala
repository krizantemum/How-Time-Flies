package com.yurekce.sparkproject

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit, regexp_extract_all, size}

object LonelinessAnalyzer {

  val lonelyWords = List("lonely", "loneliness", "alone", "cold", "nobody", "empty", "silence ", "silent")
  val lonelyRegex = "(?i)\\b(" + lonelyWords.mkString("|") + ")\\b"

  def analyzeLyrics(df: DataFrame): DataFrame = {
    df.select(
      col("year"),
      col("lyrics"),
      size(regexp_extract_all(col("lyrics"), lit(lonelyRegex), lit(0))).as("lonely_count"))
  }

  def getWordStats(df: DataFrame): Unit = {
    val countedDf = df.withColumn("lonely_count", size(regexp_extract_all(col("lyrics"),
      lit(lonelyRegex), lit(0))))

    countedDf.createOrReplaceTempView("lyric_stats")

    val summaryDf = df.sparkSession.sql(
      """
      SELECT
        year,
        SUM(lonely_count) as total_lonely_words,
        AVG(lonely_count) as avg_lonely_per_song
      FROM lyric_stats
      GROUP BY year
      ORDER BY year DESC
    """)

    summaryDf.coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/lonely")

    summaryDf.show()
  }
}
