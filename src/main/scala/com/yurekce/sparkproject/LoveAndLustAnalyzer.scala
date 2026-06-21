package com.yurekce.sparkproject
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object LoveAndLustAnalyzer {

  val loveWords = List("heart", "love", "loved", "loving", "honey", "stay", "angel", "trust", "promise", "forever",
    "miracle", "darling", "beautiful", "cute")
  val lustWords = List("body", "bodies", "kiss", "kissing", "kissed", "touch", "touching", "touched", "lip", "lips",
    "pussy", "taste", "tasted", "tasting", "fuck", "fucked", "fucking")

  val loveRegex = "(?i)\\b(" + loveWords.mkString("|") + ")\\b"
  val lustRegex = "(?i)\\b(" + lustWords.mkString("|") + ")\\b"

  def analyzeLyrics(df: DataFrame): DataFrame = {
    df.select(
      col("genre"),
      col("year"),
      col("lyrics"),
      // Find all love/lust words and count the size of the resulting array
      size(regexp_extract_all(col("lyrics"), lit(loveRegex), lit(0))).as("love_count"),
      size(regexp_extract_all(col("lyrics"), lit(lustRegex), lit(0))).as("lust_count"))

  }

  def getWordStats(df: DataFrame): Unit = {
    // First, add the count columns to the DF
    val countedDf = df.withColumn("love_count", size(regexp_extract_all(col("lyrics"), lit(loveRegex),
      lit(0))))
      .withColumn("lust_count", size(regexp_extract_all(col("lyrics"), lit(lustRegex), lit(0))))

    countedDf.createOrReplaceTempView("lyric_stats")


    // Use SQL to sum them up by genre
    val summaryDf = df.sparkSession.sql(
      """
      SELECT
        genre,
        SUM(love_count) as total_love_words,
        SUM(lust_count) as total_lust_words,
        AVG(love_count) as avg_love_per_song,
        AVG(lust_count) as avg_lust_per_song
      FROM lyric_stats
      GROUP BY genre
      ORDER BY total_love_words DESC
    """)

    summaryDf.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/loveAndLust")

    val loveLustYearDF = df.sparkSession.sql(
      """
          SELECT
            year,
            SUM(love_count) as total_love_words,
            SUM(lust_count) as total_lust_words,
            AVG(love_count) as avg_love_per_song,
            AVG(lust_count) as avg_lust_per_song
          FROM lyric_stats
          GROUP BY year
          ORDER BY year DESC
        """)

    loveLustYearDF.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/loveLustYear")

    summaryDf.show()
  }
}
