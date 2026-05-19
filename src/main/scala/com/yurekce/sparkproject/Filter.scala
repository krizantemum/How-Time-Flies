package com.yurekce.sparkproject
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


object Filter {
  def clean(df: DataFrame): DataFrame = {
    df.filter("year >= 1990 AND year <= 2022 AND valence >= 0 AND valence <= 1")
      .na.drop("any",Seq("valence", "lyrics", "genre", "year", "id", "popularity"))
  }
  def lyricsClean(df: DataFrame): DataFrame = {

    val cleanNewLinesUDF = udf((text: String) => {
      if (text == null || text.trim.isEmpty) {
        ""
      } else {
        text
          .replace("\\n", "\n") // fix escaped newlines
          .split("\n")          // split into lines
          .map(_.trim)
          .filter(_.nonEmpty)   // remove empty lines
          .mkString(" ")        // Join with a single space to make a "clean" paragraph
      }
    })

    // 2. Apply the UDF and return the DataFrame
    df.withColumn("lyrics", cleanNewLinesUDF(col("lyrics")))
  }
  def popularityClean(df: DataFrame): DataFrame = {
    val filteredDf = df.filter(col("popularity") > 0)
    println("yo bigger 0 " + filteredDf.count())
    filteredDf
  }
}
