package com.yurekce.sparkproject

object TempoOfGenresAnalyzer {

  import org.apache.spark.sql.DataFrame

  def tempoByGenre(df: DataFrame): Unit = {
    df.createOrReplaceTempView("temp_music_table")

    val tempoDf = df.sparkSession.sql(
      """
      SELECT
        genre,
        AVG(tempo) AS avg_tempo,
        STDDEV(tempo) AS tempo_std
      FROM temp_music_table
      GROUP BY genre
      ORDER BY avg_tempo DESC
    """)

    tempoDf.coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles/comparisons/tempo_by_genre")
  }
}
