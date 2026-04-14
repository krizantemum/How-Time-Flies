package com.yurekce.sparkproject
import org.apache.spark.sql.DataFrame


object Filter {
  def clean(df: DataFrame): DataFrame = {
    df.filter("year >= 1990 AND year <= 2022 AND valence >= 0 AND valence <= 1")
      .na.drop("any",Seq("valence", "lyrics", "genre", "year", "id"))
  }
}
