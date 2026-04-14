package com.yurekce.sparkproject
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.annotator.ClassifierDLModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Pipeline

object LyricsParser {
  def lyricsToDataFrame(spark: SparkSession, lyrics: String): DataFrame = {
    import spark.implicits._

    val lines = lyrics
      .split("\r?\n")
      .map(_.trim)
      .filter(_.nonEmpty)

    lines.toSeq.toDF("text")
  }
}
