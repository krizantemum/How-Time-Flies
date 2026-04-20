package com.yurekce.sparkproject

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.classifier.dl.BertForSequenceClassification
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object EmotionProcessing {

  // Fix 1: build and fit the pipeline ONCE, return the fitted model
  def buildAndFitModel(df: DataFrame, inputCol: String = "chunk"): PipelineModel = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol(inputCol)
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val classifier = BertForSequenceClassification
      .pretrained("bert_sequence_classifier_emotion", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("emotion")
      .setBatchSize(32) // process 32 chunks at once instead of 1
      .setMaxSentenceLength(128) // lyrics chunks won't exceed 128 tokens — saves memory


    new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, classifier))
      .fit(df)
  }

  private val dominantEmotionUDF = udf((metadata: Map[String, String]) => {
    if (metadata == null || metadata.isEmpty) null.asInstanceOf[String]
    else {
      val emotionKeys = Set("joy", "sadness", "anger", "love", "fear")
      metadata
        .filter { case (k, _) => emotionKeys.contains(k) }
        .map { case (k, v) => (k, scala.util.Try(v.toFloat).getOrElse(0f)) }
        .maxBy(_._2)
        ._1
    }
  })

  // Fix 1 cont.: accept a pre-fitted PipelineModel instead of building inside run()
  def run(df: DataFrame, model: PipelineModel, inputCol: String = "chunk"): DataFrame = {
    val result = model.transform(df)

    val emotions = Seq("joy", "sadness", "anger", "love", "fear")

    // Fix 3: extract each emotion's confidence as its own float column
    val withConfidences = emotions.foldLeft(
      result.select(
        col("id"),
        col("year"),
        col(inputCol).as("chunk"),
        col("emotion.result").getItem(0).as("emotion_label"),
        col("emotion.metadata").getItem(0).as("metadata")
      )
    ) { (df, emotionName) =>
      df.withColumn(
        s"conf_$emotionName",
        col("metadata").getItem(emotionName).cast("float")
      )
    }

    // Add dominant_emotion column using the UDF
    withConfidences
      .withColumn("dominant_emotion", dominantEmotionUDF(col("metadata")))
      .drop("metadata")
  }

  def aggregateBySong(df: DataFrame): DataFrame = {
    // Pivot to get count of each dominant emotion per song
    df.groupBy("id", "year")
      .pivot("dominant_emotion", Seq("joy", "sadness", "anger", "love", "fear"))
      .count()
      .na.fill(0)
  }

  // Derives the overall dominant emotion for each song
  def dominantSongEmotion(df: DataFrame): DataFrame = {
    val emotions = Seq("joy", "sadness", "anger", "love", "fear")
    val cols = emotions.map(col)

    val pickDominantLabel = udf((counts: Seq[Long]) => {
      emotions(counts.indexOf(counts.max))
    })

    df.withColumn(
      "song_emotion",
      pickDominantLabel(array(cols: _*))
    )
  }
}