package com.yurekce.sparkproject

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import com.johnsnowlabs.nlp.annotators.classifier.dl.BertForSequenceClassification
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

object ExampleEmotion {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ExampleEmotion")
      .master("local[7]")
      .getOrCreate()

    import spark.implicits._

    // -----------------------------
    // 1. Sample Data (lyrics)
    // -----------------------------
    val df = Seq(
      "I'll meet you at the divide To break the spell",
      "A point where two worlds collide Yeah, we'll rebel",
      "And we run, and we run, and we run, and we run And we run, and we run, and we run Until we break through",
      "If I get high enough If I get high enough Will I see you again?",
      "I fill my lungs every night Not long to wait",
      "And if I do this thing right I dream of our escape",
      "Oh, and we run, and we run, and we run, and we run And we run, and we run, and we run Until we break through",
      "If I get high enough If I get high enough Will I see you again? Will I see you again?",
    ).toDF("text")

    // -----------------------------
    // 2. Split lyrics into lines
    // -----------------------------
    val dfSplit = df
      .withColumn("text", explode(split($"text", "\n")))
      .filter(length($"text") > 0)

    // -----------------------------
    // 3. NLP Pipeline
    // -----------------------------
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val classifier = BertForSequenceClassification
      .pretrained("bert_sequence_classifier_emotion", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("emotion")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, classifier))

    // -----------------------------
    // 4. Run pipeline
    // -----------------------------
    val model = pipeline.fit(dfSplit)
    val result = model.transform(dfSplit)

    // -----------------------------
    // 5. Show results
    // -----------------------------
    result.select(
      $"text",
      $"emotion.result".as("emotion"),
      $"emotion.metadata"
    ).show(false)

    spark.stop()
  }
}