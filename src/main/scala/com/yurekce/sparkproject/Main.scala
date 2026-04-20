package com.yurekce.sparkproject

import com.yurekce.sparkproject.config.SparkConfig
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.storage.StorageLevel

object Main {
  def main(args: Array[String]): Unit = {
    println("Available cores: " + Runtime.getRuntime.availableProcessors())

    val bean = java.lang.management.ManagementFactory.getOperatingSystemMXBean
      .asInstanceOf[com.sun.management.OperatingSystemMXBean]
    println(f"Fiziksel Toplam RAM: ${bean.getTotalPhysicalMemorySize / (1024.0 * 1024 * 1024)}%.2f GB")

    val spark = SparkConfig.createSession()

    // Single scan — clean and cache in one shot, no raw count
    val healthyData = Filter.clean(DataLoader.load(spark, "spotify_data/songs.csv"))
      .persist(StorageLevel.MEMORY_AND_DISK)
    println("Healthy rows: " + healthyData.count()) // warms the cache

    val lyricsHealthyData = Filter.lyricsClean(healthyData)
      .persist(StorageLevel.MEMORY_AND_DISK)
    println("Lyrics rows: " + lyricsHealthyData.count()) // warms the cache

    lyricsHealthyData.show(10, truncate = false)

    /*
    LoveAndLustAnalyzer.getWordStats(
      LoveAndLustAnalyzer.analyzeLyrics(lyricsHealthyData.select("id", "year", "lyrics", "genre"))
    )
    LonelinessAnalyzer.getWordStats(
      LonelinessAnalyzer.analyzeLyrics(lyricsHealthyData.select("id", "year", "lyrics"))
    )
    */

    val lyricsChunked = Filter.popularityClean(lyricsHealthyData)
      .withColumn("chunks", LyricsCleaner.splitByFourLines(col("lyrics")))
      .withColumn("chunk", explode(col("chunks")))
      .select("id", "year", "chunk")
      .repartition(200, col("id"))
      .persist(StorageLevel.MEMORY_AND_DISK)

    println("Chunk count: " + lyricsChunked.count()) // warms BEFORE BERT runs

    // Free memory no longer needed before loading BERT
    healthyData.unpersist()
    lyricsHealthyData.unpersist()

    val takeSome = lyricsChunked.sample(50000)

    val fittedModel = EmotionProcessing.buildAndFitModel(takeSome)
    val emotionPerChunk = EmotionProcessing.run(takeSome, fittedModel)

    // Write FIRST — breaks lineage so BERT never runs twice
    emotionPerChunk
      .write
      .mode("overwrite")
      .parquet("checkpoints/emotionPerChunk")

    takeSome.unpersist()

    // All downstream work reads from parquet — BERT is done
    val savedChunks = spark.read.parquet("checkpoints/emotionPerChunk")

    savedChunks.show(10, false)

    val emotionPerSong = EmotionProcessing.aggregateBySong(savedChunks)
    val finalResult = EmotionProcessing.dominantSongEmotion(emotionPerSong)

    finalResult
      .write
      .mode("overwrite")
      .parquet("checkpoints/finalResult")

    spark.read.parquet("checkpoints/finalResult").show(10, false)
  }
}