package com.yurekce.sparkproject

import com.yurekce.sparkproject.ValenceAnalyzer._
import com.yurekce.sparkproject.DanceabilityAnalyzer._
import com.yurekce.sparkproject.config.SparkConfig
import org.apache.spark.sql.functions.{col, explode, udf}
import org.apache.spark.storage.StorageLevel

object Main {
  def main(args: Array[String]): Unit = {
    println("Available cores: " + Runtime.getRuntime.availableProcessors())

    val bean = java.lang.management.ManagementFactory.getOperatingSystemMXBean
      .asInstanceOf[com.sun.management.OperatingSystemMXBean]

    val totalRAM = bean.getTotalPhysicalMemorySize / (1024.0 * 1024 * 1024)
    println(f"Fiziksel Toplam RAM: $totalRAM%.2f GB")

    val path = "spotify_data/songs.csv"

    val spark = SparkConfig.createSession()
    val data = DataLoader.load(spark, path)

    println("how many data " + data.count())

    val healthyData = Filter.clean(data).cache()
    println(healthyData.count())


    /*
    val danceabilityData = healthyData.select("id", "year", "danceability")
    DanceabilityAnalyzer.danceabilityAverageByYear(danceabilityData)

    val livenessData = healthyData.select("id", "year", "liveness")
    LivenessAnalyzer.livenessAverageByYear(livenessData)

    val valenceData = healthyData.select("id", "year", "valence")
    ValenceAnalyzer.valenceAverageByYear(valenceData)

    val tempoData = healthyData.select("id", "year", "tempo")
    TempoAnalyzer.TempoAverageByYear(tempoData)

    val tempoGenre = healthyData.select("id", "genre", "tempo")
    TempoOfGenresAnalyzer.tempoByGenre(tempoGenre)

    val loveData = healthyData.select("id", "genre", "lyrics")
    LoveAndLustAnalyzer.getWordStats(LoveAndLustAnalyzer.analyzeLyrics(loveData))

     */

    val lonelinessData = healthyData.select("id", "year", "lyrics")
    LonelinessAnalyzer.getWordStats(LonelinessAnalyzer.analyzeLyrics(lonelinessData))




    val lyricsChunked = healthyData
      .withColumn("chunks", LyricsCleaner.splitByFourLines(col("lyrics")))
      .withColumn("chunk", explode(col("chunks")))
      .select("id", "year", "chunk")
      .persist(StorageLevel.MEMORY_AND_DISK)


  /*
    val fittedModel = EmotionProcessing.buildAndFitModel(lyricsChunked)
    val emotionPerChunk = EmotionProcessing.run(lyricsChunked, fittedModel)

    println("incoming count of how many chunks may universe save us")
    println(lyricsChunked.count())

   */

    /*
    emotionPerChunk
      .write
      .mode("overwrite")
      .parquet("checkpoints/emotionPerChunk")

     */

    /*

    val num = 10000
    emotionPerChunk.show(num, false)

    // Full aggregation pipeline
    val emotionPerSong = EmotionProcessing.aggregateBySong(emotionPerChunk.limit(num))
    emotionPerSong.show(truncate = false)
    val finalResult = EmotionProcessing.dominantSongEmotion(emotionPerSong)
    finalResult.show(num, false)

     */
  }
}
