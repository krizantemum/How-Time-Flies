package com.yurekce.sparkproject

import com.yurekce.sparkproject.config.SparkConfig
import org.apache.spark.sql.functions.{avg, col, explode, max, min}
import org.apache.spark.storage.StorageLevel
import scala.sys.process._

object Main {
  
   def speak(text: String): Unit = {
    val escaped = text.replace("'", "''")

    Seq(
      "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
      "-Command",
        s"Add-Type -AssemblyName System.Speech; " +
        s"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('$escaped')"
      ).!
  }

  def main(args: Array[String]): Unit = {
    speak("Starting the Spark job. This may take a while.")

    println("Available cores: " + Runtime.getRuntime.availableProcessors())

    val bean = java.lang.management.ManagementFactory.getOperatingSystemMXBean
      .asInstanceOf[com.sun.management.OperatingSystemMXBean]
    println(f"Fiziksel Toplam RAM: ${bean.getTotalPhysicalMemorySize / (1024.0 * 1024 * 1024)}%.2f GB")

    val spark = SparkConfig.createSession()

    try {
      val healthyData = Filter.clean(DataLoader.load(spark, "spotify_data/songs.csv"))
        .persist(StorageLevel.MEMORY_AND_DISK)
      println("Healthy rows: " + healthyData.count()) // warms the cache

      // Per-feature averages by year. Each writes to its own csvFiles/<feature>Data folder.
      DanceabilityAnalyzer.danceabilityAverageByYear(healthyData)
      EnergyAnalyzer.energyAverageByYear(healthyData)
      LoudnessAnalyzer.loudnessAverageByYear(healthyData)
      SpeechinessAnalyzer.speechinessAverageByYear(healthyData)
      AcousticnessAnalyzer.acousticnessAverageByYear(healthyData)
      InstrumentalnessAnalyzer.instrumentalnessAverageByYear(healthyData)
      LivenessAnalyzer.livenessAverageByYear(healthyData)
      ValenceAnalyzer.valenceAverageByYear(healthyData)
      TempoAnalyzer.TempoAverageByYear(healthyData)
      DurationAnalyzer.durationAverageByYear(healthyData)

      // Cluster years on (avg_valence, avg_energy); K auto-picked by elbow.
      YearMoodClusterer.runAll(healthyData)

      // Cluster years on their genre mix; K auto-picked by elbow.
      YearGenreClusterer.runAll(healthyData)

      // Recommender disabled for the full BERT run — not needed for emotion analysis.
      // // Offline: fit the LSH pipeline once and cache the transformed catalog.
      // val (annModel, annCatalogRaw) = RecommenderANN.build(healthyData)
      // val annCatalog = annCatalogRaw.persist(StorageLevel.MEMORY_AND_DISK)
      // println("ANN catalog rows: " + annCatalog.count()) // forces the build
      //
      // // Online: each call is an approxNearestNeighbors query against the cache.
      // RecommenderANN
      //   .recommend(annModel, annCatalog, seedId = "36G7q9VtOVEYhuALFcO04W", topK = 5)
      //   .show(false)
      //
      // annCatalog.unpersist()

      /*
      // lyricsClean flattens newlines to spaces — useful for the word-stats analyzers
      // below, but it BREAKS splitByFourLines (which needs \n to chunk on).
      // For the BERT path we feed healthyData directly; splitByFourLines does its
      // own \\n -> \n normalization.
      val lyricsHealthyData = Filter.lyricsClean(healthyData)
        .persist(StorageLevel.MEMORY_AND_DISK)
      println("Lyrics rows: " + lyricsHealthyData.count())

      LoveAndLustAnalyzer.getWordStats(
        LoveAndLustAnalyzer.analyzeLyrics(lyricsHealthyData.select("id", "year", "lyrics", "genre"))
      )
      LonelinessAnalyzer.getWordStats(
        LonelinessAnalyzer.analyzeLyrics(lyricsHealthyData.select("id", "year", "lyrics"))
      )

      lyricsHealthyData.unpersist()
      */

      // FULL RUN with per-year resumable checkpoints.
      // Each year is written to checkpoints/emotionPerChunk/year=YYYY/ with a
      // _SUCCESS marker. On restart, completed years are skipped; pending years
      // (including any that crashed mid-write) are re-run from scratch.
      val outputBase = "checkpoints/emotionPerChunk"
      new java.io.File(outputBase).mkdirs()

      val allYears: Array[Int] = healthyData
        .select("year").distinct()
        .collect()
        .map(_.getInt(0))
        .sorted

      def yearDone(y: Int): Boolean =
        new java.io.File(s"$outputBase/year=$y/_SUCCESS").exists()

      val (doneYears, pendingYears) = allYears.partition(yearDone)
      println(s"[run] Years total=${allYears.length}  " +
        s"done=${doneYears.length}  pending=${pendingYears.length}")
      if (doneYears.nonEmpty)
        println(s"[run] Resuming. Already done: ${doneYears.mkString(", ")}")
      if (pendingYears.nonEmpty)
        println(s"[run] To process: ${pendingYears.mkString(", ")}")

      if (pendingYears.nonEmpty) {
        // Build chunked dataset for pending years only.
        val pendingSeq: Seq[Any] = pendingYears.toSeq.map(_.asInstanceOf[Any])
        val lyricsChunked = healthyData
          .filter(col("year").isin(pendingSeq: _*))
          .withColumn("chunks", LyricsCleaner.splitByFourLines(col("lyrics")))
          .withColumn("chunk", explode(col("chunks")))
          .select("id", "year", "genre", "niche_genres", "chunk")
          .persist(StorageLevel.MEMORY_AND_DISK)
        val pendingChunkCount = lyricsChunked.count()
        println(s"[run] Pending chunks (with duplicates): $pendingChunkCount")

        healthyData.unpersist()

        // Fit BERT pipeline ONCE — reused across all years.
        val tModelStart = System.nanoTime()
        val fittedModel = EmotionProcessing.buildAndFitModel(lyricsChunked.limit(1))
        val modelLoadSec = (System.nanoTime() - tModelStart) / 1e9
        println(f"[run] Model load + fit: $modelLoadSec%.2f s")

        val tRunStart = System.nanoTime()
        pendingYears.zipWithIndex.foreach { case (y, idx) =>
          speak(s"Starting processing for year $y. This may take a while.")
          val tYearStart = System.nanoTime()
          val yearChunked = lyricsChunked.filter(col("year") === y)
          val distinctY = yearChunked.select("chunk").distinct().coalesce(1)
          val emotionY = EmotionProcessing.run(distinctY, fittedModel)
          val joinedY = yearChunked.join(emotionY, Seq("chunk")).drop("chunk")

          // Per-year write. mode("overwrite") wipes any partial dir from a
          // prior crash, then writes fresh files + _SUCCESS marker.
          joinedY.write.mode("overwrite").parquet(s"$outputBase/year=$y")

          val secs = (System.nanoTime() - tYearStart) / 1e9
          val elapsed = (System.nanoTime() - tRunStart) / 1e9
          val remaining = if (idx + 1 < pendingYears.length)
            elapsed / (idx + 1) * (pendingYears.length - idx - 1) else 0.0
          println(f"[run] year=$y  done in ${secs}%.1f s  " +
            f"(${idx + 1}/${pendingYears.length}, " +
            f"elapsed ${elapsed / 60}%.1f min, " +
            f"ETA ${remaining / 60}%.1f min)")
            speak(s"Completed year $y. ${pendingYears.length - idx - 1} years to go.")
        }
        val totalRunSec = (System.nanoTime() - tRunStart) / 1e9
        println(f"[run] BERT inference complete. Total: ${totalRunSec / 60}%.1f min " +
          f"(${totalRunSec / 3600}%.2f h)")

        lyricsChunked.unpersist()
      } else {
        println("[run] All years already done — skipping BERT inference.")
        healthyData.unpersist()
      }

      // Each per-year write keeps `year` as a real column AND lands in a
      // year=YYYY directory. Reading the base dir would make Spark treat
      // year=YYYY as a Hive partition, colliding with the in-data `year`
      // column ("Found duplicate column(s) in the data schema and the
      // partition schema: year"). Passing the leaf parquet files directly
      // disables partition discovery, so the in-data `year` column is used
      // as-is — and the already-computed year checkpoints are read, never
      // recomputed.
      val parquetFiles: Seq[String] = {
        def leaves(f: java.io.File): Seq[String] =
          if (f.isDirectory) f.listFiles().toSeq.flatMap(leaves)
          else if (f.getName.endsWith(".parquet")) Seq(f.getPath.replace('\\', '/'))
          else Seq.empty
        leaves(new java.io.File(outputBase))
      }
      val savedChunks = spark.read.parquet(parquetFiles: _*)
      println(s"[run] Reading all years. Total chunk rows: ${savedChunks.count()}")

      val emotionPerSong = EmotionProcessing.aggregateBySong(savedChunks)
      val finalResult = EmotionProcessing.dominantSongEmotion(emotionPerSong)

      // [verify-6] per-song aggregation: counts + song_emotion + genre/niche carried.
      println("=== [verify-6] finalResult schema ===")
      finalResult.printSchema()
      println("=== [verify-6] sample songs with per-emotion counts and dominant ===")
      finalResult.show(10, false)

      // [verify-7] sanity: distribution of song_emotion across the full corpus.
      println("=== [verify-7] song_emotion distribution across all songs ===")
      finalResult.groupBy("song_emotion").count().orderBy(col("count").desc).show(false)

      finalResult
        .write
        .mode("overwrite")
        .parquet("checkpoints/finalResult")
    } finally {
      spark.stop()
    }
  }
}