package com.yurekce.sparkproject

import com.yurekce.sparkproject.ValenceAnalyzer._
import com.yurekce.sparkproject.DanceabilityAnalyzer._
import com.yurekce.sparkproject.config.SparkConfig

object Main {
  def main(args: Array[String]): Unit = {
    println("Available cores: " + Runtime.getRuntime.availableProcessors())
    println("Available RAM: " + Runtime.getRuntime.maxMemory() / 1024 / 1024 / 1024.0)

    val path = "spotify_data/songs.csv"

    val spark = SparkConfig.createSession()
    val data = DataLoader.load(spark, path)

    println("how many data" + data.count())

    val healthyData = Filter.clean(data)
    println(healthyData.count())

    val valenceData = healthyData.select("id", "year", "valence")

    val valenceByYear = valenceAverageByYear(valenceData)
    valenceByYear.show(Int.MaxValue, false)
    valenceByYear.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("csvFiles")

    val danceabilityData = healthyData.select("id", "year", "danceability")

    val danceabilityByYear = danceabilityAverageByYear(danceabilityData)
    danceabilityByYear.show(Int.MaxValue, false)

    danceabilityByYear.write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save("generatedDate/danceabilityData")

  }
}
