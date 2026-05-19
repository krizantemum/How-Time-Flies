package com.yurekce.sparkproject

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{
  BucketedRandomProjectionLSH,
  BucketedRandomProjectionLSHModel,
  StandardScaler,
  VectorAssembler
}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._

/**
 * Approximate-nearest-neighbour recommender on a standardized 11-d vector.
 *
 * Features in the vector (in order):
 *   energy, valence, danceability, acousticness, instrumentalness,
 *   speechiness, loudness, tempo, liveness, mode, key_folded
 *
 * Key folding: 11→0, 10→1, 9→2, 8→3, 7→4, 6→5; keys 0..5 unchanged.
 * Equivalent to key_folded = min(key, 11 - key).
 *
 * Pipeline:
 *   VectorAssembler  → packs the 11 columns into one Vector
 *   StandardScaler   → centers + unit-variances each dimension (so binary
 *                      `mode` and continuous `tempo` contribute comparably)
 *   BucketedRandomProjectionLSH → Euclidean-distance LSH index
 *
 * Build is offline: fit the pipeline once and cache the transformed catalog.
 * Recommendation is online: `approxNearestNeighbors` against the cached
 * catalog, with overfetch to absorb same-artist exclusions.
 */
object RecommenderANN {

  // 10 numeric features + key handled separately.
  private val NUMERIC_FEATURES: Seq[String] = Seq(
    "energy", "valence", "danceability", "acousticness", "instrumentalness",
    "speechiness", "loudness", "tempo", "liveness", "mode"
  )

  private val ALL_FEATURES: Seq[String] = NUMERIC_FEATURES :+ "key_folded"

  // LSH tuning. bucketLength ≈ √(numFeatures) per Spark guidance, rounded
  // down. More hash tables → better recall, more memory.
  private val LSH_BUCKET_LENGTH   = 2.0
  private val LSH_NUM_HASH_TABLES = 5

  // Columns shown for seed and recs. Excludes album_name, duration_ms,
  // total_artist_followers, avg_artist_popularity, artist_ids.
  private val DISPLAY_COLS: Seq[String] = Seq(
    "id", "name", "artists", "genre", "year", "popularity", "niche_genres",
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "lyrics"
  )

  /**
   * Offline tree/index construction. Returns the fitted pipeline (which
   * carries the LSH model) and the transformed catalog (with `rawFeatures`,
   * `features`, and `hashes` columns added). The caller is expected to
   * persist the catalog.
   */
  def build(songs: DataFrame): (PipelineModel, DataFrame) = {
    val cleaned = songs.na.drop("any", "key" +: NUMERIC_FEATURES)

    val withKey = cleaned.withColumn(
      "key_folded",
      least(col("key"), lit(11) - col("key")).cast("double")
    )

    // Parse the `artists` column into a clean array of individual artist
    // names. Robust to comma/semicolon-separated values and Python-list-style
    // bracketed/quoted formats like ['Artist A', 'Artist B'].
    val withArtists = withKey.withColumn(
      "artist_list",
      filter(
        transform(split(col("artists"), "[\\[\\]'\",;]"), s => trim(s)),
        s => length(s) > 0
      )
    )

    val assembler = new VectorAssembler()
      .setInputCols(ALL_FEATURES.toArray)
      .setOutputCol("rawFeatures")

    val scaler = new StandardScaler()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setWithMean(true)
      .setWithStd(true)

    val lsh = new BucketedRandomProjectionLSH()
      .setInputCol("features")
      .setOutputCol("hashes")
      .setBucketLength(LSH_BUCKET_LENGTH)
      .setNumHashTables(LSH_NUM_HASH_TABLES)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, lsh))
    val model = pipeline.fit(withArtists)
    val transformed = model.transform(withArtists)
    (model, transformed)
  }

  /**
   * Online retrieval. Overfetches `topK * 5 + 50` neighbours so the
   * same-artist filter and the seed self-match leave enough survivors.
   */
  def recommend(model: PipelineModel,
                catalog: DataFrame,
                seedId: String,
                topK: Int = 10): DataFrame = {

    val seedDisplay = catalog.filter(col("id") === lit(seedId))
      .select(DISPLAY_COLS.head, DISPLAY_COLS.tail: _*)

    println("Seed song:")
    seedDisplay.show(false)

    val seedRow = catalog.filter(col("id") === lit(seedId))
      .select("features", "artist_list")
      .head()

    val seedVec = seedRow.getAs[Vector]("features")
    val seedArtists: Seq[String] = {
      val raw = seedRow.getAs[Seq[String]]("artist_list")
      if (raw == null) Seq.empty else raw
    }

    // A candidate passes if none of its parsed artists appears in the seed's
    // artist set. Implemented as `intersection is empty`.
    val artistFilter: Column =
      if (seedArtists.isEmpty) lit(true)
      else {
        val seedArr = array(seedArtists.map(lit): _*)
        col("artist_list").isNull ||
          size(array_intersect(col("artist_list"), seedArr)) === lit(0)
      }

    val lshModel = model.stages.last.asInstanceOf[BucketedRandomProjectionLSHModel]

    val overfetch = topK * 5 + 50

    val displayCols: Seq[Column] = DISPLAY_COLS.map(col) :+ col("distCol")

    lshModel
      .approxNearestNeighbors(catalog, seedVec, overfetch)
      .filter(col("id") =!= lit(seedId))
      .filter(artistFilter)
      .orderBy("distCol")
      .limit(topK)
      .select(displayCols: _*)
  }
}