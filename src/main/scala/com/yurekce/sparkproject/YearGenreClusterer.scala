package com.yurekce.sparkproject

import java.awt.{BasicStroke, Color, Font, RenderingHints}
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, count, row_number}

/**
 * K-Means clustering of years on their genre mix.
 *
 * Each clustering observation is a calendar year (1990–2022) represented by a
 * vector of genre proportions — the share of that year's songs falling in each
 * genre. Years therefore cluster by overall genre composition ("eras"). The
 * single most-common genre of each year (the genre its songs has the most) is
 * carried along as a descriptive label. K is chosen automatically by the
 * max-distance-from-chord ("Kneedle-style") elbow heuristic over WSSSE for
 * K = 2..10.
 *
 * Outputs (all under csvFiles/clustering/year_genre/):
 *   elbow/          — k, wssse  (one row per K in the loop)
 *   best_k/         — best_k, best_wssse  (the auto-picked K)
 *   year_clusters/  — year, num_songs, dominant_genre, dominant_genre_count, cluster
 *   year_genre_timeline.png — a year-by-year timeline colored by cluster
 */
object YearGenreClusterer {

  private val K_RANGE: Range.Inclusive = 2 to 10
  private val SEED: Long = 42L
  private val OUT_DIR: String = "csvFiles/clustering/year_genre"

  def runAll(songs: DataFrame): Unit = {
    val spark = songs.sparkSession
    import spark.implicits._

    // Step 1 — the stable, sorted list of genres = the feature space.
    val genres: Array[String] = songs
      .select("genre")
      .distinct()
      .collect()
      .map(_.getString(0))
      .filter(_ != null)
      .sorted

    require(genres.length >= 2,
      s"Need at least 2 genres to cluster on; found ${genres.length}")

    // Step 2 — per-year genre counts via pivot, then proportions.
    // Pivot columns are renamed g0..gN so genre names containing dots or
    // spaces can't be misread as nested-field accessors downstream.
    val genreCountsRaw = songs
      .groupBy("year")
      .pivot("genre", genres.toSeq)
      .count()
      .na.fill(0)

    val genreCounts = genres.zipWithIndex.foldLeft(genreCountsRaw) {
      case (df, (g, i)) => df.withColumnRenamed(g, s"g$i")
    }

    val totals = songs.groupBy("year").agg(count("*").as("num_songs"))

    // Feature = genre share of the year: g_i / num_songs.
    val featureCols: Array[String] = genres.indices.map(i => s"prop_$i").toArray
    val propExprs = genres.indices.map(i =>
      (col(s"g$i") / col("num_songs")).as(s"prop_$i"))

    val perYear = genreCounts
      .join(totals, "year")
      .select((col("year") +: col("num_songs") +: propExprs): _*)
      .orderBy("year")
      .cache()

    // Step 3 — dominant genre per year (most songs; ties broken by name).
    val topGenre = Window.partitionBy("year")
      .orderBy(col("genre_count").desc, col("genre").asc)

    val dominant = songs
      .groupBy("year", "genre")
      .agg(count("*").as("genre_count"))
      .withColumn("rn", row_number().over(topGenre))
      .filter(col("rn") === 1)
      .select(
        col("year"),
        col("genre").as("dominant_genre"),
        col("genre_count").as("dominant_genre_count"))

    // Step 4 — vectorize + standardize (equal weight per genre, so a rare
    // genre's swing counts as much as the always-large mainstream one).
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("rawFeatures")

    val scaler = new StandardScaler()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setWithMean(true)
      .setWithStd(true)

    val assembled = assembler.transform(perYear)
    val scaled = scaler.fit(assembled).transform(assembled).cache()

    // Step 5 — elbow loop.
    val elbowRows: Seq[(Int, Double)] = K_RANGE.map { k =>
      val model = new KMeans()
        .setK(k)
        .setSeed(SEED)
        .setFeaturesCol("features")
        .fit(scaled)
      (k, model.summary.trainingCost)
    }.toList

    val elbowDf = elbowRows.toDF("k", "wssse").orderBy("k")
    elbowDf.coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save(s"$OUT_DIR/elbow")
    elbowDf.show(false)

    // Step 6 — auto-pick the elbow K.
    val bestK = pickElbowK(elbowRows)
    val bestWssse = elbowRows.find(_._1 == bestK).map(_._2).getOrElse(Double.NaN)
    println(s"YearGenreClusterer: auto-picked bestK = $bestK (wssse = $bestWssse)")

    Seq((bestK, bestWssse)).toDF("best_k", "best_wssse")
      .coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save(s"$OUT_DIR/best_k")

    // Step 7 — final clustering at the auto-picked K.
    val finalModel = new KMeans()
      .setK(bestK)
      .setSeed(SEED)
      .setFeaturesCol("features")
      .fit(scaled)

    val clustered = finalModel
      .transform(scaled)
      .select("year", "num_songs", "prediction")
      .withColumnRenamed("prediction", "cluster")
      .join(dominant, "year")
      .select("year", "num_songs", "dominant_genre", "dominant_genre_count", "cluster")
      .orderBy("year")

    clustered.coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save(s"$OUT_DIR/year_clusters")

    clustered.show(1000, truncate = false)

    // Step 8 — render the year-by-year cluster timeline.
    val plotRows: Seq[(Int, String, Int)] = clustered
      .select("year", "dominant_genre", "cluster")
      .collect()
      .map(r => (r.getInt(0), r.getString(1), r.getInt(2)))
      .toSeq
    renderTimelinePng(plotRows, s"$OUT_DIR/year_genre_timeline.png")

    perYear.unpersist()
    scaled.unpersist()
  }

  /** Render a left-to-right timeline: one colored cell per year (color = its
    * cluster), the dominant genre rotated above it and the year below. The
    * legend lists every dominant genre observed in each cluster. */
  private def renderTimelinePng(rows: Seq[(Int, String, Int)],
                                outPath: String): Unit = {
    // Safe on JVMs without a display.
    System.setProperty("java.awt.headless", "true")

    val sorted = rows.sortBy(_._1)
    val n = sorted.length

    val cellW   = 46
    val cellGap = 4
    val marginL = 70
    val marginT = 175
    val marginB = 70
    val bandH   = 110
    val legendW = 360

    // Dominant genres seen in each cluster — drives the legend and height.
    val byCluster: Seq[(Int, Seq[String])] = sorted
      .groupBy(_._3)
      .toSeq
      .sortBy(_._1)
      .map { case (c, rs) => (c, rs.map(_._2).distinct.sorted) }

    val width      = marginL + n * cellW + legendW
    val timelineH  = marginT + bandH + marginB
    // Generous per-cluster legend allowance so wrapped genre lists never clip.
    val legendH    = marginT + 30 + byCluster.size * 96 + marginB
    val height     = math.max(timelineH, legendH)

    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g   = img.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)

    // Background.
    g.setColor(Color.WHITE)
    g.fillRect(0, 0, width, height)

    // Cluster palette (color-blind friendly-ish), shared with YearMoodClusterer.
    val palette = Array(
      new Color(228,  26,  28),
      new Color( 55, 126, 184),
      new Color( 77, 175,  74),
      new Color(152,  78, 163),
      new Color(255, 127,   0),
      new Color(255, 215,   0),
      new Color(166,  86,  40),
      new Color(247, 129, 191),
      new Color( 64, 224, 208),
      new Color(102, 102, 102)
    )

    // Black or white text, whichever reads better on the given fill.
    def textOn(c: Color): Color = {
      val lum = 0.299 * c.getRed + 0.587 * c.getGreen + 0.114 * c.getBlue
      if (lum > 140) Color.BLACK else Color.WHITE
    }

    // Title.
    g.setColor(Color.BLACK)
    g.setFont(new Font("SansSerif", Font.BOLD, 18))
    val title = "Year genre clusters  (genre-mix similarity)"
    g.drawString(title, (width - g.getFontMetrics.stringWidth(title)) / 2, 34)

    val bandTop      = marginT
    val bandBottom   = marginT + bandH
    val genreAnchorY = marginT - 80  // rotated genre labels are centered here

    // One cell per year.
    sorted.zipWithIndex.foreach { case ((year, genre, cluster), i) =>
      val cx = marginL + i * cellW + cellW / 2
      val rx = marginL + i * cellW + cellGap / 2
      val rw = cellW - cellGap
      val fill = palette(cluster % palette.length)

      g.setColor(fill)
      g.fillRect(rx, bandTop, rw, bandH)
      g.setColor(Color.BLACK)
      g.setStroke(new BasicStroke(1f))
      g.drawRect(rx, bandTop, rw, bandH)

      // Cluster id inside the cell.
      g.setColor(textOn(fill))
      g.setFont(new Font("SansSerif", Font.BOLD, 13))
      val cLab = cluster.toString
      g.drawString(cLab,
        cx - g.getFontMetrics.stringWidth(cLab) / 2,
        bandTop + bandH / 2 + 5)

      // Year label below the cell.
      g.setColor(Color.BLACK)
      g.setFont(new Font("SansSerif", Font.PLAIN, 11))
      val yLab = year.toString
      g.drawString(yLab,
        cx - g.getFontMetrics.stringWidth(yLab) / 2,
        bandBottom + 20)

      // Dominant genre rotated above the cell.
      val saved = g.getTransform
      g.rotate(-math.Pi / 2, cx, genreAnchorY)
      g.drawString(genre,
        cx - g.getFontMetrics.stringWidth(genre) / 2,
        genreAnchorY + 4)
      g.setTransform(saved)
    }

    // Legend — each cluster and the dominant genres seen in it.
    val legendX = marginL + n * cellW + 22
    var legendY = marginT
    g.setColor(Color.BLACK)
    g.setFont(new Font("SansSerif", Font.BOLD, 14))
    g.drawString("Clusters", legendX, legendY)
    legendY += 26

    val plainFont = new Font("SansSerif", Font.PLAIN, 12)
    byCluster.foreach { case (c, gs) =>
      g.setColor(palette(c % palette.length))
      g.fillRect(legendX, legendY - 11, 14, 14)
      g.setColor(Color.BLACK)
      g.drawRect(legendX, legendY - 11, 14, 14)
      g.setFont(new Font("SansSerif", Font.BOLD, 12))
      g.drawString(s"Cluster $c", legendX + 22, legendY)
      legendY += 18

      g.setFont(plainFont)
      wrapText(gs.mkString(", "), g, legendW - 60).foreach { line =>
        g.drawString(line, legendX + 22, legendY)
        legendY += 16
      }
      legendY += 12
    }

    g.dispose()

    val file = new File(outPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    ImageIO.write(img, "png", file)
    println(s"YearGenreClusterer: wrote $outPath")
  }

  /** Greedily wrap a space-separated string into lines no wider than maxWidth
    * pixels under the Graphics2D's current font. */
  private def wrapText(text: String,
                       g: java.awt.Graphics2D,
                       maxWidth: Int): Seq[String] = {
    val fm = g.getFontMetrics
    val (lines, last) = text.split(" ").foldLeft((Vector.empty[String], "")) {
      case ((acc, cur), word) =>
        val candidate = if (cur.isEmpty) word else s"$cur $word"
        if (fm.stringWidth(candidate) <= maxWidth || cur.isEmpty) (acc, candidate)
        else (acc :+ cur, word)
    }
    (if (last.isEmpty) lines else lines :+ last).filter(_.nonEmpty)
  }

  /**
   * Pick the elbow K as the point with the maximum perpendicular distance
   * from the chord joining the first and last (k, wssse) points.
   *
   * For the line through (x1, y1) and (x2, y2), the distance from (k, w) is
   *   |(y2 - y1) k - (x2 - x1) w + (x2 y1 - x1 y2)| / sqrt((y2 - y1)^2 + (x2 - x1)^2)
   * The denominator is constant across points, so we maximize the numerator.
   */
  private def pickElbowK(points: Seq[(Int, Double)]): Int = {
    require(points.length >= 3,
      "Need at least 3 (k, wssse) points for elbow detection")

    val sorted = points.sortBy(_._1)
    val (x1, y1) = (sorted.head._1.toDouble, sorted.head._2)
    val (x2, y2) = (sorted.last._1.toDouble, sorted.last._2)

    sorted.maxBy { case (k, wssse) =>
      math.abs((y2 - y1) * k - (x2 - x1) * wssse + (x2 * y1 - x1 * y2))
    }._1
  }
}