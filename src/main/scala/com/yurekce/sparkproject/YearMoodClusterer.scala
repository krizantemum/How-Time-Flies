package com.yurekce.sparkproject

import java.awt.{BasicStroke, Color, Font, RenderingHints}
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, count}

/**
 * K-Means clustering of years on the 2-D (valence, energy) plane.
 *
 * Each clustering observation is a calendar year (1990–2022) represented by
 * the mean valence and mean energy of every song released that year. K is
 * chosen automatically by the max-distance-from-chord ("Kneedle-style")
 * elbow heuristic over WSSSE for K = 2..10.
 *
 * Outputs (all under csvFiles/clustering/year_mood/):
 *   elbow/          — k, wssse  (one row per K in the loop)
 *   best_k/         — best_k, best_wssse  (the auto-picked K)
 *   year_clusters/  — year, avg_valence, avg_energy, num_songs, cluster
 */
object YearMoodClusterer {

  private val K_RANGE: Range.Inclusive = 2 to 10
  private val SEED: Long = 42L
  private val OUT_DIR: String = "csvFiles/clustering/year_mood"

  def runAll(songs: DataFrame): Unit = {
    val spark = songs.sparkSession
    import spark.implicits._

    // Step 1 — build the 33-point year dataset.
    val perYear = songs
      .groupBy("year")
      .agg(
        avg("valence").as("avg_valence"),
        avg("energy").as("avg_energy"),
        count("valence").as("num_songs")
      )
      .orderBy("year")
      .cache()

    // Step 2 — vectorize + standardize.
    val assembler = new VectorAssembler()
      .setInputCols(Array("avg_valence", "avg_energy"))
      .setOutputCol("rawFeatures")

    val scaler = new StandardScaler()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setWithMean(true)
      .setWithStd(true)

    val assembled = assembler.transform(perYear)
    val scaled = scaler.fit(assembled).transform(assembled).cache()

    // Step 3 — elbow loop.
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

    // Step 4 — auto-pick the elbow K.
    val bestK = pickElbowK(elbowRows)
    val bestWssse = elbowRows.find(_._1 == bestK).map(_._2).getOrElse(Double.NaN)
    println(s"YearMoodClusterer: auto-picked bestK = $bestK (wssse = $bestWssse)")

    Seq((bestK, bestWssse)).toDF("best_k", "best_wssse")
      .coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save(s"$OUT_DIR/best_k")

    // Step 5 — final clustering at the auto-picked K.
    val finalModel = new KMeans()
      .setK(bestK)
      .setSeed(SEED)
      .setFeaturesCol("features")
      .fit(scaled)

    val clustered = finalModel
      .transform(scaled)
      .select("year", "avg_valence", "avg_energy", "num_songs", "prediction")
      .withColumnRenamed("prediction", "cluster")
      .orderBy("year")

    // Step 6 — write the consumable scatter-plot input.
    clustered.coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("sep", "\t")
      .mode("overwrite")
      .save(s"$OUT_DIR/year_clusters")

    clustered.show(1000, truncate = false)

    // Step 7 — render PNG scatter (one labeled dot per year, colored by cluster).
    val plotRows: Seq[(Int, Double, Double, Int)] = clustered
      .select("year", "avg_valence", "avg_energy", "cluster")
      .collect()
      .map(r => (r.getInt(0), r.getDouble(1), r.getDouble(2), r.getInt(3)))
      .toSeq
    renderScatterPng(plotRows, s"$OUT_DIR/year_mood_scatter.png")

    perYear.unpersist()
    scaled.unpersist()
  }

  /** Render a labeled 2-D scatter of (avg_valence, avg_energy) with each
    * year drawn as a colored dot and its 4-digit year label beneath. */
  private def renderScatterPng(rows: Seq[(Int, Double, Double, Int)],
                               outPath: String): Unit = {
    // BufferedImage doesn't strictly require headless mode, but set it to be
    // safe on JVMs without a display.
    System.setProperty("java.awt.headless", "true")

    val width  = 1400
    val height = 1000
    val marginL = 90
    val marginR = 160  // room for legend
    val marginT = 70
    val marginB = 90
    val plotW = width  - marginL - marginR
    val plotH = height - marginT - marginB

    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g   = img.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)

    // Background.
    g.setColor(Color.WHITE)
    g.fillRect(0, 0, width, height)

    // Data extents with 5% padding so labels don't get cut off.
    val xs = rows.map(_._2)
    val ys = rows.map(_._3)
    val (xMin, xMax) = (xs.min, xs.max)
    val (yMin, yMax) = (ys.min, ys.max)
    val xPad = (xMax - xMin) * 0.05 max 0.01
    val yPad = (yMax - yMin) * 0.05 max 0.01
    val xLo = xMin - xPad
    val xHi = xMax + xPad
    val yLo = yMin - yPad
    val yHi = yMax + yPad

    def px(x: Double): Int = marginL + ((x - xLo) / (xHi - xLo) * plotW).toInt
    def py(y: Double): Int = height - marginB - ((y - yLo) / (yHi - yLo) * plotH).toInt

    // Plot frame.
    g.setColor(Color.BLACK)
    g.setStroke(new BasicStroke(1.5f))
    g.drawLine(marginL, height - marginB, width - marginR, height - marginB)
    g.drawLine(marginL, marginT, marginL, height - marginB)

    // Axis ticks (6 ticks per axis).
    val tickFont = new Font("SansSerif", Font.PLAIN, 12)
    g.setFont(tickFont)
    for (i <- 0 to 5) {
      val frac = i / 5.0
      val xVal = xLo + frac * (xHi - xLo)
      val xPix = marginL + (frac * plotW).toInt
      g.drawLine(xPix, height - marginB, xPix, height - marginB + 6)
      val xLab = f"$xVal%.2f"
      g.drawString(xLab, xPix - g.getFontMetrics.stringWidth(xLab) / 2, height - marginB + 22)

      val yVal = yLo + frac * (yHi - yLo)
      val yPix = height - marginB - (frac * plotH).toInt
      g.drawLine(marginL - 6, yPix, marginL, yPix)
      val yLab = f"$yVal%.2f"
      g.drawString(yLab, marginL - 10 - g.getFontMetrics.stringWidth(yLab), yPix + 4)
    }

    // Axis titles.
    g.setFont(new Font("SansSerif", Font.BOLD, 14))
    val xTitle = "avg_valence"
    g.drawString(xTitle,
      marginL + (plotW - g.getFontMetrics.stringWidth(xTitle)) / 2,
      height - 30)
    val yTitle = "avg_energy"
    val savedTransform = g.getTransform
    g.rotate(-math.Pi / 2, 25, marginT + plotH / 2)
    g.drawString(yTitle, 25 - g.getFontMetrics.stringWidth(yTitle) / 2, marginT + plotH / 2)
    g.setTransform(savedTransform)

    // Title.
    g.setFont(new Font("SansSerif", Font.BOLD, 18))
    val title = "Year mood clusters  (valence × energy)"
    g.drawString(title, (width - g.getFontMetrics.stringWidth(title)) / 2, 36)

    // Cluster palette (color-blind friendly-ish).
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

    // Points + year labels.
    val pointR = 8
    val labelFont = new Font("SansSerif", Font.PLAIN, 11)
    g.setFont(labelFont)
    val lfm = g.getFontMetrics
    rows.foreach { case (year, valence, energy, cluster) =>
      val x = px(valence)
      val y = py(energy)
      val c = palette(cluster % palette.length)

      g.setColor(c)
      g.fillOval(x - pointR, y - pointR, pointR * 2, pointR * 2)
      g.setColor(Color.BLACK)
      g.setStroke(new BasicStroke(1f))
      g.drawOval(x - pointR, y - pointR, pointR * 2, pointR * 2)

      val lab = year.toString
      g.drawString(lab, x - lfm.stringWidth(lab) / 2, y + pointR + 14)
    }

    // Legend.
    g.setFont(new Font("SansSerif", Font.BOLD, 13))
    val legX = width - marginR + 20
    var legY = marginT + 20
    g.setColor(Color.BLACK)
    g.drawString("Cluster", legX, legY)
    legY += 20
    g.setFont(new Font("SansSerif", Font.PLAIN, 12))
    rows.map(_._4).distinct.sorted.foreach { c =>
      g.setColor(palette(c % palette.length))
      g.fillOval(legX, legY - 9, 12, 12)
      g.setColor(Color.BLACK)
      g.drawOval(legX, legY - 9, 12, 12)
      g.drawString(s"$c", legX + 20, legY)
      legY += 18
    }

    g.dispose()

    val file = new File(outPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    ImageIO.write(img, "png", file)
    println(s"YearMoodClusterer: wrote $outPath")
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