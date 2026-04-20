package com.yurekce.sparkproject

import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.Column

object LyricsCleaner {

  val splitByFourLinesUdf = udf((text: String) => {
    if (text == null || text.trim.isEmpty) {
      Seq.empty[String]
    } else {
      text
        .replace("\\n", "\n")              // fix escaped newlines
        .split("\n")                      // split into lines
        .map(_.trim)
        .filter(_.nonEmpty)               // remove empty lines
        .grouped(8)                       // group every 4 lines
        .map(_.mkString(" "))             // merge into single string
        .toSeq
    }
  })

  def splitByFourLines(col: Column): Column = splitByFourLinesUdf(col)
}