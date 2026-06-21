ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
      name := "How-Love-Flies",
      resolvers += "Maven Central" at "https://repo1.maven.org/maven2/",
      libraryDependencies ++= Seq(
          "org.apache.spark" %% "spark-core" % "3.5.8",
          "org.apache.spark" %% "spark-sql"  % "3.5.8",
          "org.apache.spark" %% "spark-mllib" % "3.5.8",
          "com.johnsnowlabs.nlp" %% "spark-nlp-gpu" % "6.3.3"
      ),
      dependencyOverrides += "org.scala-lang.modules" %% "scala-parser-combinators" % "2.3.0",
      // Fork into a fresh JVM so we control its heap. With master("local[*]"),
      // Spark runs INSIDE this forked JVM, so its driver heap == this -Xmx.
      // 391MB BERT model + Kryo broadcast needs more than sbt's default ~512MB.
      Compile / run / fork := true,
      Compile / run / javaOptions ++= Seq("-Xmx12g", "-Xss4m")
  )