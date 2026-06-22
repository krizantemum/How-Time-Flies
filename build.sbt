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
      // Spark 3.5 reaches into JDK internals (e.g. sun.nio.ch.DirectBuffer in
      // StorageUtils). On Java 17+ those packages are no longer exported, so the
      // job dies at SparkContext startup with IllegalAccessError unless we
      // re-open them. This is Spark's own recommended --add-opens set.
      Compile / run / javaOptions ++= Seq("-Xmx12g", "-Xss4m") ++ Seq(
          "--add-opens=java.base/java.lang=ALL-UNNAMED",
          "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
          "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
          "--add-opens=java.base/java.io=ALL-UNNAMED",
          "--add-opens=java.base/java.net=ALL-UNNAMED",
          "--add-opens=java.base/java.nio=ALL-UNNAMED",
          "--add-opens=java.base/java.util=ALL-UNNAMED",
          "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
          "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
          "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
          "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
          "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
          "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
          "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
      )
  )