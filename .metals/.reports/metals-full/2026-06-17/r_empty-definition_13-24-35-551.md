error id: file:///C:/Users/yurek/OneDrive/Documents/GitHub/How-Love-Flies/How-Love-Flies/src/main/scala/com/yurekce/sparkproject/config/SparkConfig.scala:getOrCreate.
file:///C:/Users/yurek/OneDrive/Documents/GitHub/How-Love-Flies/How-Love-Flies/src/main/scala/com/yurekce/sparkproject/config/SparkConfig.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -scala/Predef.
	 -scala/Predef#
	 -scala/Predef().
offset: 747
uri: file:///C:/Users/yurek/OneDrive/Documents/GitHub/How-Love-Flies/How-Love-Flies/src/main/scala/com/yurekce/sparkproject/config/SparkConfig.scala
text:
```scala
package com.yurekce.sparkproject.config
import org.apache.spark.sql.SparkSession

object SparkConfig {
  def createSession(): SparkSession = {
    SparkSession.builder()
      .appName("HowLoveFlies")
      .master("local[7]") // monster has 8
      .config("spark.driver.memory", "12g")
      .config("spark.driver.maxResultSize", "10g")
      .config("spark.sql.shuffle.partitions", "14")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "2000M")
      .config("spark.memory.fraction", "0.7")
      .config("spark.memory.storageFraction", "0.5")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:6.3.3")
      .getOrCreat@@e()
  }
}
```


#### Short summary: 

empty definition using pc, found symbol in pc: 