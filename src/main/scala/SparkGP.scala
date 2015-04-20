/* SimpleApp.scala */
import java.nio.file.Paths

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

object Functions {
  def getType(s: String): String = { s.substring(12, 16) }
  def getYear(s: String): String = { s.substring(6, 10) }
  def getMonth(s: String): String = { s.substring(10, 12) }
  def getDayValue(s: String, day: Int): Int = {
    val dayIndex = day - 1
    s.substring(16 + (dayIndex * 8), 21 + (dayIndex * 8))
      .trim
      .toInt
  }
  def getStation(s: String): String = { s.substring(0, 6) }
  def getStationLocation(s: String): Tuple2[String, Tuple2[Double, Double]] = {
    val elements = s.split("\\s+").slice(0, 4)
    (elements(0), (elements(1).toDouble, elements(2).toDouble))
  }
}

object SparkGP {
  def main(args: Array[String]) {
    // A quick check
    if (args.length < 3) {
      System.err.println("Usage: SparkGP </path/to/data> <type> <yyyy-mm-dd>")
      System.exit(1)
    }

    // Some setup
    val date = args(2).split("-")
    val year = date(0)
    val month = date(1)
    val day = date(2).toInt

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val ushcnDailyPath = Paths.get(args(0), "ushcn_daily.txt").toString
    val ushcnStationPath = Paths.get(args(0), "ushcn_stations.txt").toString

    // Import and munge USHCN daily weather data
    val data = sc.textFile(ushcnDailyPath)
      .filter(line => Functions.getType(line).equalsIgnoreCase(args(1)))
      .filter(line => Functions.getYear(line).equals(year))
      .filter(line => Functions.getMonth(line).equals(month))
      .map(line => (Functions.getStation(line), Functions.getDayValue(line, day)))

    // Import and munge USHCN weather stations data
    val stations = sc.textFile(ushcnStationPath)
      .map(Functions.getStationLocation)

    // Create IndexedRows for locations and weather values
    val rows = data.join(stations)
      .map(elem => (Vectors.dense(elem._2._1.toDouble), Vectors.dense(Array(elem._2._2._1, elem._2._2._2))))
      .zipWithIndex
      .map(row => (IndexedRow(row._2, row._1._1), IndexedRow(row._2, row._1._2)))
      .cache()

    // Initialize IndexedRowMatrix's
    val y = new IndexedRowMatrix(rows.map(_._1))
    val X = new IndexedRowMatrix(rows.map(_._2))

    // Build covariance matrix K
    val K = rows.cartesian(rows)

    println("%s".format(K.first()))

  }
}