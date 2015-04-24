/* SparkGP.scala */
import java.nio.file.Paths
import breeze.linalg.{DenseMatrix => DM}
import breeze.linalg.cholesky

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.clustering.VectorWithNorm
import org.apache.spark.mllib.random.RandomRDDs._

import math._
import util.Random.nextGaussian

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
  def distance(xs: Array[Double], ys: Array[Double]) = {
    sqrt((xs zip ys).map { case (x,y) => pow(y - x, 2) }.sum)
  }
  def covariance(i: IndexedRow, j: IndexedRow): MatrixEntry = {
    val d = distance(i.vector.toArray, j.vector.toArray)
    MatrixEntry(i.index, j.index, exp(-d))
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
    val r = 100

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

    val N = rows.count.toInt

    // Initialize IndexedRowMatrix's
    val y = new IndexedRowMatrix(rows.map(_._1))
    val X = new IndexedRowMatrix(rows.map(_._2))

    // Build covariance matrix K
    val K = new CoordinateMatrix(
      rows.cartesian(rows)
        .map(pair => Functions.covariance(pair._1._2, pair._2._2))
    ).toBlockMatrix

    // # Step 1: Form the matrix product C \Omega
    // O = np.random.normal(0, 1, size=(N, r)) / np.sqrt(r)
    // KO = K.dot(O)
//    val O = Matrices.dense(
//      N, r,
//      Array.range(0, N * r).map(_ => nextGaussian / r)
//    )

    val O = new IndexedRowMatrix(
      normalVectorRDD(sc, N.toLong, r, 20, 101L)
        .zipWithIndex
        .map(row => IndexedRow(row._2, row._1))
    ).toBlockMatrix
    val KO = K.multiply(O)

    // # Step 2: Compute \Phi^T, the left factor of the rank-m spectral projection of the small matrix C \Omega.
    // U, _, _ = np.linalg.svd(KO, full_matrices=False)
    // phi = U.T
    val svd = KO.toIndexedRowMatrix.computeSVD(r, computeU = true)
    val U = svd.U
    val phi = U.toBlockMatrix.transpose

    // # Step 3: C_1 = \Phi C \Phi^T
    // C_1 = phi.dot(K).dot(phi.T)
    val C_1 = phi.multiply(K).multiply(phi.transpose)

    // # Step 4:  Perform a Cholesky factorization of C1 = B B^T
    // B = np.linalg.cholesky(C_1)
    val B = cholesky(breeze.linalg.DenseMatrix(C_1.toLocalMatrix.toArray))

    println("%s".format(t))

  }
}