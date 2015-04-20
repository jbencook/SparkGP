# SparkGP
## Using a Gaussian Process over large data to predict weather data at county centroids.

To build:

    sbt package

To run:

    $SPARK_HOME/bin/spark-submit --class SparkGP --master local[*] target/scala-2.10/sparkgp_2.10-1.0.jar </path/to/ushcn/directory> <data element> <yyyy-mm-dd>