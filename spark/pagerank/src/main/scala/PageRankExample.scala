/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
//package org.apache.spark.examples.graphx

// $example on$
import org.apache.spark.graphx.GraphLoader
// $example off$
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

/**
 * A PageRank example on social network dataset
 * Run with
 * {{{
 * bin/run-example graphx.PageRankExample
 * }}}
 */
object pagerank {
  def main(args: Array[String]): Unit = {
    // Creates a SparkSession.
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()
    val sc = spark.sparkContext

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val home_dir = System.getProperty("user.home")
    val spark_home =  home_dir + "/spark-2.4.0-bin-hadoop2.7/"
    val data_file = spark_home + "data/sosp/web-BerkStan.txt"
    // $example on$
    // Load the edges as a graph
    //val graph = GraphLoader.edgeListFile(sc, "data/berkeley_stanford/web-BerkStan.txt")
    val graph = GraphLoader.edgeListFile(sc, data_file)
    // Run PageRank
    val ranks = graph.pageRank(0.0001).vertices
    // Join the ranks with the usernames
    //val users = sc.textFile("data/graphx/users.txt").map { line =>
/*    val users = sc.textFile("users.txt").map { line =>
      val fields = line.split(",")
      (fields(0).toLong, fields(1))
    }
    val ranksByUsername = users.join(ranks).map {
      case (id, (username, rank)) => (username, rank)
    }
    // Print the result
    println(ranksByUsername.collect().mkString("\n"))
*/
    // $example off$
    spark.stop()
  }
}
// scalastyle:on println
