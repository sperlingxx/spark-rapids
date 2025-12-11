/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids

import scala.collection.mutable

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, Predicate}
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.SQLMetrics

case class DebugFilterExec(condition: Expression, child: SparkPlan)
    extends UnaryExecNode with Logging {

  final override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"))

  override def output: Seq[Attribute] = child.output

  override protected def withNewChildInternal(newChild: SparkPlan): DebugFilterExec =
    copy(child = newChild)

  protected override def doExecute(): RDD[InternalRow] = {
    val numOutputRows = longMetric("numOutputRows")
    child.execute().mapPartitionsWithIndexInternal { (batchIdx, iter) =>
      val predicate = Predicate.create(condition, child.output)
      predicate.initialize(0)
      val errRows = mutable.ArrayBuffer.empty[String]

      val implIter = iter.filter { row =>
        val r = predicate.eval(row)
        if (r) {
          numOutputRows += 1
        } else {
          val sb = mutable.StringBuilder.newBuilder
          (0 until row.numFields).foreach { i =>
            if (i > 0) sb.append(", ")
            sb.append(
              s"${output(i).name}=${row.get(i, output(i).dataType)}(NULL=${row.isNullAt(i)})")
          }
          errRows += sb.toString()
        }
        r
      }

      new Iterator[InternalRow] {
        override def hasNext: Boolean = {
          val r = implIter.hasNext
          if (!r && errRows.nonEmpty) {
            val errMsg = errRows.mkString(
              s"DebugFilterExec found ${errRows.size} rows that did not pass the filter " +
                  s"condition '$condition' in batch $batchIdx:\n", "\n", "\n")
            logError(errMsg)
            throw new RuntimeException(errMsg)
          }
          r
        }

        override def next(): InternalRow = implIter.next()
      }
    }
  }
}
