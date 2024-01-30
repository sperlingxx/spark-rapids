/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

package org.apache.spark.sql.execution.datasources.parquet.rapids

import java.util.TimeZone

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, Table}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.schema.MessageType

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.memory.MemoryMode
import org.apache.spark.sql.execution.vectorized.rapids.HostWritableColumnVector
import org.apache.spark.sql.types.StructType

class VectorizedParquetGpuProducer(
    conf: Configuration,
    rowBatchSize: Int,
    fileBuffer: HostMemoryBuffer,
    offset: Long,
    len: Long,
    metrics: Map[String, GpuMetric],
    dateRebaseMode: DateTimeRebaseMode,
    timestampRebaseMode: DateTimeRebaseMode,
    hasInt96Timestamps: Boolean,
    clippedSchema: MessageType,
    readDataSchema: StructType) extends GpuDataProducer[Table] with Logging {

  logDebug(s"ColumnDescriptors ${clippedSchema.getColumns.asScala.mkString(" | ")}")
  logDebug(s"ColumnFieldTypes ${clippedSchema.asGroupType().getFields.asScala.mkString(" | ")}")
  logDebug(s"ReadDataSchema ${readDataSchema.sql}")

  private var curBatchSize: Int = _

  private val pageReader: ParquetFileReader = {
    val options = HadoopReadOptions.builder(conf)
        .withRange(offset, offset + len)
        .build()
    val bufferFile = new HMBInputFile(fileBuffer, length = Some(offset + len))
    val reader = new ParquetFileReader(bufferFile, options)
    // The fileSchema here has already been clipped
    reader.setRequestedSchema(clippedSchema)
    reader
  }

  private val parquetColumn: ParquetColumn = {
    val converter = new ParquetToSparkSchemaConverter()
    converter.convertParquetColumn(clippedSchema, None)
  }

  private val writerVersion: ParsedVersion = try {
    VersionParser.parse(pageReader.getFileMetaData.getCreatedBy)
  } catch {
    case _: Exception =>
      // If any problems occur trying to parse the writer version, fallback to sequential reads
      // if the column is a delta byte array encoding (due to PARQUET-246).
      null
  }

  private lazy val hostColumnBuilders: Array[HostWritableColumnVector] = {
    parquetColumn.sparkType.asInstanceOf[StructType].fields.map { f =>
      new HostWritableColumnVector(curBatchSize, f.dataType)
    }
  }

  private lazy val columnVectors: Array[ParquetColumnVector] = {
    hostColumnBuilders.indices.toArray.map { i =>
      new ParquetColumnVector(parquetColumn.children(i),
        hostColumnBuilders(i), curBatchSize, MemoryMode.ON_HEAP,
        Set.empty[ParquetColumn].asJava, true, null);
    }
  }

  // Performed all the host-side reading work before transferring to device
  private lazy val hostBatches: mutable.Queue[Array[HostColumnVector]] = {

    val buffer = mutable.Queue.empty[Array[HostColumnVector]]
    var pages = pageReader.readNextFilteredRowGroup()
    curBatchSize = (pages.getRowCount min rowBatchSize).toInt

    while (pages != null) {

      val stack = mutable.Stack[ParquetColumnVector](columnVectors: _*)
      while (stack.nonEmpty) {
        stack.pop() match {
          case cv if cv.getColumn.isPrimitive =>
            cv.setColumnReader(
              new VectorizedColumnReader(
                cv.getColumn.descriptor.get,
                cv.getColumn.required,
                pages,
                null,
                dateRebaseMode.value,
                TimeZone.getDefault.getID,
                timestampRebaseMode.value,
                TimeZone.getDefault.getID,
                writerVersion))
          case cv =>
            cv.getChildren.asScala.foreach(stack.push)
        }
      }

      (0L until pages.getRowCount by rowBatchSize).foreach { from =>

        curBatchSize = ((pages.getRowCount - from) min rowBatchSize).toInt

        if (buffer.nonEmpty) {
          hostColumnBuilders.foreach(_.reAllocate(curBatchSize))
          columnVectors.foreach(_.reset())
        }
        columnVectors.foreach { cv =>
          cv.getLeaves.asScala.foreach { leafCv =>
            val reader = leafCv.getColumnReader
            if (reader != null) {
              reader.readBatch(curBatchSize, leafCv.getValueVector,
                leafCv.getRepetitionLevelVector, leafCv.getDefinitionLevelVector)
            }
          }
          cv.assemble()
        }

        buffer.enqueue(hostColumnBuilders.map(_.build()))
      }

      // logInfo(s"PageReadStore contains ${pages.getRowCount} rows")

      pages = pageReader.readNextFilteredRowGroup()
    }

    buffer
  }

  private var firstBatch = true

  override def hasNext: Boolean = {
    hostBatches.nonEmpty
  }

  override def next: Table = {
    withResource(hostBatches.dequeue()) { hostCVs =>
      if (firstBatch) {
        // About to start using the GPU
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        firstBatch = false
      }
      logInfo(s"VectorizedParquetGpuProducer batches ${hostCVs.head.getRowCount} rows")

      withResource(hostCVs.indices.map(i => hostCVs(i).copyToDevice())) { dCVs =>
        new Table(dCVs: _*)
      }
    }
  }

  override def close(): Unit = {
    if (!firstBatch) {
      hostBatches.foreach { hcvArray => hcvArray.foreach(_.close()) }
    }
    pageReader.close()
    fileBuffer.close()
  }
}
