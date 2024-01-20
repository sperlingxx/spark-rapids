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

package org.apache.spark.sql.execution.datasources.parquet

import java.lang.reflect.Method

import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, Table}
import com.nvidia.spark.rapids.{DateTimeRebaseMode, GpuDataProducer, GpuMetric, GpuSemaphore, HMBInputFile, RapidsWritableColumnVector}
import com.nvidia.spark.rapids.Arm.withResource
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.schema.MessageType

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.datasources.parquet.rapids.shims.ShimVectorizedColumnReader
import org.apache.spark.sql.execution.vectorized.WritableColumnVector
import org.apache.spark.sql.types.StructType


class VectorizedParquetGpuProducer(
    conf: Configuration,
    rowBatchSize: Long,
    fileBuffer: HostMemoryBuffer,
    offset: Long,
    len: Long,
    metrics: Map[String, GpuMetric],
    dateRebaseMode: DateTimeRebaseMode,
    timestampRebaseMode: DateTimeRebaseMode,
    hasInt96Timestamps: Boolean,
    clippedSchema: MessageType,
    readDataSchema: StructType) extends GpuDataProducer[Table] with Logging {

  private val pageReader: ParquetFileReader = {
    val readOpt = HadoopReadOptions.builder(conf).build()
    val bufferFile = new HMBInputFile(fileBuffer, Some(offset), Some(len))
    val reader = new ParquetFileReader(bufferFile, readOpt)
    // The fileSchema here has already been clipped
    reader.setRequestedSchema(clippedSchema)
    reader
  }

  val writerVersion: ParsedVersion = try {
    VersionParser.parse(pageReader.getFileMetaData.getCreatedBy)
  } catch {
    case _: Exception =>
      // If any problems occur trying to parse the writer version, fallback to sequential reads
      // if the column is a delta byte array encoding (due to PARQUET-246).
      null
  }

  // Follow Spark 3.2
  private val sparkSchema = new ParquetToSparkSchemaConverter(conf).convert(clippedSchema)
  private val colDesc = clippedSchema.getColumns
  private val colTypes = clippedSchema.asGroupType().getFields

  // Performed all the host-side reading work before transferring to device
  private lazy val hostBatches: mutable.Queue[Array[HostColumnVector]] = {

    val buffer = mutable.Queue.empty[Array[HostColumnVector]]
    var pages = pageReader.readNextFilteredRowGroup()

    while (pages != null) {
      val readers = (0 until colDesc.size()).map { i =>
        new ShimVectorizedColumnReader(
          i,
          colDesc,
          colTypes,
          pages,
          convertTz = null,
          dateRebaseMode.value,
          timestampRebaseMode.value,
          int96CDPHive3Compatibility = false,
          writerVersion)
      }

      (0L until pages.getRowCount by rowBatchSize).foreach { from =>
        val batchNum = ((pages.getRowCount - from) min rowBatchSize).toInt

        buffer.enqueue(
          sparkSchema.fields.zip(readers).map { case (f, reader) =>
            val rapidsVec = new RapidsWritableColumnVector(batchNum, f.dataType)
            VectorizedParquetGpuProducer.readBatchMethod
              .invoke(reader, batchNum.asInstanceOf[AnyRef], rapidsVec.asInstanceOf[AnyRef])
            rapidsVec.build()
          }
        )
      }
      logWarning(s"HostParquetReader read ${pages.getRowCount} rows")

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
      val dCVs = hostCVs.indices.map(i => hostCVs(i).copyToDevice())
      new Table(dCVs: _*)
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

object VectorizedParquetGpuProducer {
  lazy val readBatchMethod: Method = {
    val method = classOf[VectorizedColumnReader].getDeclaredMethod("readBatch", Integer.TYPE,
      classOf[WritableColumnVector])
    method.setAccessible(true)
    method
  }
}
