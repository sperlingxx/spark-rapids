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
import java.util.concurrent.Future
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}

import scala.collection.JavaConverters._
import scala.collection.mutable

import ai.rapids.cudf.{HostColumnVector, HostMemoryBuffer, NvtxColor, Table}
import com.nvidia.spark.rapids.{DateTimeRebaseMode, GpuDataProducer, GpuMetric, GpuSemaphore, HMBInputFile, NvtxWithMetrics}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import org.apache.hadoop.conf.Configuration
import org.apache.parquet.{HadoopReadOptions, VersionParser}
import org.apache.parquet.VersionParser.ParsedVersion
import org.apache.parquet.column.page.PageReadStore
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.schema.MessageType

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.datasources.parquet.rapids.HybridTableProducer.preloadMemPool
import org.apache.spark.sql.execution.vectorized.rapids.HostWritableColumnVector
import org.apache.spark.sql.rapids.ComputeThreadPool
import org.apache.spark.sql.types.{ArrayType, BinaryType, DataType, DecimalType, MapType, StructType}


case class AsyncBatchResult(data: Array[HostColumnVector], sizeInByte: Long)

class AsyncParquetReader(
    conf: Configuration,
    tgtBatchSize: Int,
    fileBuffer: HostMemoryBuffer,
    offset: Long,
    len: Long,
    metrics: Map[String, GpuMetric],
    dateRebaseMode: DateTimeRebaseMode,
    timestampRebaseMode: DateTimeRebaseMode,
    clippedSchema: MessageType)
  extends Iterator[AsyncBatchResult] with AutoCloseable with Logging {

  private var readerClosed = false

  private val pageReader: ParquetFileReader = {
    val options = HadoopReadOptions.builder(conf)
      .withRange(offset, offset + len)
      .withCodecFactory(new ParquetCodecFactory(conf, 0))
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

  private lazy val rowGroupQueue: mutable.Queue[PageReadStore] = {
    val rowGroups = mutable.Queue.empty[PageReadStore]
    var curRowGroup: PageReadStore = pageReader.readNextFilteredRowGroup()
    while (curRowGroup != null) {
      rowGroups += curRowGroup;
      curRowGroup = pageReader.readNextFilteredRowGroup()
    }
    rowGroups
  }

  private lazy val totalRowCnt = rowGroupQueue.foldLeft(0)((s, x) => s + x.getRowCount.toInt)
  private lazy val rowBatchSize = {
    val value = (tgtBatchSize.toDouble / len * totalRowCnt).toInt max 1
    logInfo(s"total row count: $totalRowCnt ; batch size in row: $value")
    remainTotalRows = totalRowCnt
    remainBatchRows = value min totalRowCnt
    remainBatchRows
  }
  private var remainTotalRows: Int = _
  private var remainBatchRows: Int = _
  private var remainPageRows: Int = 0

  private lazy val hostColumnBuilders: Array[HostWritableColumnVector] = {
    parquetColumn.sparkType.asInstanceOf[StructType].fields.map { f =>
      new HostWritableColumnVector(rowBatchSize min totalRowCnt, f.dataType)
    }
  }
  private lazy val columnVectors: Array[ParquetColumnVector] =  {
    hostColumnBuilders.indices.toArray.map { i =>
      new ParquetColumnVector(parquetColumn.children(i),
        hostColumnBuilders(i),
        rowBatchSize min totalRowCnt,
        Set.empty[ParquetColumn].asJava, true, -1, null);
    }
  }

  private def releaseParquetCV(parquetCVs: Array[ParquetColumnVector]): Unit = {
    parquetCVs.foreach { pcv =>
      pcv.getValueVector.close()
      if (pcv.getColumn.isPrimitive) {
        pcv.setColumnReader(null)
        if (pcv.getDefinitionLevelVector != null) {
          pcv.getDefinitionLevelVector.close()
        }
        if (pcv.getRepetitionLevelVector != null) {
          pcv.getRepetitionLevelVector.close()
        }
      }
      if (pcv.getChildren.size() > 0) {
        releaseParquetCV(pcv.getChildren.asScala.toArray)
      }
    }
  }

  private def readImpl(): AsyncBatchResult = {
    var currentGroup = rowGroupQueue.head

    while (remainBatchRows > 0) {
      if (remainPageRows == 0) {
        rowGroupQueue.dequeue()
        currentGroup = rowGroupQueue.head

        // update column readers to read the new page
        metrics("cpuDecodeDictTime").ns {
          val stack = mutable.Stack[ParquetColumnVector](columnVectors: _*)
          while (stack.nonEmpty) {
            stack.pop() match {
              case cv if cv.getColumn.isPrimitive =>
                cv.setColumnReader(
                  new VectorizedColumnReader(
                    cv.getColumn.descriptor.get,
                    cv.getColumn.required,
                    cv.maxRepetitiveDefLevel,
                    currentGroup,
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
        }

        remainPageRows = currentGroup.getRowCount.toInt
      }

      metrics("cpuDecodeDataTime").ns {
        val readSize = remainBatchRows min remainPageRows
        remainPageRows -= readSize
        remainBatchRows -= readSize
        remainTotalRows -= readSize

        columnVectors.foreach { cv =>
          cv.getLeaves.asScala.foreach {
            case leaf if leaf.getColumnReader != null =>
              leaf.getColumnReader.readBatch(readSize, leaf.getValueVector,
                leaf.getRepetitionLevelVector, leaf.getDefinitionLevelVector)
            case _ =>
          }
          cv.assemble()
          // Reset all value vectors(HostWritableColumnVector) along with def/repVectors.
          // The reset is essential because we are going to either finalize current batch
          // or read another RowGroup, or even both.
          // As of value vectors, reset means update the offsets of target buffers.
          // As of def/repVectors vectors, reset simply means re-initialize.
          cv.reset()
        }
      }
    }

    // Finalize current batch and reset all buffers for the next batch
    metrics("hostVecBuildTime").ns {
      // materialize current batch in the memory layout of cuDF column vector
      val sizeCounter: Array[Long] = Array.fill(1)(0L)
      val result = hostColumnBuilders.map(_.build(sizeCounter))
      // update batch size and remaining
      remainBatchRows = rowBatchSize min remainTotalRows
      // Reset all the HostColumnBuffers for the upcoming batch
      hostColumnBuilders.foreach(_.reallocate(remainBatchRows))

      AsyncBatchResult(result, sizeCounter.head)
    }
  }

  private def launchTask(): Unit = {
    val task = new ComputeThreadPool.TaskWithPriority(() => readImpl(), 1)
    future = task.future()
    ComputeThreadPool.submitTask(task)
  }

  override def hasNext: Boolean = {
    if (future != null || remainTotalRows > 0) {
      if (future == null) launchTask()
      true
    } else {
      false
    }
  }

  override def next(): AsyncBatchResult = {
    val ret = future.get()
    future = null
    if (remainTotalRows > 0) launchTask()
    ret
  }

  override def close(): Unit = {
    if(!readerClosed) {
      if (future != null) {
        future.cancel(true)
      }
      // release all work buffers since all work are done
      releaseParquetCV(columnVectors)
      // close ParquetFileReader to release all decompressors
      pageReader.close()
      // close fileBuffer additionally to reduce refCount to 0
      fileBuffer.close()
      readerClosed = true
    }
  }

  private var future: Future[AsyncBatchResult] = _

}

object AsyncParquetReader {
  def apply(conf: Configuration,
            tgtBatchSize: Int,
            fileBuffer: HostMemoryBuffer,
            offset: Long,
            len: Long,
            metrics: Map[String, GpuMetric],
            dateRebaseMode: DateTimeRebaseMode,
            timestampRebaseMode: DateTimeRebaseMode,
            hasInt96Timestamps: Boolean,
            clippedSchema: MessageType,
            readDataSchema: StructType): AsyncParquetReader = {
    new AsyncParquetReader(conf,
      tgtBatchSize, fileBuffer, offset, len,
      metrics,
      dateRebaseMode, timestampRebaseMode,
      clippedSchema)
  }
}

class HybridTableProducer(
    asyncHostReader: AsyncParquetReader,
    hybridOpts: HybridParquetOpts,
    metrics: Map[String, GpuMetric]) extends GpuDataProducer[Table] with Logging {

  private val enablePreload = hybridOpts.maxDevicePreloadBytes > 0
  private var preloadedMemSize: Long = 0
  private var holdGPUSemaphore = false

  override def hasNext: Boolean = {
    asyncHostReader.hasNext
  }

  override def next: Table = {
    val asyncRet = withResource(new NvtxWithMetrics("waitAsyncCpuDecode", NvtxColor.YELLOW,
      metrics("waitAsyncDecode"))) { _ =>
      asyncHostReader.next()
    }

    if (!holdGPUSemaphore) {
      if (enablePreload && tryAcquirePreloadH2DSlots(asyncRet.sizeInByte)) {
        preloadedMemSize += asyncRet.sizeInByte
      } else {
        withResource(new NvtxWithMetrics(
          "Wait GPU for HostToDevice", NvtxColor.WHITE, metrics("h2dWaitGPU"))) { _ =>
          takeSemaphore()
        }
      }
    }

    withResource(new NvtxWithMetrics("Transfer HostVectors to Device", NvtxColor.CYAN,
      metrics("hostVecToDeviceTime"))) { _ =>
      closeOnExcept(asyncRet.data) { hostCVs =>

        val batchRows = hostCVs.head.getRowCount
        logInfo(s"VectorizedParquetGpuProducer batches $batchRows rows; ")
        metrics.get("cpuDecodeRows").foreach(_.+=(batchRows))
        metrics.get("cpuDecodeBatches").foreach(_.+=(1))
        metrics.get("numOutputBatches").foreach(_.+=(1))

        val deviceCVs = hostCVs.safeMap { hcv =>
          val dcv = hcv.copyToDevice()
          hcv.close()
          dcv
        }
        withResource(deviceCVs) { dCVs => new Table(dCVs: _*) }
      }
    }
  }

  private def tryAcquirePreloadH2DSlots(batchMemSize: Long): Boolean = {
    if (preloadMemPool.getAndAdd(-batchMemSize) < 0) {
      preloadMemPool.getAndAdd(batchMemSize)
      false
    } else {
      true
    }
  }

  private def takeSemaphore(): Unit = {
    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    holdGPUSemaphore = true
    if (enablePreload) {
      preloadMemPool.getAndAdd(-preloadedMemSize)
      preloadedMemSize = 0
    }
  }

  override def close(): Unit = {
    asyncHostReader.close()
    if (!holdGPUSemaphore) takeSemaphore()
  }

}

object HybridTableProducer {
  def parseHybridParquetOpts(str: String): HybridParquetOpts = {
    str match {
      case "" =>
        HybridParquetOpts(DeviceOnly, 0, 0, 0L)
      case "GPU_ONLY" =>
        HybridParquetOpts(DeviceOnly, 0, 0, 0L)
      case "CPU_ONLY" =>
        HybridParquetOpts(HostOnly, 0, 0, 0L)
      case s if s.startsWith("DeviceFirst(") && s.endsWith(")") =>
        val args = s.slice(12, str.length -1).split(",")
        val maxHostThreads = args(0).toInt
        val pollInterval = args(1).toInt
        val maxDevicePreloadBytes = if (args.length > 1) args(2).toLong else 0L
        HybridParquetOpts(DeviceFirst, maxHostThreads, pollInterval, maxDevicePreloadBytes)
      case _ =>
        throw new IllegalArgumentException(s"illegal HybridParquetOpts $str")
    }
  }

  def schemaSupportCheck(types: Array[DataType]): Boolean = {
    types.collectFirst {
      case _: BinaryType =>
        false
      case _: DecimalType =>
        // if DecimalType.isByteArrayDecimalType(dt) =>
        false
      case st: StructType =>
        schemaSupportCheck(st.fields.map(_.dataType))
      case ArrayType(et, _) =>
        schemaSupportCheck(Array(et))
      case MapType(kt, vt, _) =>
        schemaSupportCheck(Array(kt, vt))
    }.getOrElse(true)
  }

  def initialize(opts: HybridParquetOpts): Unit = {
    if (!initialized.get()) synchronized {
      if (!initialized.get()) {
        ComputeThreadPool.launch(opts.maxHostThreads, 1024)
        preloadMemPool = new AtomicLong(opts.maxDevicePreloadBytes)
        initialized.set(true)
      }
    }
  }

  private var preloadMemPool: AtomicLong = _
  private val initialized: AtomicBoolean = new AtomicBoolean(false)

}

sealed trait ReadMode
object HostOnly extends ReadMode
object DeviceOnly extends ReadMode
object DeviceFirst extends ReadMode

case class HybridParquetOpts(mode: ReadMode,
                             maxHostThreads: Int,
                             pollInterval: Int,
                             maxDevicePreloadBytes: Long)
