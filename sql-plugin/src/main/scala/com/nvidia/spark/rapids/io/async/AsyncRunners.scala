/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.io.async

import java.util.concurrent.Callable
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}
import java.util.concurrent.locks.ReentrantLock
import java.util.function.LongUnaryOperator

import scala.collection.mutable

import ai.rapids.cudf.{HostMemoryAllocator, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.HostAlloc
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.jni.TaskPriority

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.execution.TrampolineUtil.{bytesToString => bToStr}

/**
 * Marker trait for resources required by AsyncRunners.
 */
sealed trait AsyncRunResource

/**
 * HostResource represents host memory resource requirement for CPU-bound tasks.
 */
case class HostResource(sizeInBytes: Long) extends AsyncRunResource {
  override def toString: String = s"HostResource(${bToStr(sizeInBytes)})"
}

/**
 * DeviceResource is a marker object for GPU resources, no additional fields needed.
 */
object DeviceResource extends AsyncRunResource

/**
 * Result wrapper for AsyncRunner execution that carries the output data and optional resource
 * release callback for deferred resource management.
 */
trait AsyncResult[T] extends AutoCloseable {
  val data: T

  /**
   * Metrics associated with the task execution, such as scheduling and execution time.
   */
  val metrics: AsyncMetrics

  /**
   * Indicates whether the result holds a post-run close hook to extend resource lifecycle.
   */
  val closeHook: Option[() => Unit] = None

  /**
   * Trigger release hook if it exists. Do not try to close the data itself, because
   * the lifecycle of the data is managed by the caller.
   */
  override def close(): Unit = {
    closeHook.foreach(callback => callback())
  }
}

case class AsyncMetrics(scheduleTimeMs: Long, executionTimeMs: Long)

class AsyncMetricsBuilder {
  private var scheduleTimeMs: Long = 0L
  private var executionTimeMs: Long = 0L

  def setScheduleTimeMs(time: Long): AsyncMetricsBuilder = {
    this.scheduleTimeMs = time
    this
  }

  def setExecutionTimeMs(time: Long): AsyncMetricsBuilder = {
    this.executionTimeMs = time
    this
  }

  def build(): AsyncMetrics = {
    AsyncMetrics(scheduleTimeMs, executionTimeMs)
  }
}

/**
 * Basic implementation of AsyncResult for runners that release resources immediately upon
 * completion. This result type is suitable for short-lived operations that don't need to retain
 * resources after execution. FastReleaseResult has no release hook and resources are freed
 * automatically as soon as the runner finishes.
 */
class FastReleaseResult[T](override val data: T,
    override val metrics: AsyncMetrics) extends AsyncResult[T]

/**
 * Basic implementation of AsyncResult that allows resources to be held after runner completion
 * and released later via an explicit callback. This result type is suitable for operations that
 * need to retain resources after execution until the consumer explicitly releases them.
 * DecayReleaseResult provides a release hook that must be triggered by calling close() when
 * resources are no longer needed.
 */
class DecayReleaseResult[T](override val data: T,
    override val metrics: AsyncMetrics,
    releaseCallback: () => Unit) extends AsyncResult[T] {
  override val closeHook: Option[() => Unit] = Some(releaseCallback)
}

/**
 * States of AsyncRunner during its lifecycle
 *
 * - Init: The initial state when the runner is polled from the worker queue. The firstTime flag
 * indicates if it's the first time the runner is being scheduled.
 * - Pending: The runner is waiting for the required resource.
 * - ScheduleFailed: The runner failed to be scheduled due to resource acquisition failure.
 * - Running: The runner is currently executing.
 * - Completed: The runner has completed execution successfully, but still holds the resource.
 * - ExecFailed: The runner execution failed with an exception, but still holds the resource.
 * - Closed: The runner has been closed and released the resource. If an exception is provided,
 * it indicates a failed workload.
 * - Cancelled: The runner has been cancelled from outside (usually due to Spark Task failure,
 * interruption)
 */
sealed trait AsyncRunnerState

case class Init(firstTime: Boolean) extends AsyncRunnerState

case object Pending extends AsyncRunnerState

case class ScheduleFailed(exception: Throwable) extends AsyncRunnerState

case object Running extends AsyncRunnerState

case object Completed extends AsyncRunnerState

case class ExecFailed(exception: Throwable) extends AsyncRunnerState

case class Closed(exception: Option[Throwable]) extends AsyncRunnerState

case object Cancelled extends AsyncRunnerState

/**
 * The AsyncRunner interface represents a resource-aware runner that can be scheduled by
 * ResourceBoundedThreadExecutor. Runners define their resource requirements, execution priority,
 * and can optionally hold resources after completion for deferred release.
 *
 * AsyncRunner provides hooks for resource lifecycle management and supports both immediate
 * and deferred resource release patterns based on the corresponding AsyncResult implementation:
 * - FastReleaseResult: resources are released immediately after the runner completion.
 * - DecayReleaseResult: resources are held after runner completion and released when the caller
 *   explicitly invokes the release callback. This is useful for runners that need to retain
 *   resources for a certain period after execution.
 */
trait AsyncRunner[T] extends Callable[AsyncResult[T]] with AutoCloseable {
  /**
   * Resource required by the runner, such as host memory or GPU semaphore.
   */
  def resource: AsyncRunResource

  /**
   * Priority of the async runner, higher value means higher priority.
   */
  def priority: Long

  /**
   * Optional method which attempts to free up resources that have finished being used.
   * This method is for recycling resources piece by piece, as soon as the piece ends its usage.
   * In addition, it is a [mutable] operation, the returned resource should be deducted from the
   * current resource requirement of the runner. If no resources can be freed at the moment,
   * return None.
   */
  protected[async] def tryFree(byForce: Boolean): Option[AsyncRunResource] = None

  /**
   * The abstract method defines the actual execution logic, which should be implemented by the
   * subclass.
   */
  protected def callImpl(): T

  /**
   * Builds the result of the runner execution, which includes the data and a callback to release
   * resources if needed. This method can be overridden by subclasses to customize the result
   * format to carry additional metadata or state.
   */
  protected def buildResult(resultData: T, metrics: AsyncMetrics): AsyncResult[T]

  // Close method of AsyncRunner should be idempotent
  override def close(): Unit = {
    // Ensure we do NOT block multiple closings for the same runner
    if (!closeStarted.compareAndSet(false, true)) {
      return
    }
    withStateLock(releaseAnyway = true) { rr =>
      rr.getState match {
        case Cancelled | Closed(_) | Completed | ExecFailed(_) =>
        // terminated states, safe to close
        case state =>
          throw new IllegalStateException(s"Unexpected runner state: $state of $this")
      }
      Option(poolPtr).foreach(_.finishUpRunner(this))
    }
  }

  def call(): AsyncResult[T] = {
    require(result == null, s"AsyncRunner.call() should only be called once: $this")

    val startTime = System.nanoTime()
    val resultData = try {
      beforeExecuteHooks.foreach { hook => hook() }
      callImpl()
    } finally {
      afterExecuteHooks.foreach { hook => hook() }
    }
    metricsBuilder.setExecutionTimeMs(System.nanoTime() - startTime)

    result = buildResult(resultData, metricsBuilder.build())
    result
  }

  private val beforeExecuteHooks = mutable.ArrayBuffer.empty[() => Unit]
  private val afterExecuteHooks = mutable.ArrayBuffer.empty[() => Unit]

  // Add hook to be executed right before the task execution.
  def addPreHook(hook: () => Unit): Unit = beforeExecuteHooks += hook

  // Add hook to be executed right after the task execution.
  def addPostHook(hook: () => Unit): Unit = afterExecuteHooks += hook

  /**
   * This method is called when the required resource has been just acquired from pool.
   * It can be overridden by subclasses to perform actions right after the acquisition.
   */
  protected[async] def onStart(pool: ResourcePool): Unit = {
    require(Option(poolPtr).isEmpty, "The resource has already been acquired")
    poolPtr = pool
  }

  /**
   * This method is called when the runner is being closed and its acquired resource is about to be
   * released back into the pool. It can be overridden by subclasses to perform cleanup actions.
   */
  protected[async] def onClose(): Unit = {
    poolPtr = null
  }

  // Cache the result for 2 purposes:
  // 1. Ensure call() is only called once
  // 2. Facilitate accessing the status of AsyncRunner inside ThreadExecutor after execution
  @volatile private[async] var result: AsyncResult[T] = _

  private[async] lazy val metricsBuilder = new AsyncMetricsBuilder

  /**
   * AsyncRunner is not necessarily tied to a Spark task, despite it is usually the case.
   * sparkTaskId is None if the runner is not associated with any Spark task. Otherwise, it
   * carries the corresponding Spark task ID.
   */
  def sparkTaskContext: Option[TaskContext] = None

  // Pointer to the resource pool from which the resource is acquired.
  @volatile protected var poolPtr: ResourcePool = _

  // Atomic flag to ensure close() is only executed once.
  private[async] val closeStarted: AtomicBoolean = new AtomicBoolean(false)

  /**
   * Get the current state of the AsyncRunner, nonblocking method.
   */
  private[async] def getState: AsyncRunnerState = state

  /**
   * Set the state of the AsyncRunner, blocking method.
   * The state transition is only recommended to be done inside ResourcePool, in order to
   * maintain the consistency between the state and resource allocation.
   */
  private[async] def setState(newState: AsyncRunnerState): Unit = {
    require(isHoldingStateLock, s"The caller must hold the state lock: $this")
    if (newState != state) {
      state = newState
    }
  }

  @volatile private var state: AsyncRunnerState = Init(firstTime = true)

  def isHoldingStateLock: Boolean = stateLock.isHeldByCurrentThread

  /**
   * Helper method to execute a function while holding the state lock, which means that
   * no other thread can modify the state of the AsyncRunner during the execution.
   */
  def withStateLock[R](holdAnyway: Boolean = false,
      releaseAnyway: Boolean = false)(fn: AsyncRunner[T] => R): R = {
    if (stateLock.isHeldByCurrentThread) {
      try {
        fn(this)
      } finally {
        if (releaseAnyway) {
          stateLock.unlock()
        }
      }
    } else {
      stateLock.lock()
      var holdStateLock = holdAnyway
      try {
        fn(this)
      } catch {
        case t: Throwable =>
          holdStateLock = false
          throw t
      } finally {
        if (!holdStateLock) {
          stateLock.unlock()
        }
      }
    }
  }

  private val stateLock = new ReentrantLock()

  // Unique ID for the runner, mainly for logging and tracking purpose.
  private[async] val runnerId: Long = AsyncRunner.nextRunnerId()

  override def toString: String = {
    val tid = sparkTaskContext.map(_.taskAttemptId()).getOrElse(-1L)
    s"AsyncRunner(state=$state, runnerId=$runnerId, " +
        s"taskId=$tid, resource=$resource, priority=$priority)"
  }

  private[async] def setTag(key: String, value: String): Unit = {
    require(isHoldingStateLock, s"The caller must hold the state lock: $this")
    TAGS.put(key, value)
  }

  protected val TAGS = mutable.HashMap[String, String]()
}

/**
 * An AsyncRunner that requires no resource limits and has the highest scheduling priority.
 * Useful for lightweight jobs that should execute immediately without resource constraints.
 */
abstract class UnboundedAsyncRunner[T] extends AsyncRunner[T] {
  // Unbounded runners do not have a resource limit.
  override val resource: AsyncRunResource = HostResource(0L)

  // Unbounded runners have the highest priority.
  override val priority: Long = Long.MaxValue

  // Unbounded runners use FastReleaseResult as the placeholder.
  override protected def buildResult(resultData: T, metrics: AsyncMetrics): AsyncResult[T] = {
    new FastReleaseResult(resultData, metrics)
  }
}

/**
 * The base runner class for memory-bound runners that require a specific amount of host memory.
 * This type of runner is supposed to be scheduled by the ResourceBoundedThreadPoolExecutor
 * with HostMemoryPool.
 *
 * The runner priority is derived from the TaskPriority if it is associated with a Spark task,
 * which makes runners from the same Spark task to be scheduled one after another. It is important
 * to avoid starvation of runners of the same Spark task which might depend on each other.
 */
abstract class MemoryBoundedAsyncRunner[T] extends AsyncRunner[T]
    with HostMemoryAllocator with Logging {

  // The memory requirement in bytes for the memory-bound runner.
  val requiredMemoryBytes: Long

  // The base memory allocator from which the runner actually allocates memory.
  val baseMemoryAllocator: HostMemoryAllocator

  final override def resource: AsyncRunResource = {
    if (localPool < 0) {
      require(requiredMemoryBytes > 0, s"requiredMemoryBytes($requiredMemoryBytes) should > 0")
      localPool = requiredMemoryBytes
    }
    HostResource(localPool)
  }

  override protected[async] def tryFree(byForce: Boolean): Option[AsyncRunResource] = {
    require(isHoldingStateLock, s"The caller must hold the state lock: $this")

    if (getState == Running) {
      None
    } else if (byForce) {
      val ret = Some(HostResource(localPool))
      localPool = 0L
      ret
    } else {
      usedMem.get() match {
        case mem if mem == localPool =>
          None
        case mem =>
          val ret = Some(HostResource(localPool - mem))
          localPool = mem
          ret
      }
    }
  }

  override def priority: Long = {
    sparkTaskContext match {
      case Some(ctx) => TaskPriority.getTaskPriority(ctx.taskAttemptId())
      case None => 0L
    }
  }

  override protected[async] def onStart(pool: ResourcePool): Unit = {
    super.onStart(pool)
    // Initialize the local memory pool when the resource is acquired
    require(localPool == requiredMemoryBytes,
      s"localPool(${bToStr(localPool)}) != initial value(${bToStr(requiredMemoryBytes)}")
    usedMem.set(0L)
    peakUsedMem = 0L
  }

  // AsyncRunner.close() operates on the virtual memory, while MemoryBoundedAsyncRunner is backed
  // by actual host memory allocations, so the close operation cannot be achieved until all host
  // buffers are actually closed.
  // Also, we cannot close host buffers explicitly here as this would cause double-free errors,
  // so we have no choice but waiting for all allocated buffers being closed.
  override protected[async] def onClose(): Unit = {
    // wait until all allocated host memory buffers are actually closed
    bufCloseLock.lockInterruptibly()
    try {
      while (usedMem.get() > 0L) {
        bufCloseCond.await()
      }
    } finally {
      bufCloseLock.unlock()
    }
    super.onClose()
    logInfo(s"[OnClose] MemoryBoundedRunner: ID($runnerId), TaskID(" +
        s"${TAGS.getOrElse("tid", "None")}) initial budget(${bToStr(requiredMemoryBytes)})," +
        s" peak used memory(${bToStr(peakUsedMem)})")
  }

  override def allocate(size: Long, preferPinned: Boolean): HostMemoryBuffer = {
    require(getState == Running, s"Memory allocation is only allowed in Running state: $this")
    require(isHoldingStateLock, s"The caller must hold the state lock: $this")

    withRetryNoSplit[HostMemoryBuffer] {
      // Check and update the used memory atomically
      var newUsed = usedMem.addAndGet(size)
      // If the local pool is insufficient, try to borrow from the global pool
      if (newUsed > localPool) {
        val delta: Long = newUsed - localPool
        usedMem.addAndGet(-delta)
        logWarning(s"[$this] Local memory pool ${bToStr(localPool)} " +
            s"is insufficient for the upcoming allocation ${bToStr(size)}, " +
            s"trying to borrowing addition ${bToStr(delta)} from the global pool")
        // Blocking call to borrow memory from the global pool, allocating from pool
        // regardlessly if current runner is under bypassed mode in case of deadlock.
        val isBypassed = TAGS.get("isBypassed").contains("TRUE")
        poolPtr.asInstanceOf[HostMemoryPool].borrowMemory(delta, byForce = isBypassed)
        localPool += delta
        newUsed = usedMem.addAndGet(delta)
      }
      if (newUsed > peakUsedMem) {
        peakUsedMem = newUsed
      }
      // Call the base allocator to allocate the actual buffer
      val buf = baseMemoryAllocator.allocate(size, preferPinned)
      // Register a close handler to return the memory back either to the local or global pool
      HostAlloc.addEventHandler(buf, new OnCloseHandler(size))
      buf
    }
  }

  override def allocate(size: Long): HostMemoryBuffer = {
    allocate(size, preferPinned = true)
  }

  private class OnCloseHandler(bufferSize: Long) extends MemoryBuffer.EventHandler {

    def onClosed(refCount: Int): Unit = if (refCount == 0) {
      // Return the memory back to the local pool if the runner is still running,
      // otherwise return it back to the ResourcePool.
      usedMem.addAndGet(-bufferSize)
      withStateLock[Unit]() { rr =>
        rr.getState match {
          case Running =>
            bufCloseLock.lockInterruptibly()
            try {
              bufCloseCond.signal()
            } finally {
              bufCloseLock.unlock()
            }
          case _ =>
            poolPtr.release(rr, forcefully = false)
        }
      }
    }
  }

  private var localPool: Long = -1 // could be reduced/released by `tryFree` gradually
  private val usedMem = new AtomicLong(0L) // memory currently allocated from the runner
  private var peakUsedMem: Long = 0L // peak memory allocated from the runner

  private val bufCloseLock = new ReentrantLock()
  private val bufCloseCond = bufCloseLock.newCondition()
}

/**
 * A simple AsyncRunner implementation that wraps a function to be executed asynchronously.
 * NOTE: This class is only used in tests currently, removing the forTest suffix if it is used
 * in production code in the future.
 */
class AsyncFunctorForTest[T](
    override val resource: AsyncRunResource,
    override val priority: Long,
    functor: () => T) extends AsyncRunner[T] {

  override protected def buildResult(resultData: T, metrics: AsyncMetrics): AsyncResult[T] = {
    new FastReleaseResult(resultData, metrics)
  }

  override def callImpl(): T = functor()
}

object AsyncRunner {

  // Create a CPU-bound AsyncRunner with specified memory requirement and priority.
  // NOTE: The API is only used in tests currently
  def newCpuTask[T](fn: () => T,
      memoryBytes: Long,
      priority: Long = 0L): AsyncRunner[T] = {
    require(priority >= 0, s"Priority must be non-negative, got: $priority")
    val p = hostMemoryPenalty(memoryBytes, priority)
    new AsyncFunctorForTest[T](HostResource(memoryBytes), p, fn)
  }

  // Create a light-weight unbounded AsyncRunner with the highest priority.
  // NOTE: The API is only used in tests currently
  def newUnboundedTask[T](fn: () => T): AsyncRunner[T] = {
    new AsyncFunctorForTest[T](HostResource(0L), Long.MaxValue, fn)
  }

  // Atomically increase the global runner ID with overflow protection
  private def nextRunnerId(): Long = globalRunnerId.getAndUpdate(idUpdateFn)

  private lazy val idUpdateFn = new LongUnaryOperator {
    override def applyAsLong(operand: Long): Long = {
      if (operand == Long.MaxValue) 0L else operand + 1L
    }
  }

  private lazy val globalRunnerId = new AtomicLong(0)

  // Adjust the priority based on the memory overhead to minimal the potential clogging:
  // lightweight tasks should have higher priority.
  private def hostMemoryPenalty(memoryBytes: Long, priority: Long): Long = {
    require(memoryBytes >= 0, s"Memory bytes must be non-negative, got: $memoryBytes")
    priority - (memoryBytes >> 10)
  }
}
