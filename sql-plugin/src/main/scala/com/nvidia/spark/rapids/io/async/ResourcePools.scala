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

import java.lang.{Long => JLong}
import java.util.concurrent.{ConcurrentHashMap, TimeUnit}
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.execution.TrampolineUtil.bytesToString

// Being thrown when a task requests resources that are not valid or exceed the limits
class InvalidResourceRequest(msg: String) extends RuntimeException(
  s"Invalid resource request: $msg")

// Represents the status of acquiring resources for a task
sealed trait AcquireStatus

case class AcquireSuccessful(elapsedTime: Long) extends AcquireStatus

// AcquireFailed indicates that the task could not be scheduled due to resource constraints
case object AcquireFailed extends AcquireStatus

// AcquireExcepted indicates that an exception occurred while trying to acquire resources
case class AcquireExcepted(exception: Throwable) extends AcquireStatus

/**
 * ResourceManager interface to be implemented for AsyncRunners requiring different kinds of
 * resources.
 *
 * Currently, only HostMemoryManager is implemented, which limits the maximum in-flight host
 * memory bytes. In the future, we can add more.
 */
trait ResourcePool {
  /**
   * Returns true if the task can be accepted, false otherwise.
   * TrafficController will block the task from being scheduled until this method returns true.
   */
  def acquire[T](runner: AsyncRunner[T], timeout: Long): AcquireStatus

  /**
   * Closed a completed AsyncRunner (either successfully or failed) and cleanup its resources.
   *
   * This method is typically called as a callback hooked to various completion events to handle
   * scenarios such as cancellation or end of lifecycle.
   */
  def finishUpRunner[T](runner: AsyncRunner[T]): Unit


  def release[T](runner: AsyncRunner[T], forcefully: Boolean): Unit
}

/**
 * HostMemoryPool enforces a maximum limit on total host memory bytes that can be held
 * by in-flight runner simultaneously. It provides blocking resource acquisition with
 * configurable timeout.
 *
 * The implementation uses condition variables to efficiently block and wake up waiting
 * tasks when resources become available through task completion and resource release.
 */
class HostMemoryPool(val maxHostMemoryBytes: Long) extends ResourcePool with Logging {

  private val lock = new ReentrantLock()

  private val condition = lock.newCondition()

  // Tracking running AsyncRunners which actually acquires host memory, which is mainly for deadlock
  // prevention for now.
  private val numRunnerInFlight: AtomicLong = new AtomicLong(0L)

  private var remaining: Long = maxHostMemoryBytes

  // Only counts for the AsyncRunners which actually acquired host memory.
  private val numRunnerInPool: AtomicLong = new AtomicLong(0L)

  // Map of TaskAttemptId -> number of active runners, use Java Long for nullability
  private val tasksInPool = new ConcurrentHashMap[Long, JLong](64)

  override def acquire[T](runner: AsyncRunner[T], timeoutMs: Long): AcquireStatus = {
    // step 1: extract the resource requirements and runner info
    val memoryRequire: Long = extractResource(runner).sizeInBytes

    // step 2: try to acquire the resource with blocking and timeout
    // 2.1 If no resource needed, acquire immediately
    if (memoryRequire == 0L) {
      AcquireSuccessful(elapsedTime = 0L)
    }
    // 2.2 The main path for acquiring resource with blocking and timeout
    else {
      var isDone = false
      var isTimeout = false
      val timeoutNs = TimeUnit.MILLISECONDS.toNanos(timeoutMs)
      var waitTimeNs = timeoutNs
      // Enter into the critical section which is guarded by the lock from concurrent access
      lock.lockInterruptibly()
      try {
        // The main loop to try acquiring the resource with blocking and timeout
        do {
          if (remaining >= memoryRequire || {
            // [Deadlock Prevention]
            // Due to the decay release, the virtual memory limit may interact with other dependency
            // mechanisms, such as in a local join. In this scenario, both sides of the join may
            // perform multithreaded scans limited by the HostMemoryPool. The join operator requires
            // outputs from both sides, but one side may occupy all the memory budget, leaving the
            // other side blocked and waiting for memory to be released.
            //
            // [Solution]
            // If there is no runner in flight, run the request runner immediately regardless of
            // the current available resource.
            numRunnerInFlight.compareAndSet(0L, 1L)
          }) {
            remaining -= memoryRequire
            isDone = true
            // When remaining < 0, the increment was done by the CAS above
            if (remaining >= 0L) {
              numRunnerInFlight.incrementAndGet()
            }
            // register a post-hook to decrement it as soon as the runner is done
            runner.addPostHook(() => numRunnerInFlight.decrementAndGet())
          } else if (waitTimeNs > 0L) {
            waitTimeNs = condition.awaitNanos(waitTimeNs)
          } else {
            isTimeout = true
            logWarning(s"Failed to acquire ${bytesToString(memoryRequire)}, remaining=" +
                s"${bytesToString(remaining)}, AsyncRunners=$numRunnerInPool, " +
                s"SparkTasks=${tasksInPool.size}")
          }
        }
        while (!isDone && !isTimeout)

        if (!isDone) {
          AcquireFailed
        } else {
          // Update nonAtomic states if the resource is acquired successfully
          runner.sparkTaskContext.foreach { ctx =>
            registerRunner(ctx)
          }
          // Log a warning when the resource is over-committed
          if (remaining < 0) {
            logWarning(
              s"Over-committed HostMemoryPool: exceeded_amount=${bytesToString(-remaining)}, " +
                  s"AsyncRunners=$numRunnerInPool, SparkTasks=${tasksInPool.size}")
          }
          AcquireSuccessful(elapsedTime = timeoutNs - waitTimeNs)
        }
      } catch {
        case ex: Throwable => AcquireExcepted(ex)
      } finally {
        lock.unlock()
      }
    }
  }

  override def release[T](rr: AsyncRunner[T], forcefully: Boolean): Unit = {
    val freeAmount: Long = if (forcefully) {
      extractResource(rr).sizeInBytes
    } else {
      rr.tryFree.map(_.asInstanceOf[HostResource].sizeInBytes).getOrElse(0L)
    }

    if (freeAmount > 0L) {
      lock.lockInterruptibly()
      val canBeClosed: Boolean = try {
        // Return the budget and wake up waiters
        remaining += freeAmount
        condition.signalAll()
        // Check if current runner can be closed
        rr.getState != Running && extractResource(rr).sizeInBytes == 0L
      } finally {
        lock.unlock()
      }
      if (canBeClosed) { // Close runner does NOT require the pool lock
        closeRunner(rr)
      }
    }
  }

  override def finishUpRunner[T](runner: AsyncRunner[T]): Unit = {
    if (extractResource(runner).sizeInBytes > 0) {
      release(runner, forcefully = true)
    } else {
      closeRunner(runner)
    }
  }

  // Close the runner, requires the state lock of the runner
  private def closeRunner[T](runner: AsyncRunner[T]): Unit = {
    require(runner.isHoldingStateLock, s"The caller must hold the state lock: $this")

    // Callback for onClose actions
    runner.onClose()
    // Finalize the runner state
    runner.getState match {
      case Completed => // Completed -> Closed
        runner.setState(Closed(None))
      case ExecFailed(ex) => // ExecFailed -> Closed
        runner.setState(Closed(Some(ex)))
      case Cancelled => // Cancelled -> Closed
        runner.setState(Closed(Some(new IllegalStateException("cancelled"))))
      case _ =>
        throw new IllegalStateException(s"Should NOT reach here: $this")
    }
    // Unregister the runner from the Pool, within the lock
    runner.sparkTaskContext.foreach { ctx =>
      unregisterRunner(ctx)
    }
  }

  private def registerRunner(ctx: TaskContext): Unit = {
    numRunnerInPool.incrementAndGet()
    tasksInPool.compute(ctx.taskAttemptId(), (_, v: JLong) => {
      if (v == null) new JLong(1) else v + 1L
    })
  }

  private def unregisterRunner(ctx: TaskContext): Unit = {
    numRunnerInPool.decrementAndGet()
    val tid = ctx.taskAttemptId()
    tasksInPool.computeIfPresent(tid, (_, v: JLong) => {
      // It is possible runnersForTask == 0, if some runners were cancelled from caller side
      if (v <= 1L) {
        logDebug(s"[LOG POINT] remaining=${bytesToString(remaining)}, " +
            s"AsyncRunners=$numRunnerInPool, SparkTasks=${tasksInPool.size})")
        null
      } else {
        v - 1L
      }
    })
  }

  override def toString: String = {
    s"HostMemoryPool(maxHostMemoryBytes=${bytesToString(maxHostMemoryBytes)})"
  }

  private def extractResource(task: AsyncRunner[_]): HostResource = {
    task.resource match {
      case r: HostResource => r
      case r => throw new InvalidResourceRequest(
        s"Task ${task.getClass.getName} does not require HostResource, but got $r")
    }
  }
}
