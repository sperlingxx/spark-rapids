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

import java.util.concurrent.{Callable, TimeUnit}
import java.util.concurrent.locks.ReentrantLock
import javax.annotation.concurrent.GuardedBy

case class TaskResource(hostMemoryBytes: Long, requireGpuSemaphore: Boolean)

object TaskResource {
  def newCpuResource(hostMemoryBytes: Long): TaskResource = {
    new TaskResource(hostMemoryBytes, false)
  }

  def newGpuResource(): TaskResource = new TaskResource(0L, true)
}

/**
 * Simple wrapper around a [[Callable]] that also keeps track of the host memory bytes used by
 * the task.
 *
 * Note: we may want to add more metadata to the task in the future, such as the device memory,
 * as we implement more throttling strategies.
 */
class AsyncTask[T](val resource: TaskResource,
    val priority: Float,
    functor: () => T) extends Callable[T] {
  override def call(): T = functor()
}

object AsyncTask {
  // Adjust the priority based on the memory overhead to minimal the potential clogging:
  // lightweight tasks should have higher priority
  def hostMemoryPenalty(memoryBytes: Long, priority: Float = 0.0f): Float = {
    require(memoryBytes >= 0, s"Memory bytes must be non-negative, got: $memoryBytes")
    priority + (Long.MaxValue - memoryBytes).toFloat / Long.MaxValue
  }

  def newCpuTask[T](fn: () => T,
      memoryBytes: Long,
      priority: Float = 0.0f): AsyncTask[T] = {
    val adjustedPriority = hostMemoryPenalty(memoryBytes, priority)
    new AsyncTask[T](TaskResource.newCpuResource(memoryBytes), adjustedPriority, fn)
  }

  def newUnboundedTask[T](fn: () => T): AsyncTask[T] = {
    new AsyncTask[T](TaskResource.newCpuResource(0L), Float.MaxValue, fn)
  }
}

/**
 * Throttle interface to be implemented by different throttling strategies.
 *
 * Currently, only HostMemoryThrottle is implemented, which limits the maximum in-flight host
 * memory bytes. In the future, we can add more throttling strategies, such as limiting the
 * device memory usage, the number of tasks, etc.
 */
trait ResourceManager {
  /**
   * Returns true if the task can be accepted, false otherwise.
   * TrafficController will block the task from being scheduled until this method returns true.
   */
  def acquireResource[T](task: AsyncTask[T], timeout: Long): Boolean

  /**
   * Callback to be called when a task is completed, either successfully or with an exception.
   */
  def releaseResource[T](task: AsyncTask[T]): Unit
}

/**
 * Throttle implementation that limits the total host memory used by the in-flight tasks.
 */
class HostMemoryManager(val maxInFlightHostMemoryBytes: Long) extends ResourceManager {

  private val lock = new ReentrantLock()

  private val condition = lock.newCondition()

  @GuardedBy("lock")
  private var remaining: Long = maxInFlightHostMemoryBytes

  override def acquireResource[T](task: AsyncTask[T], timeoutMs: Long): Boolean = {
    task.resource.hostMemoryBytes match {
      case 0 =>
        true
      case required if required > maxInFlightHostMemoryBytes =>
        throw new IllegalArgumentException(
          s"Task requires more host memory than total size of memory pool: " +
              s"required=$required, maxAllowed=$maxInFlightHostMemoryBytes")
      case required: Long =>
        var isDone = false
        var isTimeout = false
        var waitTimeNs = TimeUnit.MILLISECONDS.toNanos(timeoutMs)
        lock.lockInterruptibly()
        try {
          while (!isDone && !isTimeout) {
            if (remaining >= required) {
              remaining -= required
              isDone = true
            } else if (waitTimeNs > 0) {
              waitTimeNs = condition.awaitNanos(waitTimeNs)
            } else {
              isTimeout = true
            }
          }
        } finally {
          lock.unlock()
        }
        isDone
    }
  }

  override def releaseResource[T](task: AsyncTask[T]): Unit = {
    if (task.resource.hostMemoryBytes > 0) {
      lock.lockInterruptibly()
      try {
        remaining += task.resource.hostMemoryBytes
        require(remaining <= maxInFlightHostMemoryBytes,
          s"Released more host memory than allowed: " +
              s"released=${task.resource.hostMemoryBytes}, remaining=$remaining, " +
              s"maxAllowed=$maxInFlightHostMemoryBytes")
        condition.signalAll()
      } finally {
        lock.unlock()
      }
    }
  }
}
