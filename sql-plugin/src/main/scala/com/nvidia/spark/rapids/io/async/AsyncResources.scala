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

import org.apache.spark.internal.Logging

case class TaskResource(hostMemoryBytes: Long, requireGpuSemaphore: Boolean)

object TaskResource {
  def newCpuResource(hostMemoryBytes: Long): TaskResource = {
    new TaskResource(hostMemoryBytes, false)
  }

  def newGpuResource(): TaskResource = new TaskResource(0L, true)
}

case class AsyncResult[T](
    result: T,
    releaseResourceCallback: Option[() => Unit])

/**
 * The AsyncTask interface represents a task that can be scheduled by Resource.
 */
trait AsyncTask[T] extends Callable[AsyncResult[T]] {
  /**
   * Resource required by the task, such as host memory or GPU semaphore.
   */
  def resource: TaskResource

  /**
   * Priority of the task, higher value means higher priority.
   */
  def priority: Float

  /**
   * The abstract method defines the actual task logic, which should be implemented by the
   * subclass.
   */
  protected def callImpl(): T

  def call(): AsyncResult[T] = {
    val result = callImpl()
    if (holdResourceAfterCompletion) {
      require(releaseResourceCallback != null,
        "Release callback is not set, please set it before calling fetchReleaseCallback")
      AsyncResult(result, Some(releaseResource))
    } else {
      AsyncResult(result, None)
    }
  }

  /**
   * Returns true if the task should release its resources after completion.
   * This is useful for tasks that are not long-lived and can release resources
   * immediately after they finish.
   */
  def holdResourceAfterCompletion: Boolean = false

  /**
   * Guarantees the release callback will NOT be called unless the task holds the resource.
   */
  private def releaseResource(): Unit = {
    require(holdResource, "Task does NOT hold resource after completion")
    require(releaseResourceCallback != null, "releaseResourceCallback is not registered")
    releaseResourceCallback()
    holdResource = false
  }

  private[async] var releaseResourceCallback: () => Unit = _

  private[async] var holdResource = false
}

abstract class UnboundedAsyncTask[T] extends AsyncTask[T] {
  /**
   * Unbounded tasks do not have a resource limit, so they can be scheduled without any
   * restrictions.
   */
  override val resource: TaskResource = TaskResource.newCpuResource(0L)

  /**
   * Unbounded tasks have the highest priority.
   */
  override val priority: Float = Float.MaxValue
}

class AsyncFunctor[T](
    override val resource: TaskResource,
    override val priority: Float,
    functor: () => T) extends AsyncTask[T] {
  override def callImpl(): T = functor()
}

object AsyncTask {
  // Adjust the priority based on the memory overhead to minimal the potential clogging:
  // lightweight tasks should have higher priority
  def hostMemoryPenalty(memoryBytes: Long, priority: Float = 0.0f): Float = {
    require(memoryBytes >= 0, s"Memory bytes must be non-negative, got: $memoryBytes")
    priority - math.log10(memoryBytes).toFloat
  }

  def newCpuTask[T](fn: () => T,
      memoryBytes: Long,
      priority: Float = 0.0f): AsyncTask[T] = {
    val adjustedPriority = hostMemoryPenalty(memoryBytes, priority)
    new AsyncFunctor[T](TaskResource.newCpuResource(memoryBytes), adjustedPriority, fn)
  }

  def newUnboundedTask[T](fn: () => T): AsyncTask[T] = {
    new AsyncFunctor[T](TaskResource.newCpuResource(0L), Float.MaxValue, fn)
  }
}

// Being thrown when a task requests resources that are not valid or exceed the limits
class InvalidResourceRequest(msg: String) extends RuntimeException(
  s"Invalid resource request: $msg")

// Represents the status of acquiring resources for a task
sealed trait AcquireStatus
case object AcquireSuccessful extends AcquireStatus
// AcquireFailed indicates that the task could not be scheduled due to resource constraints
case object AcquireFailed extends AcquireStatus
// AcquireExcepted indicates that an exception occurred while trying to acquire resources
case class AcquireExcepted(exception: Throwable) extends AcquireStatus

/**
 * ResourceManager interface to be implemented for AsyncTasks requiring different kinds of
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
  def acquireResource[T](task: AsyncTask[T], timeout: Long): AcquireStatus

  /**
   * Callback to be called when a task is completed, either successfully or with an exception.
   */
  def releaseResource[T](task: AsyncTask[T]): Unit
}

/**
 * Throttle implementation that limits the total host memory used by the in-flight tasks.
 */
class HostMemoryPool(val maxHostMemoryBytes: Long) extends ResourcePool with Logging {

  private val lock = new ReentrantLock()

  private val condition = lock.newCondition()

  @GuardedBy("lock")
  private var remaining: Long = maxHostMemoryBytes

  override def acquireResource[T](task: AsyncTask[T], timeoutMs: Long): AcquireStatus = {
    task.resource.hostMemoryBytes match {
      case 0 =>
        AcquireSuccessful
      case required if required > maxHostMemoryBytes =>
        // Call the failure callback to notify the caller about the failure
        AcquireExcepted(
          new InvalidResourceRequest(
            s"Task requires more host memory($required) than pool size($maxHostMemoryBytes)"))
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
              val beforeWait = waitTimeNs
              waitTimeNs = condition.awaitNanos(waitTimeNs)
              logWarning(s"Waiting ${TimeUnit.NANOSECONDS.toMillis(beforeWait - waitTimeNs)}ms " +
                  s"for HostMemory: required=${required >> 10}KB, remaining=${remaining >> 10}KB")
            } else {
              isTimeout = true
            }
          }
          if (isDone) AcquireSuccessful else AcquireFailed
        } catch {
          case ex: Throwable => AcquireExcepted(ex)
        } finally {
          lock.unlock()
        }
    }
  }

  override def releaseResource[T](task: AsyncTask[T]): Unit = {
    if (task.resource.hostMemoryBytes > 0) {
      lock.lockInterruptibly()
      try {
        remaining += task.resource.hostMemoryBytes
        require(remaining <= maxHostMemoryBytes,
          s"Released more host memory than allowed: " +
              s"released=${task.resource.hostMemoryBytes}, remaining=$remaining, " +
              s"maxAllowed=$maxHostMemoryBytes")
        condition.signalAll()
      } finally {
        lock.unlock()
      }
    }
  }

  override def toString: String = {
    s"HostMemoryPool(maxHostMemoryBytes=${maxHostMemoryBytes >> 20}MB)"
  }
}
