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

import java.util.concurrent.{BlockingQueue, Callable, FutureTask, LinkedBlockingQueue, RunnableFuture, ThreadFactory, ThreadPoolExecutor, TimeUnit}

import com.google.common.util.concurrent.ThreadFactoryBuilder


class RapidsFutureTask[T](val task: AsyncTask[T]) extends FutureTask[T](task)
    with Comparable[RapidsFutureTask[T]] {
  private var priority: Float = task.priority
  private var heldResource: Boolean = false
  private var completed: Boolean = false

  override def run(): Unit = {
    require(!completed, "Task has already been completed")
    if (heldResource) {
      super.run()
      completed = true
    }
  }

  def holdResource(): Unit = {
    require(!completed, "Task has already been completed")
    require(!heldResource, "Cannot hold resource that is already held")
    heldResource = true
  }

  def releaseResource(): Unit = {
    require(heldResource, "Cannot release resource that was not held")
    heldResource = false
  }

  def adjustPriority(delta: Float): Float = {
    require(!completed, "Task has already been completed")
    priority += delta
    priority
  }

  def isHeldResource: Boolean = heldResource

  def isCompleted: Boolean = completed

  override def compareTo(o: RapidsFutureTask[T]): Int = {
    priority.compareTo(o.priority)
  }
}

class ResourceBoundedThreadExecutor(mgr: ResourcePool,
    waitResourceTimeoutMs: Long,
    priorityPenalty: Float,
    corePoolSize: Int,
    maximumPoolSize: Int,
    workQueue: BlockingQueue[Runnable],
    threadFactory: ThreadFactory,
    keepAliveTime: Long = 100L) extends ThreadPoolExecutor(corePoolSize,
  maximumPoolSize, keepAliveTime, TimeUnit.SECONDS, workQueue, threadFactory) {

  override protected def newTaskFor[T](fn: Callable[T]): RunnableFuture[T] = {
    fn match {
      case task: AsyncTask[T] =>
        new RapidsFutureTask(task)
      case f =>
        throw new IllegalArgumentException(
          s"Callable must be of type Task, but got ${f.getClass.getName}")
    }
  }

  override def beforeExecute(t: Thread, r: Runnable): Unit = {
    r match {
      case fut: RapidsFutureTask[_] =>
        if (mgr.acquireResource(fut.task, waitResourceTimeoutMs)) {
          fut.holdResource()
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Runnable must be of type RapidsFutureTask, but got ${r.getClass.getName}")
    }
  }

  override def afterExecute(r: Runnable, t: Throwable): Unit = {
    r match {
      case fut: RapidsFutureTask[_] =>
        if (fut.isHeldResource) {
          mgr.releaseResource(fut.task)
          fut.releaseResource()
        }
        // If the task failed to acquire enough resource, we bypass the execution and re-add it to
        // the task queue with a priority penalty to avoid starvation.
        if (t == null && !fut.isCompleted) {
          fut.adjustPriority(-priorityPenalty)
          require(workQueue.add(fut),
            s"Failed to re-add task ${fut.task} to the work queue after execution")
        }
      case _ =>
        throw new IllegalArgumentException(
          s"Runnable must be of type RapidsFutureTask, but got ${r.getClass.getName}")
    }
  }
}

object ResourceBoundedThreadExecutor {
  def apply(name: String,
      pool: ResourcePool,
      maxThreadNumber: Int,
      waitResourceTimeoutMs: Long = 60 * 1000L,
      priorityPenalty: Float = 10.0f): ResourceBoundedThreadExecutor = {
    val taskQueue = new LinkedBlockingQueue[Runnable]()
    val threadFactory: ThreadFactory = new ThreadFactoryBuilder()
        .setDaemon(true)
        .setNameFormat(name)
        .build()

    new ResourceBoundedThreadExecutor(pool,
      waitResourceTimeoutMs,
      priorityPenalty,
      corePoolSize = maxThreadNumber,
      maximumPoolSize = maxThreadNumber,
      workQueue = taskQueue,
      threadFactory = threadFactory)
  }
}
