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

import java.util.concurrent.{Callable, CompletionService, Future, FutureTask, LinkedBlockingQueue, TimeUnit}

class BoundedCompletionService[V](
    executor: ResourceBoundedThreadExecutor) extends CompletionService[V] {

  private val completionQueue = new LinkedBlockingQueue[Future[V]]()

  private class CompletionFutureTask(task: AsyncTask[V]) extends FutureTask[V](task) {
    private var completed: Boolean = false

    override def run(): Unit = {
      if (!completed) {
        super.run()
        completed = true
      }
    }

    override def done(): Unit = {
      completionQueue.offer(this)
      super.done()
    }
  }

  override def submit(task: Callable[V]): Future[V] = {
    task match {
      case asyncTask: AsyncTask[V] =>
        val futureTask = new CompletionFutureTask(asyncTask)
        executor.submit(futureTask, null)
      case _ =>
        throw new IllegalArgumentException("Task must be an instance of AsyncTask")
    }
  }

  override def submit(task: Runnable, result: V): Future[V] = {
    throw new UnsupportedOperationException("Runnable tasks are not supported")
  }

  override def take(): Future[V] = completionQueue.take()

  override def poll(): Future[V] = completionQueue.poll()

  override def poll(timeout: Long, unit: TimeUnit): Future[V] = {
    completionQueue.poll(timeout, unit)
  }
}
