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

package com.nvidia.spark.rapids

import scala.collection.mutable

import com.nvidia.spark.rapids.io.async.{AsyncTask, HostMemoryPool, ResourceBoundedThreadExecutor}
import org.scalatest.funsuite.AnyFunSuite

class ResourceBoundedExecutorSuite extends AnyFunSuite with RmmSparkRetrySuiteBase {

  test("AsyncCpuTask priority penalty based on memory usage") {
    new HostMemoryPool(10L << 20)

    val executor = ResourceBoundedThreadExecutor("single-thread-executor",
      new HostMemoryPool(100L << 20),
      maxThreadNumber = 1,
      waitResourceTimeoutMs = 0,
      priorityPenalty = 0.0f)

    val queue = mutable.Queue[Int]()
    def fnBuilder(index: Int): () => Int = {
      () => {
        queue.enqueue(index)
        Thread.sleep(50)
        index
      }
    }

    (1 to 5).foreach { i =>
      executor.submit(AsyncTask.newCpuTask(fnBuilder(i), i << 20))
    }
    assertResult(Array(1, 2, 3, 4, 5))(queue.toArray)

    queue.clear()
    (5 to 1 by -1).foreach { i =>
      executor.submit(AsyncTask.newCpuTask(fnBuilder(i), i << 20))
    }
    assertResult(Array(5, 1, 2, 3, 4))(queue.toArray)
  }
}
