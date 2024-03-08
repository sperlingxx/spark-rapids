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

package org.apache.spark.sql.rapids;

import java.util.Comparator;
import java.util.concurrent.*;

public class ComputeThreadPool {


	public static class TaskWithPriority<Result> {

		public TaskWithPriority(Callable<Result> task, int priority) {
			this.task = task;
			this.priority = priority;
			this.promise = new CompletableFuture<>();
		}

		private void run() {
			try {
				Result ret = task.call();
				promise.complete(ret);
			} catch (Exception e) {
				promise.completeExceptionally(e);
			}
		}

		public Future<Result> future() {
			return promise;
		}

		private final CompletableFuture<Result> promise;
		private final Callable<Result> task;
		private final int priority;
	}

	private ComputeThreadPool(int threadNum, int taskQueueCapacity) {
		this.taskQueue = new PriorityBlockingQueue<>(
				taskQueueCapacity, Comparator.comparingInt(o -> o.priority));
		this.threadGroup = new ThreadGroup("CPU_INTENSIVE_THREADS");
		this.workerSemaphore = new Semaphore(threadNum);

		workers = new Worker[threadNum];
		for (int i = 0; i < threadNum; ++i) {
			workers[i] = new Worker();
			workers[i].start();
		}
	}

	private class Worker extends Thread {

		Worker() {
			super(threadGroup, () -> {
				while (true) {
					try {
						TaskWithPriority<?> task = taskQueue.take();
						workerSemaphore.acquire();
						task.run();
						workerSemaphore.release();
					} catch (InterruptedException e) {
						throw new RuntimeException(e);
					}
				}
			});
		}
	}

	public void close() {
		for (Worker worker : workers) {
			if (worker.isAlive()) {
				worker.interrupt();
			}
		}
	}

	public static void submitTask(TaskWithPriority<?> task) {
		INSTANCE.taskQueue.add(task);
	}

	public static synchronized void launch(int threadNum, int taskQueueCapacity) {
		if (INSTANCE == null) {
			INSTANCE = new ComputeThreadPool(threadNum, taskQueueCapacity);
		}
	}

	public static boolean waitForIdleWorker(long timeout) {
		try {
			// Release the acquired semaphore instantly. Here we just leverage tryAcquire
			// interface to implement the wait for idle worker within given timeout.
			if (INSTANCE.workerSemaphore.tryAcquire(timeout, TimeUnit.MILLISECONDS)) {
				INSTANCE.workerSemaphore.release();
				return true;
			}
			return false;
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
	}

	public static ComputeThreadPool getInstance() {
		if (INSTANCE == null) {
			throw new RuntimeException("CpuIntensiveThreadPool is NOT initialized");
		}
		return INSTANCE;
	}

	private static ComputeThreadPool INSTANCE = null;

	private final Semaphore workerSemaphore;
	private final PriorityBlockingQueue<TaskWithPriority<?>> taskQueue;
	private final ThreadGroup threadGroup;
	private final Worker[] workers;

}
