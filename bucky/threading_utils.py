import random
import threading
import queue
import time
from typing import Callable, Dict, Tuple, List, Any


class ThreadWorkerPool:
    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.workers: List[threading.Thread] = []

        self.input_queue: queue.Queue = queue.Queue()
        self.next_input_index: int = 0

        self.output_queue: queue.Queue = queue.Queue()
        self.next_output_index: int = 0

    def worker(self, worker_index: int) -> None:
        while True:
            task = self.input_queue.get()
            if task is None:
                break  # Stop the worker

            index, job, process_job, callback = task

            result = None
            try:
                result = process_job(worker_index, job)
            except Exception as ex:
                result = ex

            # Put the result in the output queue (preserving order)
            self.output_queue.put((index, result, callback))

            self.input_queue.task_done()

    def callback_handler(self) -> None:
        """Continuously processes completed jobs in order."""
        results: Dict[int, Tuple[Any, Callable[[Any], None]]] = {}
        while True:
            job_result = self.output_queue.get()
            if job_result is None:
                self.output_queue.task_done()
                break  # Stop the worker

            index, result, callback = job_result

            results[index] = (result, callback)
            # Call the callbacks in the correct order
            while self.next_output_index in results:
                result, callback = results.pop(self.next_output_index)
                self.next_output_index += 1
                try:
                    callback(result)
                except Exception as ex:
                    print(str(ex))
            self.output_queue.task_done()

    def start(self) -> None:
        """Start worker threads and callback handler."""
        self.workers = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(self.num_workers)]
        for worker in self.workers:
            worker.start()

        self.callback_thread = threading.Thread(target=self.callback_handler, daemon=True)
        self.callback_thread.start()

    def add_job(self, job: Any, process_job: Callable[[int, Any], Any], callback: Callable[[Any], None]) -> None:
        """Add a job to the input queue."""
        self.input_queue.put((self.next_input_index, job, process_job, callback))
        self.next_input_index += 1

    def wait_completion(self) -> None:
        """Wait for all jobs to be processed."""
        self.input_queue.join()
        self.output_queue.join()

    def stop(self) -> None:
        """Stop the worker pool and allow graceful shutdown."""
        self.wait_completion()
        for _ in range(self.num_workers):
            self.input_queue.put(None)  # Unblock workers
        for worker in self.workers:
            worker.join()
        self.output_queue.put(None)
        self.callback_thread.join()


def main() -> None:
    def example_process_job(worker_index: int, job: int) -> str:
        print("calc", worker_index, "job", job)
        time.sleep(random.random() + 0.5)  # Simulate work
        return f"result {job * 2}"  # Example processing

    def job_finished(result: str) -> None:
        print(f"Job finished with result: {result}")

    num_workers = 4
    jobs = [1, 2, 3, 4, 5, 6, 7, 8]

    pool = ThreadWorkerPool(num_workers)
    pool.start()

    for job in jobs:
        pool.add_job(job, example_process_job, job_finished)

    pool.stop()


if __name__ == "__main__":
    main()
