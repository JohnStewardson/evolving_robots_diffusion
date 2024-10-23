import multiprocessing
import time
import traceback
import gc

def job_wrapper_diff(func, args, result_queue):
    try:
        out_value = func(*args)
    except Exception as e:
        print("ERROR\n")
        traceback.print_exc()
        print()
        out_value = e  # Store exception for debugging

    result_queue.put(out_value)  # Put the result in the queue

class Group_DM:
    def __init__(self):
        print("Init of group")
        self.jobs = []
        self.result_queues = []  # Keep the queues alive
        self.callbacks = []

    def add_job(self, func, args, callback):
        print("Adding job")
        result_queue = multiprocessing.Queue()  # Create a new queue for this job's results
        job = multiprocessing.Process(target=job_wrapper_diff, args=(func, args, result_queue))
        self.jobs.append(job)
        self.result_queues.append(result_queue)
        self.callbacks.append(callback)

    def run_jobs(self, num_proc):
        print("In run_jobs")
        next_job = 0
        num_jobs_open = 0
        jobs_finished = 0

        jobs_open = set()
        print(f"jobs_open: {self.jobs}")
        while jobs_finished != len(self.jobs):
            jobs_closed = []
            for job_index in jobs_open:
                if not self.jobs[job_index].is_alive():
                    self.jobs[job_index].join()
                    self.jobs[job_index].terminate()
                    num_jobs_open -= 1
                    jobs_finished += 1
                    jobs_closed.append(job_index)
                    print("Job closed")
            for job_index in jobs_closed:
                jobs_open.remove(job_index)

            while num_jobs_open < num_proc and next_job != len(self.jobs):
                self.jobs[next_job].start()
                jobs_open.add(next_job)
                next_job += 1
                num_jobs_open += 1

            time.sleep(0.1)

        # Execute callbacks with the proper result values
        for i in range(len(self.jobs)):
            try:
                result = self.result_queues[i].get(timeout=10)  # Get result from the queue with timeout
                if isinstance(result, Exception):
                    print(f"Error in job {i}: {result}")
                else:
                    self.callbacks[i](result)
                print("Return callback of job")
            except Exception as e:
                print(f"Failed to get result for job {i}: {e}")

    def cleanup(self):
        print("Cleaning up resources...")
        # Ensure all jobs are terminated
        for job in self.jobs:
            if job.is_alive():
                job.terminate()
                job.join()

        # Do not delete queues immediately; keep them alive until the end
        time.sleep(1)  # Short delay to ensure all processes have completed cleanly
        self.jobs.clear()
        self.result_queues.clear()  # Clear queues after waiting
        self.callbacks.clear()

        gc.collect()
