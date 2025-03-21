"""
Batch Processor for efficient request handling.

This module provides batch processing capabilities, grouping similar requests
to reduce overhead and improve throughput.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union, Awaitable
import functools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from atia.config import settings
from atia.performance.models import BatchRequest, RequestPriority
from atia.utils.error_handling import catch_and_log


# Type variable for return values
T = TypeVar('T')

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes requests in batches for improved efficiency.
    """

    def __init__(self, 
                max_batch_size: int = settings.batch_size,
                max_wait_time: float = 0.1):
        """
        Initialize the batch processor.

        Args:
            max_batch_size: Maximum number of requests in a batch
            max_wait_time: Maximum time to wait for batch to fill up (in seconds)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batches = {
            RequestPriority.HIGH: [],
            RequestPriority.MEDIUM: [],
            RequestPriority.LOW: []
        }
        self.batch_ready_events = {
            RequestPriority.HIGH: asyncio.Event(),
            RequestPriority.MEDIUM: asyncio.Event(),
            RequestPriority.LOW: asyncio.Event()
        }
        self.batch_locks = {
            RequestPriority.HIGH: asyncio.Lock(),
            RequestPriority.MEDIUM: asyncio.Lock(),
            RequestPriority.LOW: asyncio.Lock()
        }
        self.is_running = False
        self.processor_task = None
        self.stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0,
            "max_batch_size": 0,
            "avg_processing_time": 0
        }

    async def start(self):
        """Start the batch processor."""
        if self.is_running:
            return

        self.is_running = True
        self.processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info("Batch processor started")

    async def stop(self):
        """Stop the batch processor."""
        if not self.is_running:
            return

        self.is_running = False
        if self.processor_task:
            await self.processor_task
            self.processor_task = None
        logger.info("Batch processor stopped")

    @catch_and_log(component="batch_processor")
    async def add_request(self, 
                        request: Dict[str, Any], 
                        priority: RequestPriority = RequestPriority.MEDIUM) -> str:
        """
        Add a request to the batch queue.

        Args:
            request: The request to add
            priority: Priority of the request

        Returns:
            Request ID
        """
        request_id = f"{int(time.time())}-{len(self.batches[priority])}"

        # Add metadata to the request
        request_with_meta = {
            "id": request_id,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "data": request
        }

        # Add to appropriate batch queue
        async with self.batch_locks[priority]:
            self.batches[priority].append(request_with_meta)

            # Signal batch is ready if full
            if len(self.batches[priority]) >= self.max_batch_size:
                self.batch_ready_events[priority].set()

        logger.debug(f"Added request {request_id} to {priority} batch queue")
        return request_id

    @catch_and_log(component="batch_processor")
    async def process_batch(self, 
                          batch: List[Dict[str, Any]],
                          process_fn: Callable[[List[Dict[str, Any]]], Awaitable[List[Any]]]) -> List[Any]:
        """
        Process a batch of requests.

        Args:
            batch: Batch of requests to process
            process_fn: Function to process the batch

        Returns:
            List of results
        """
        if not batch:
            return []

        start_time = time.time()

        # Create a batch request object
        batch_request = BatchRequest(
            requests=batch,
            priority=batch[0]["priority"],  # Use priority of first request
            processing_started=datetime.now()
        )

        try:
            # Process the batch
            results = await process_fn([r["data"] for r in batch])

            # Update batch request
            batch_request.completed_at = datetime.now()

            # Update statistics
            self._update_stats(batch_request, time.time() - start_time)

            return results
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return errors for all requests in batch
            return [{"error": str(e)} for _ in batch]

    async def _batch_processor_loop(self):
        """Main processor loop for handling batches."""
        while self.is_running:
            try:
                # Process high priority batches first
                if self.batches[RequestPriority.HIGH]:
                    await self._process_priority_queue(RequestPriority.HIGH)
                # Then medium priority
                elif self.batches[RequestPriority.MEDIUM]:
                    await self._process_priority_queue(RequestPriority.MEDIUM)
                # Then low priority
                elif self.batches[RequestPriority.LOW]:
                    await self._process_priority_queue(RequestPriority.LOW)
                else:
                    # No requests to process, wait for a short time
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                await asyncio.sleep(0.1)  # Avoid tight error loop

    async def _process_priority_queue(self, priority: RequestPriority):
        """Process a specific priority queue."""
        # Wait for either max batch size or max wait time
        if len(self.batches[priority]) < self.max_batch_size:
            # Start a timer
            wait_task = asyncio.create_task(asyncio.sleep(self.max_wait_time))
            batch_ready_task = asyncio.create_task(self.batch_ready_events[priority].wait())

            # Wait for either timeout or batch ready
            done, pending = await asyncio.wait(
                [wait_task, batch_ready_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the pending task
            for task in pending:
                task.cancel()

            # Reset the event
            self.batch_ready_events[priority].clear()

        # Get the batch to process
        async with self.batch_locks[priority]:
            # Take up to max_batch_size requests
            to_process = self.batches[priority][:self.max_batch_size]
            # Remove from queue
            self.batches[priority] = self.batches[priority][len(to_process):]

        # No requests to process after all
        if not to_process:
            return

        # Process the batch
        # This would be hooked up to actual processing in a real implementation
        logger.info(f"Would process batch of {len(to_process)} {priority} priority requests")

        # Simulate processing for testing
        await asyncio.sleep(0.05)

    def _update_stats(self, batch_request: BatchRequest, processing_time: float):
        """Update processor statistics."""
        batch_size = len(batch_request.requests)

        self.stats["total_batches"] += 1
        self.stats["total_requests"] += batch_size

        # Update average batch size
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1)) + batch_size
        ) / self.stats["total_batches"]

        # Update max batch size
        if batch_size > self.stats["max_batch_size"]:
            self.stats["max_batch_size"] = batch_size

        # Update average processing time
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (self.stats["total_batches"] - 1)) + processing_time
        ) / self.stats["total_batches"]

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        # Calculate queue sizes
        queue_sizes = {
            "high_priority": len(self.batches[RequestPriority.HIGH]),
            "medium_priority": len(self.batches[RequestPriority.MEDIUM]),
            "low_priority": len(self.batches[RequestPriority.LOW])
        }

        return {
            **self.stats,
            "queue_sizes": queue_sizes,
            "is_running": self.is_running
        }


def batch_decorator(batch_size: int = 5, 
                  timeout: float = 0.1):
    """
    Decorator for batch processing a function.

    Args:
        batch_size: Maximum number of requests in a batch
        timeout: Maximum time to wait for batch to fill up (in seconds)

    Returns:
        Decorated function
    """
    def decorator(func):
        # Store batched requests
        batched_requests = []
        # List to store corresponding futures
        futures = []
        # Lock for thread safety
        lock = asyncio.Lock()
        # Event to signal batch is ready
        batch_ready = asyncio.Event()
        # Task for batch processing
        processor_task = None

        async def process_batch():
            """Process a batch of requests."""
            nonlocal batched_requests, futures

            async with lock:
                # Get current batch
                current_batch = batched_requests.copy()
                current_futures = futures.copy()
                # Reset for next batch
                batched_requests.clear()
                futures.clear()
                # Reset event
                batch_ready.clear()

            if not current_batch:
                return

            try:
                # Process the batch
                results = await func(current_batch)

                # Set results on futures
                for future, result in zip(current_futures, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                # Set exception on all futures
                for future in current_futures:
                    if not future.done():
                        future.set_exception(e)

        @functools.wraps(func)
        async def wrapper(request):
            """Wrapper function that batches requests."""
            nonlocal processor_task, batched_requests, futures

            # Create a future for this request
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            async with lock:
                # Add request to batch
                batched_requests.append(request)
                futures.append(future)

                # Start processor task if batch is full or first request
                if len(batched_requests) >= batch_size:
                    batch_ready.set()
                elif len(batched_requests) == 1:
                    # Start timeout for first request
                    processor_task = asyncio.create_task(start_processing())

            # Wait for result
            return await future

        async def start_processing():
            """Start processing after timeout or when batch is full."""
            # Wait for either timeout or batch ready
            try:
                await asyncio.wait_for(batch_ready.wait(), timeout)
            except asyncio.TimeoutError:
                # Timeout reached, process whatever we have
                pass

            await process_batch()

        # For batch processing a list directly
        wrapper.process_batch = func

        return wrapper

    return decorator


class ParallelProcessor:
    """
    Processes tasks in parallel for improved throughput.
    """

    def __init__(self, 
                max_workers: int = settings.parallel_requests,
                use_processes: bool = False):
        """
        Initialize the parallel processor.

        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor = None
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_task_time": 0
        }

    def start(self):
        """Start the parallel processor."""
        if self.executor is None:
            if self.use_processes:
                self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            logger.info(f"Parallel processor started with {self.max_workers} workers ({'processes' if self.use_processes else 'threads'})")

    def stop(self):
        """Stop the parallel processor."""
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
            logger.info("Parallel processor stopped")

    @catch_and_log(component="parallel_processor")
    async def process_tasks(self, 
                          tasks: List[Callable[[], T]]) -> List[Union[T, Exception]]:
        """
        Process tasks in parallel.

        Args:
            tasks: List of task functions to process

        Returns:
            List of results or exceptions
        """
        if not self.executor:
            self.start()

        if not tasks:
            return []

        loop = asyncio.get_event_loop()
        results = []
        start_time = time.time()

        # Update stats
        self.stats["total_tasks"] += len(tasks)

        try:
            # Submit tasks to the executor
            futures = [loop.run_in_executor(self.executor, task) for task in tasks]

            # Wait for all tasks to complete
            completed_tasks = await asyncio.gather(*futures, return_exceptions=True)

            # Process results
            for result in completed_tasks:
                if isinstance(result, Exception):
                    self.stats["failed_tasks"] += 1
                    results.append(result)
                else:
                    self.stats["completed_tasks"] += 1
                    results.append(result)

            # Update timing stats
            task_time = (time.time() - start_time) / len(tasks)
            self.stats["avg_task_time"] = (
                (self.stats["avg_task_time"] * (self.stats["total_tasks"] - len(tasks))) + 
                (task_time * len(tasks))
            ) / self.stats["total_tasks"]

            return results
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Mark all tasks as failed
            self.stats["failed_tasks"] += len(tasks)
            return [e for _ in tasks]

    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processor statistics."""
        return {
            **self.stats,
            "is_running": self.executor is not None,
            "max_workers": self.max_workers,
            "worker_type": "processes" if self.use_processes else "threads"
        }


# Async to sync adapter for running sync code in the background
async def run_sync_in_background(func, *args, **kwargs):
    """Run a synchronous function in a background thread."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, lambda: func(*args, **kwargs)
        )