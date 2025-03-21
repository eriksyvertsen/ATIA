"""
Performance Profiler for monitoring and analyzing system performance.

This module provides performance profiling capabilities, tracking execution time,
memory usage, and other metrics to identify bottlenecks.
"""

import asyncio
import cProfile
import io
import json
import logging
import os
import pstats
import time
import tracemalloc
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union, Awaitable
from contextlib import contextmanager
import functools

from atia.config import settings
from atia.performance.models import PerformanceMetrics, PerformanceProfile, ResourceUtilization
from atia.utils.error_handling import catch_and_log


# Type variable for return values
T = TypeVar('T')

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Monitors and analyzes system performance.
    """

    def __init__(self, storage_dir: str = "data/performance"):
        """
        Initialize the performance profiler.

        Args:
            storage_dir: Directory to store performance profiles
        """
        self.storage_dir = storage_dir
        self.metrics_dir = os.path.join(storage_dir, "metrics")
        self.profiles_dir = os.path.join(storage_dir, "profiles")
        self.resources_dir = os.path.join(storage_dir, "resources")

        # Create storage directories
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.resources_dir, exist_ok=True)

        # Metrics storage
        self.metrics: Dict[str, PerformanceMetrics] = {}

        # Resource monitoring
        self.resource_monitoring = False
        self.resource_monitor_task = None
        self.resource_monitor_interval = 10  # seconds

        # Memory profiling
        self.memory_profiling = False

        # Load existing metrics
        self._load_metrics()

    @catch_and_log(component="performance_profiler")
    async def start_resource_monitoring(self, interval_seconds: int = 10) -> None:
        """
        Start monitoring system resources.

        Args:
            interval_seconds: Interval between resource checks in seconds
        """
        if self.resource_monitoring:
            return

        self.resource_monitoring = True
        self.resource_monitor_interval = interval_seconds
        self.resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        logger.info(f"Resource monitoring started with {interval_seconds}s interval")

    @catch_and_log(component="performance_profiler")
    async def stop_resource_monitoring(self) -> None:
        """Stop monitoring system resources."""
        if not self.resource_monitoring:
            return

        self.resource_monitoring = False
        if self.resource_monitor_task:
            await self.resource_monitor_task
            self.resource_monitor_task = None
        logger.info("Resource monitoring stopped")

    @catch_and_log(component="performance_profiler")
    def start_memory_profiling(self) -> None:
        """Start memory profiling."""
        if self.memory_profiling:
            return

        tracemalloc.start()
        self.memory_profiling = True
        logger.info("Memory profiling started")

    @catch_and_log(component="performance_profiler")
    def stop_memory_profiling(self) -> None:
        """Stop memory profiling."""
        if not self.memory_profiling:
            return

        tracemalloc.stop()
        self.memory_profiling = False
        logger.info("Memory profiling stopped")

    @catch_and_log(component="performance_profiler")
    def take_memory_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of memory usage.

        Returns:
            Memory usage statistics
        """
        if not self.memory_profiling:
            self.start_memory_profiling()

        # Take snapshot
        snapshot = tracemalloc.take_snapshot()

        # Get top statistics
        top_stats = snapshot.statistics('lineno')

        # Convert to serializable format
        result = {
            "timestamp": datetime.now().isoformat(),
            "total_allocated": sum(stat.size for stat in top_stats),
            "top_allocations": [
                {
                    "file": str(stat.traceback.frame.filename),
                    "line": stat.traceback.frame.lineno,
                    "size": stat.size,
                    "count": stat.count
                }
                for stat in top_stats[:10]  # Top 10 allocations
            ]
        }

        # Save snapshot
        self._save_memory_snapshot(result)

        return result

    @contextmanager
    def profile_code(self, name: str) -> None:
        """
        Context manager for profiling a block of code.

        Args:
            name: Name of the profile
        """
        # Start profiler
        profiler = cProfile.Profile()
        profiler.enable()

        # Start time
        start_time = time.time()

        try:
            # Yield control back to the code being profiled
            yield
        finally:
            # Stop profiler
            profiler.disable()

            # Calculate duration
            duration = time.time() - start_time

            # Process results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 functions

            # Get function stats
            function_stats = []
            for func, (cc, nc, tt, ct, callers) in profiler.getstats():
                if func:
                    func_name = func.__name__
                    if hasattr(func, '__module__'):
                        module_name = func.__module__ or '<unknown>'
                        func_name = f"{module_name}.{func_name}"

                    function_stats.append({
                        "function": func_name,
                        "calls": cc,
                        "time": ct,
                        "time_per_call": ct / cc if cc > 0 else 0
                    })

            # Sort by time
            function_stats.sort(key=lambda x: x["time"], reverse=True)

            # Create profile
            profile = PerformanceProfile(
                name=name,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(start_time + duration),
                duration_ms=duration * 1000,
                operation_type="code_block",
                steps=function_stats[:20],  # Top 20 functions
                resources={
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent
                }
            )

            # Save profile
            self._save_profile(profile)

            logger.info(f"Profile '{name}' completed in {duration:.4f}s")

    @catch_and_log(component="performance_profiler")
    def profile_sync_function(self, name: Optional[str] = None) -> Callable:
        """
        Decorator for profiling a synchronous function.

        Args:
            name: Name of the profile (defaults to function name)

        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                profile_name = name or f"{func.__module__}.{func.__name__}"

                with self.profile_code(profile_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    @catch_and_log(component="performance_profiler")
    def profile_async_function(self, name: Optional[str] = None) -> Callable:
        """
        Decorator for profiling an asynchronous function.

        Args:
            name: Name of the profile (defaults to function name)

        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                profile_name = name or f"{func.__module__}.{func.__name__}"

                # Start time
                start_time = time.time()

                # Get initial memory if profiling is active
                if self.memory_profiling:
                    tracemalloc.clear_traces()
                    memory_before = tracemalloc.get_traced_memory()[0]

                try:
                    # Call the function
                    result = await func(*args, **kwargs)

                    # Calculate duration
                    duration = time.time() - start_time

                    # Get memory usage if profiling is active
                    memory_used = None
                    if self.memory_profiling:
                        memory_after = tracemalloc.get_traced_memory()[0]
                        memory_used = memory_after - memory_before

                    # Create profile
                    profile = PerformanceProfile(
                        name=profile_name,
                        start_time=datetime.fromtimestamp(start_time),
                        end_time=datetime.fromtimestamp(time.time()),
                        duration_ms=duration * 1000,
                        operation_type="async_function",
                        resources={
                            "cpu_percent": psutil.cpu_percent(),
                            "memory_percent": psutil.virtual_memory().percent,
                            "memory_used": memory_used
                        }
                    )

                    # Save profile
                    self._save_profile(profile)

                    # Update metrics
                    self._update_metrics(profile_name, "async_function", duration * 1000)

                    logger.debug(f"Profile '{profile_name}' completed in {duration:.4f}s")

                    return result
                except Exception as e:
                    # Calculate duration
                    duration = time.time() - start_time

                    # Update metrics with error
                    self._update_metrics(profile_name, "async_function", duration * 1000, error=True)

                    logger.warning(f"Profile '{profile_name}' failed after {duration:.4f}s: {e}")

                    # Re-raise the exception
                    raise

            return wrapper

        return decorator

    @catch_and_log(component="performance_profiler")
    def track_operation(self, 
                      component: str, 
                      operation: str, 
                      duration_ms: float,
                      error: bool = False,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track performance metrics for an operation.

        Args:
            component: Component name
            operation: Operation name
            duration_ms: Duration in milliseconds
            error: Whether an error occurred
            metadata: Additional metadata
        """
        # Create metrics key
        key = f"{component}:{operation}"

        # Update metrics
        self._update_metrics(key, component, duration_ms, error)

    @catch_and_log(component="performance_profiler")
    def get_metrics(self, 
                  component: Optional[str] = None) -> List[PerformanceMetrics]:
        """
        Get performance metrics.

        Args:
            component: Optional component filter

        Returns:
            List of performance metrics
        """
        if component:
            # Return metrics for specific component
            return [m for m in self.metrics.values() if m.component == component]
        else:
            # Return all metrics
            return list(self.metrics.values())

    @catch_and_log(component="performance_profiler")
    def get_recent_profiles(self, 
                          count: int = 10) -> List[PerformanceProfile]:
        """
        Get recent performance profiles.

        Args:
            count: Number of profiles to return

        Returns:
            List of recent performance profiles
        """
        try:
            # Get profile files
            profile_files = sorted(
                [f for f in os.listdir(self.profiles_dir) if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(self.profiles_dir, x)),
                reverse=True
            )

            # Load profiles
            profiles = []
            for filename in profile_files[:count]:
                try:
                    with open(os.path.join(self.profiles_dir, filename), 'r') as f:
                        profile_data = json.load(f)

                        # Convert timestamps
                        for key in ['start_time', 'end_time', 'created_at']:
                            if key in profile_data and isinstance(profile_data[key], str):
                                profile_data[key] = datetime.fromisoformat(profile_data[key])

                        # Create profile
                        profile = PerformanceProfile(**profile_data)
                        profiles.append(profile)
                except Exception as e:
                    logger.error(f"Error loading profile {filename}: {e}")

            return profiles
        except Exception as e:
            logger.error(f"Error getting recent profiles: {e}")
            return []

    @catch_and_log(component="performance_profiler")
    def get_recent_resource_data(self, 
                               count: int = 10) -> List[ResourceUtilization]:
        """
        Get recent resource utilization data.

        Args:
            count: Number of data points to return

        Returns:
            List of recent resource utilization data
        """
        try:
            # Get resource files
            resource_files = sorted(
                [f for f in os.listdir(self.resources_dir) if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(self.resources_dir, x)),
                reverse=True
            )

            # Load resource data
            resources = []
            for filename in resource_files[:count]:
                try:
                    with open(os.path.join(self.resources_dir, filename), 'r') as f:
                        resource_data = json.load(f)

                        # Convert timestamp
                        if 'timestamp' in resource_data and isinstance(resource_data['timestamp'], str):
                            resource_data['timestamp'] = datetime.fromisoformat(resource_data['timestamp'])

                        # Create resource utilization
                        resource = ResourceUtilization(**resource_data)
                        resources.append(resource)
                except Exception as e:
                    logger.error(f"Error loading resource data {filename}: {e}")

            return resources
        except Exception as e:
            logger.error(f"Error getting recent resource data: {e}")
            return []

    async def _resource_monitor_loop(self) -> None:
        """Background task for monitoring system resources."""
        try:
            while self.resource_monitoring:
                # Collect resource data
                try:
                    # Get CPU usage
                    cpu_percent = psutil.cpu_percent(interval=0.1)

                    # Get memory usage
                    memory = psutil.virtual_memory()

                    # Get disk usage
                    disk = psutil.disk_usage('/')

                    # Get network usage if available
                    network_sent = None
                    network_received = None
                    try:
                        net_io = psutil.net_io_counters()
                        network_sent = net_io.bytes_sent
                        network_received = net_io.bytes_recv
                    except:
                        pass

                    # Create resource utilization object
                    resource = ResourceUtilization(
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_used_mb=memory.used / (1024 * 1024),
                        memory_available_mb=memory.available / (1024 * 1024),
                        disk_percent=disk.percent,
                        disk_used_mb=disk.used / (1024 * 1024),
                        disk_available_mb=disk.free / (1024 * 1024),
                        network_sent_bytes=network_sent,
                        network_received_bytes=network_received
                    )

                    # Save resource data
                    self._save_resource_data(resource)

                    # Log if resources are getting low
                    if cpu_percent > 80:
                        logger.warning(f"High CPU usage: {cpu_percent}%")
                    if memory.percent > 80:
                        logger.warning(f"High memory usage: {memory.percent}%")
                    if disk.percent > 90:
                        logger.warning(f"Low disk space: {disk.free / (1024*1024*1024):.2f} GB free")
                except Exception as e:
                    logger.error(f"Error collecting resource data: {e}")

                # Wait for next interval
                await asyncio.sleep(self.resource_monitor_interval)
        except asyncio.CancelledError:
            logger.info("Resource monitoring task cancelled")
        except Exception as e:
            logger.error(f"Resource monitoring task failed: {e}")

    def _update_metrics(self, 
                     key: str, 
                     component: str, 
                     duration_ms: float,
                     error: bool = False) -> None:
        """Update performance metrics."""
        # Create or get metrics
        if key not in self.metrics:
            self.metrics[key] = PerformanceMetrics(
                component=component,
                operation=key.split(':')[1] if ':' in key else key,
                first_request_at=datetime.now(),
                last_request_at=datetime.now()
            )

        metrics = self.metrics[key]

        # Update counts
        metrics.request_count += 1
        if error:
            metrics.error_count += 1

        # Update duration stats
        if metrics.min_duration_ms is None or duration_ms < metrics.min_duration_ms:
            metrics.min_duration_ms = duration_ms

        if metrics.max_duration_ms is None or duration_ms > metrics.max_duration_ms:
            metrics.max_duration_ms = duration_ms

        # Update average
        metrics.average_duration_ms = (
            (metrics.average_duration_ms * (metrics.request_count - 1)) + duration_ms
        ) / metrics.request_count

        # Update last request time
        metrics.last_request_at = datetime.now()

        # Save metrics periodically
        if metrics.request_count % 10 == 0:
            self._save_metrics()

    def _save_profile(self, profile: PerformanceProfile) -> None:
        """Save a performance profile to disk."""
        try:
            # Create filename
            filename = f"{int(time.time())}_{profile.name.replace(':', '_')}.json"
            filepath = os.path.join(self.profiles_dir, filename)

            # Save to file
            with open(filepath, 'w') as f:
                # Convert to dict
                profile_dict = profile.model_dump()

                # Convert timestamps
                for key in ['start_time', 'end_time', 'created_at']:
                    if key in profile_dict and isinstance(profile_dict[key], datetime):
                        profile_dict[key] = profile_dict[key].isoformat()

                json.dump(profile_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profile: {e}")

    def _save_resource_data(self, resource: ResourceUtilization) -> None:
        """Save resource utilization data to disk."""
        try:
            # Create filename
            filename = f"resource_{int(time.time())}.json"
            filepath = os.path.join(self.resources_dir, filename)

            # Save to file
            with open(filepath, 'w') as f:
                # Convert to dict
                resource_dict = resource.model_dump()

                # Convert timestamp
                if 'timestamp' in resource_dict and isinstance(resource_dict['timestamp'], datetime):
                    resource_dict['timestamp'] = resource_dict['timestamp'].isoformat()

                json.dump(resource_dict, f, indent=2)

            # Limit the number of resource files
            self._cleanup_old_files(self.resources_dir, 1000)  # Keep last 1000 resource files
        except Exception as e:
            logger.error(f"Error saving resource data: {e}")

    def _save_memory_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Save a memory snapshot to disk."""
        try:
            # Create filename
            filename = f"memory_{int(time.time())}.json"
            filepath = os.path.join(self.profiles_dir, filename)

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2)

            # Limit the number of profile files
            self._cleanup_old_files(self.profiles_dir, 1000)  # Keep last 1000 profile files
        except Exception as e:
            logger.error(f"Error saving memory snapshot: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            # Create metrics file
            metrics_file = os.path.join(self.metrics_dir, "metrics.json")

            # Save to file
            with open(metrics_file, 'w') as f:
                metrics_data = []

                for key, metrics in self.metrics.items():
                    # Convert to dict
                    metrics_dict = metrics.model_dump()

                    # Convert timestamps
                    for key in ['first_request_at', 'last_request_at']:
                        if key in metrics_dict and metrics_dict[key] is not None:
                            metrics_dict[key] = metrics_dict[key].isoformat()

                    metrics_data.append(metrics_dict)

                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _load_metrics(self) -> None:
        """Load metrics from disk."""
        try:
            metrics_file = os.path.join(self.metrics_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)

                # Process each metrics object
                for metrics_dict in metrics_data:
                    try:
                        # Convert timestamps
                        for key in ['first_request_at', 'last_request_at']:
                            if key in metrics_dict and metrics_dict[key] is not None:
                                metrics_dict[key] = datetime.fromisoformat(metrics_dict[key])

                        # Create metrics object
                        metrics = PerformanceMetrics(**metrics_dict)

                        # Store in memory
                        key = f"{metrics.component}:{metrics.operation}"
                        self.metrics[key] = metrics
                    except Exception as e:
                        logger.error(f"Error loading metrics item: {e}")

                logger.info(f"Loaded {len(self.metrics)} performance metrics")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")

    def _cleanup_old_files(self, directory: str, max_files: int) -> None:
        """Clean up old files, keeping only the most recent ones."""
        try:
            # Get all files
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)

            # Remove oldest files if we have too many
            if len(files) > max_files:
                for filepath in files[:(len(files) - max_files)]:
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error removing old file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")


# Global profiler instance for easy access
profiler = PerformanceProfiler()


@contextmanager
def profile(name: str) -> None:
    """
    Context manager for profiling a block of code.

    Args:
        name: Name of the profile
    """
    with profiler.profile_code(name):
        yield


def profile_function(name: Optional[str] = None) -> Callable:
    """
    Decorator for profiling a function.

    Args:
        name: Name of the profile (defaults to function name)

    Returns:
        Decorated function
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return profiler.profile_async_function(name)(func)
        else:
            return profiler.profile_sync_function(name)(func)

    return decorator


def track_time(component: str, operation: str) -> Callable:
    """
    Decorator for tracking execution time of a function.

    Args:
        component: Component name
        operation: Operation name

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                profiler.track_operation(component, operation, duration_ms, error=True)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                profiler.track_operation(component, operation, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                profiler.track_operation(component, operation, duration_ms, error=True)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                profiler.track_operation(component, operation, duration_ms)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator