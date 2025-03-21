"""
Performance Optimization Module.

Provides tools for improving system efficiency through request batching,
rate limiting, parallel processing, and performance profiling.
"""

from atia.performance.models import (
    RequestPriority,
    BatchRequest,
    RateLimitRule,
    RateLimitState,
    PerformanceMetrics,
    PerformanceProfile,
    ResourceUtilization
)
from atia.performance.batch_processor import (
    BatchProcessor,
    batch_decorator,
    ParallelProcessor,
    run_sync_in_background
)
from atia.performance.rate_limiter import (
    TokenBucketRateLimiter,
    AdaptiveRateLimiter,
    RateLimiterRegistry,
    rate_limit,
    RateLimitExceededError
)
from atia.performance.profiler import (
    PerformanceProfiler,
    profile,
    profile_function,
    track_time,
    profiler  # Global profiler instance
)

__all__ = [
    # Models
    "RequestPriority",
    "BatchRequest",
    "RateLimitRule",
    "RateLimitState",
    "PerformanceMetrics",
    "PerformanceProfile",
    "ResourceUtilization",

    # Batch Processing
    "BatchProcessor",
    "batch_decorator",
    "ParallelProcessor",
    "run_sync_in_background",

    # Rate Limiting
    "TokenBucketRateLimiter",
    "AdaptiveRateLimiter",
    "RateLimiterRegistry",
    "rate_limit",
    "RateLimitExceededError",

    # Performance Profiling
    "PerformanceProfiler",
    "profile",
    "profile_function",
    "track_time",
    "profiler"
]