"""
Data models for the Performance Optimization component.

This module provides models for request batching, rate limiting, and performance profiling.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid

from pydantic import BaseModel, Field


class RequestPriority(str, Enum):
    """Priority levels for requests."""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"


class BatchRequest(BaseModel):
    """
    Model for a batch of requests to be processed together.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requests: List[Dict[str, Any]]
    priority: RequestPriority = RequestPriority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.now)
    processing_started: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RateLimitRule(BaseModel):
    """
    Rule for rate limiting.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    requests_per_second: float
    burst_size: int = 10
    applies_to: Dict[str, Any] = Field(default_factory=dict)  # Criteria for when rule applies
    is_enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)


class RateLimitState(BaseModel):
    """
    Current state of a rate limiter.
    """
    rule_id: str
    tokens: float  # Current token count in the bucket
    last_updated: datetime = Field(default_factory=datetime.now)
    total_requests: int = 0
    total_limited: int = 0


class PerformanceMetrics(BaseModel):
    """
    Performance metrics for a component or operation.
    """
    component: str
    operation: str
    request_count: int = 0
    error_count: int = 0
    average_duration_ms: float = 0
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    p50_duration_ms: Optional[float] = None  # 50th percentile
    p95_duration_ms: Optional[float] = None  # 95th percentile
    p99_duration_ms: Optional[float] = None  # 99th percentile
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    first_request_at: Optional[datetime] = None
    last_request_at: Optional[datetime] = None


class PerformanceProfile(BaseModel):
    """
    Performance profile for a trace or operation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    operation_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)  # CPU, memory, etc.
    created_at: datetime = Field(default_factory=datetime.now)


class ResourceUtilization(BaseModel):
    """
    System resource utilization data.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_mb: float = 0.0
    disk_available_mb: float = 0.0
    network_sent_bytes: Optional[int] = None
    network_received_bytes: Optional[int] = None