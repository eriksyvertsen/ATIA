"""
API endpoints for the Performance Optimization Module.

This module provides FastAPI endpoints for managing rate limits, performance monitoring,
and system resource tracking.
"""

import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request
from fastapi.security import OAuth2PasswordBearer

from atia.performance.models import (
    RateLimitRule,
    RateLimitState,
    PerformanceMetrics,
    PerformanceProfile,
    ResourceUtilization
)
from atia.performance.rate_limiter import RateLimiterRegistry
from atia.performance.profiler import profiler


# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/performance",
    tags=["performance"],
    responses={404: {"description": "Not found"}}
)

# Authentication scheme (to be integrated with API Gateway)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Dependency for rate limiter registry
def get_rate_limiter_registry():
    """Get a rate limiter registry instance."""
    return RateLimiterRegistry()


@router.get("/metrics", response_model=List[PerformanceMetrics])
async def get_performance_metrics(
    component: Optional[str] = Query(None, description="Filter by component name")
):
    """
    Get performance metrics.

    Args:
        component: Optional component filter

    Returns:
        List of performance metrics
    """
    metrics = profiler.get_metrics(component)

    return metrics


@router.get("/profiles", response_model=List[PerformanceProfile])
async def get_performance_profiles(
    count: int = Query(10, description="Number of profiles to return", ge=1, le=100)
):
    """
    Get recent performance profiles.

    Args:
        count: Number of profiles to return

    Returns:
        List of recent performance profiles
    """
    profiles = profiler.get_recent_profiles(count)

    return profiles


@router.get("/resources", response_model=List[ResourceUtilization])
async def get_resource_utilization(
    count: int = Query(10, description="Number of data points to return", ge=1, le=100)
):
    """
    Get recent resource utilization data.

    Args:
        count: Number of data points to return

    Returns:
        List of recent resource utilization data
    """
    resources = profiler.get_recent_resource_data(count)

    return resources


@router.post("/resources/start-monitoring", response_model=Dict[str, Any])
async def start_resource_monitoring(
    interval_seconds: int = Query(10, description="Interval between resource checks in seconds", ge=1, le=3600)
):
    """
    Start monitoring system resources.

    Args:
        interval_seconds: Interval between resource checks in seconds

    Returns:
        Status message
    """
    await profiler.start_resource_monitoring(interval_seconds)

    return {
        "status": "success",
        "message": f"Resource monitoring started with {interval_seconds}s interval"
    }


@router.post("/resources/stop-monitoring", response_model=Dict[str, Any])
async def stop_resource_monitoring():
    """
    Stop monitoring system resources.

    Returns:
        Status message
    """
    await profiler.stop_resource_monitoring()

    return {
        "status": "success",
        "message": "Resource monitoring stopped"
    }


@router.post("/memory/start-profiling", response_model=Dict[str, Any])
async def start_memory_profiling():
    """
    Start memory profiling.

    Returns:
        Status message
    """
    profiler.start_memory_profiling()

    return {
        "status": "success",
        "message": "Memory profiling started"
    }


@router.post("/memory/stop-profiling", response_model=Dict[str, Any])
async def stop_memory_profiling():
    """
    Stop memory profiling.

    Returns:
        Status message
    """
    profiler.stop_memory_profiling()

    return {
        "status": "success",
        "message": "Memory profiling stopped"
    }


@router.post("/memory/snapshot", response_model=Dict[str, Any])
async def take_memory_snapshot():
    """
    Take a snapshot of memory usage.

    Returns:
        Memory usage statistics
    """
    snapshot = profiler.take_memory_snapshot()

    return snapshot


@router.get("/rate-limits", response_model=List[RateLimitRule])
async def get_rate_limit_rules(
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Get all rate limit rules.

    Returns:
        List of rate limit rules
    """
    rules = registry.get_all_rules()

    return rules


@router.post("/rate-limits", response_model=Dict[str, str])
async def create_rate_limit_rule(
    rule: RateLimitRule,
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Create a new rate limit rule.

    Args:
        rule: The rate limit rule to create

    Returns:
        Dictionary with rule ID
    """
    rule_id = registry.add_rule(rule)

    return {"id": rule_id, "message": "Rate limit rule created successfully"}


@router.get("/rate-limits/{rule_id}", response_model=RateLimitRule)
async def get_rate_limit_rule(
    rule_id: str = Path(..., description="The rule ID"),
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Get a rate limit rule.

    Args:
        rule_id: The rule ID

    Returns:
        The rate limit rule
    """
    rule = registry.get_rule(rule_id)

    if not rule:
        raise HTTPException(
            status_code=404,
            detail=f"Rate limit rule {rule_id} not found"
        )

    return rule


@router.patch("/rate-limits/{rule_id}", response_model=Dict[str, Any])
async def update_rate_limit_rule(
    updates: Dict[str, Any],
    rule_id: str = Path(..., description="The rule ID"),
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Update a rate limit rule.

    Args:
        rule_id: The rule ID
        updates: Dictionary of updates to apply

    Returns:
        Success message
    """
    success = registry.update_rule(rule_id, updates)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Rate limit rule {rule_id} not found"
        )

    return {"message": f"Rate limit rule {rule_id} updated successfully"}


@router.delete("/rate-limits/{rule_id}", response_model=Dict[str, Any])
async def delete_rate_limit_rule(
    rule_id: str = Path(..., description="The rule ID"),
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Delete a rate limit rule.

    Args:
        rule_id: The rule ID

    Returns:
        Success message
    """
    success = registry.delete_rule(rule_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Rate limit rule {rule_id} not found"
        )

    return {"message": f"Rate limit rule {rule_id} deleted successfully"}


@router.get("/rate-limits/{rule_id}/state", response_model=Optional[RateLimitState])
async def get_rate_limit_state(
    rule_id: str = Path(..., description="The rule ID"),
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Get the current state of a rate limiter.

    Args:
        rule_id: The rule ID

    Returns:
        Current state of the rate limiter
    """
    states = registry.get_all_states()

    if rule_id not in states:
        raise HTTPException(
            status_code=404,
            detail=f"Rate limiter state for rule {rule_id} not found"
        )

    return states[rule_id]


@router.get("/rate-limits/states", response_model=Dict[str, RateLimitState])
async def get_all_rate_limit_states(
    registry: RateLimiterRegistry = Depends(get_rate_limiter_registry)
):
    """
    Get current state for all rate limiters.

    Returns:
        Dictionary mapping rule IDs to states
    """
    return registry.get_all_states()


@router.post("/track-operation", response_model=Dict[str, Any])
async def track_operation(
    data: Dict[str, Any]
):
    """
    Track performance metrics for an operation.

    Args:
        data: Dictionary with operation details:
              - component: Component name
              - operation: Operation name
              - duration_ms: Duration in milliseconds
              - error: Whether an error occurred (optional)
              - metadata: Additional metadata (optional)

    Returns:
        Success message
    """
    if "component" not in data or "operation" not in data or "duration_ms" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: component, operation, and duration_ms are required"
        )

    profiler.track_operation(
        component=data["component"],
        operation=data["operation"],
        duration_ms=data["duration_ms"],
        error=data.get("error", False),
        metadata=data.get("metadata")
    )

    return {"message": "Operation tracked successfully"}