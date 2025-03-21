"""
Data models for the Feedback System component.

This module provides models for collecting, storing, and analyzing feedback
on tools and APIs.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid

from pydantic import BaseModel, Field


class FeedbackRating(str, Enum):
    """Rating options for tool feedback."""
    EXCELLENT = "excellent"
    GOOD = "good"
    NEUTRAL = "neutral"
    POOR = "poor"
    VERY_POOR = "very_poor"


class FeedbackCategory(str, Enum):
    """Categories for tool feedback."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    RELIABILITY = "reliability"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    OTHER = "other"


class ToolFeedback(BaseModel):
    """
    Model for tool usage feedback.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    rating: FeedbackRating
    categories: List[FeedbackCategory] = []
    comment: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error_occurred: bool = False
    error_detail: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class APIFeedback(BaseModel):
    """
    Model for API usage feedback.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    api_source_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    rating: FeedbackRating
    categories: List[FeedbackCategory] = []
    comment: Optional[str] = None
    response_time_ms: Optional[float] = None
    http_status: Optional[int] = None
    error_occurred: bool = False
    error_detail: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackAnalysis(BaseModel):
    """
    Analysis of feedback for a specific tool or API.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str  # Either tool_id or api_source_id
    target_type: str  # "tool" or "api"
    total_feedback: int = 0
    average_rating: float = 0.0
    rating_distribution: Dict[str, int] = Field(default_factory=dict)
    common_issues: List[Dict[str, Any]] = []
    improvement_suggestions: List[str] = []
    top_categories: List[Dict[str, Any]] = []
    analyzed_at: datetime = Field(default_factory=datetime.now)


class FeedbackSummary(BaseModel):
    """
    High-level summary of all feedback.
    """
    total_feedback: int = 0
    average_tool_rating: float = 0.0
    average_api_rating: float = 0.0
    top_rated_tools: List[Dict[str, Any]] = []
    lowest_rated_tools: List[Dict[str, Any]] = []
    common_issues: List[Dict[str, Any]] = []
    generated_at: datetime = Field(default_factory=datetime.now)