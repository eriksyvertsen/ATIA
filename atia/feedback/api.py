"""
API endpoints for the Feedback System.

This module provides FastAPI endpoints for submitting and retrieving feedback.
"""

import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.security import OAuth2PasswordBearer

from atia.feedback.models import (
    ToolFeedback,
    APIFeedback,
    FeedbackAnalysis,
    FeedbackSummary,
    FeedbackRating,
    FeedbackCategory
)
from atia.feedback.manager import FeedbackManager
from atia.feedback.analyzer import FeedbackAnalyzer


# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    responses={404: {"description": "Not found"}}
)

# Authentication scheme (to be integrated with API Gateway)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Dependency for feedback manager
def get_feedback_manager():
    """Get a feedback manager instance."""
    return FeedbackManager()


# Dependency for feedback analyzer
def get_feedback_analyzer():
    """Get a feedback analyzer instance."""
    return FeedbackAnalyzer()


@router.post("/tool", response_model=Dict[str, str])
async def submit_tool_feedback(
    feedback: ToolFeedback,
    manager: FeedbackManager = Depends(get_feedback_manager),
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer)
):
    """
    Submit feedback for a tool.

    Args:
        feedback: The tool feedback to submit

    Returns:
        Dictionary with feedback ID
    """
    # If no categories provided, try to categorize based on comment
    if feedback.comment and not feedback.categories:
        categories = await analyzer.categorize_feedback(feedback)
        feedback.categories = categories

    # Submit the feedback
    feedback_id = await manager.submit_tool_feedback(feedback)

    return {"id": feedback_id, "message": "Feedback submitted successfully"}


@router.post("/api", response_model=Dict[str, str])
async def submit_api_feedback(
    feedback: APIFeedback,
    manager: FeedbackManager = Depends(get_feedback_manager),
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer)
):
    """
    Submit feedback for an API.

    Args:
        feedback: The API feedback to submit

    Returns:
        Dictionary with feedback ID
    """
    # If no categories provided, try to categorize based on comment
    if feedback.comment and not feedback.categories:
        categories = await analyzer.categorize_feedback(feedback)
        feedback.categories = categories

    # Submit the feedback
    feedback_id = await manager.submit_api_feedback(feedback)

    return {"id": feedback_id, "message": "Feedback submitted successfully"}


@router.get("/tool/{tool_id}/rating", response_model=Dict[str, Any])
async def get_tool_rating(
    tool_id: str = Path(..., description="The tool ID"),
    manager: FeedbackManager = Depends(get_feedback_manager)
):
    """
    Get the average rating for a tool.

    Args:
        tool_id: The tool ID

    Returns:
        Dictionary with average rating and count
    """
    rating, count = await manager.get_tool_rating(tool_id)

    return {
        "tool_id": tool_id,
        "average_rating": rating,
        "count": count
    }


@router.get("/tool/{tool_id}/analysis", response_model=FeedbackAnalysis)
async def get_tool_analysis(
    tool_id: str = Path(..., description="The tool ID"),
    force_refresh: bool = Query(False, description="Force refresh of analysis"),
    manager: FeedbackManager = Depends(get_feedback_manager)
):
    """
    Get analysis of feedback for a tool.

    Args:
        tool_id: The tool ID
        force_refresh: Whether to force a refresh of the analysis

    Returns:
        Feedback analysis
    """
    analysis = await manager.analyze_tool_feedback(tool_id)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough feedback for tool {tool_id} to perform analysis"
        )

    return analysis


@router.get("/summary", response_model=FeedbackSummary)
async def get_feedback_summary(
    manager: FeedbackManager = Depends(get_feedback_manager)
):
    """
    Get a summary of all feedback.

    Returns:
        Feedback summary
    """
    summary = await manager.get_overall_summary()

    return summary


@router.get("/tool/{tool_id}/trends", response_model=Dict[str, Any])
async def get_tool_trends(
    tool_id: str = Path(..., description="The tool ID"),
    days: int = Query(30, description="Number of days to analyze"),
    manager: FeedbackManager = Depends(get_feedback_manager),
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer)
):
    """
    Get trend analysis for a tool's feedback.

    Args:
        tool_id: The tool ID
        days: Number of days to analyze

    Returns:
        Trend analysis
    """
    # Get all feedback for this tool
    all_feedback = await manager._get_all_tool_feedback(tool_id)

    if not all_feedback:
        raise HTTPException(
            status_code=404,
            detail=f"No feedback found for tool {tool_id}"
        )

    # Analyze trends
    trends = await analyzer.analyze_feedback_trends(all_feedback, days)

    return {
        "tool_id": tool_id,
        "days_analyzed": days,
        **trends
    }


@router.post("/categorize", response_model=List[str])
async def categorize_comment(
    data: Dict[str, str],
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer)
):
    """
    Categorize a feedback comment.

    Args:
        data: Dictionary with "comment" key

    Returns:
        List of category strings
    """
    if "comment" not in data or not data["comment"]:
        raise HTTPException(
            status_code=400,
            detail="Comment is required"
        )

    # Create a simple object with a comment attribute
    class Comment:
        def __init__(self, comment):
            self.comment = comment

    comment_obj = Comment(data["comment"])

    # Categorize the comment
    categories = await analyzer.categorize_feedback(comment_obj)

    # Convert to strings
    category_strings = [str(c).replace("FeedbackCategory.", "") for c in categories]

    return category_strings


@router.post("/sentiment", response_model=Dict[str, Any])
async def analyze_sentiment(
    data: Dict[str, str],
    analyzer: FeedbackAnalyzer = Depends(get_feedback_analyzer)
):
    """
    Analyze sentiment of a feedback comment.

    Args:
        data: Dictionary with "comment" key

    Returns:
        Sentiment analysis
    """
    if "comment" not in data or not data["comment"]:
        raise HTTPException(
            status_code=400,
            detail="Comment is required"
        )

    # Analyze sentiment
    sentiment = await analyzer.extract_sentiment(data["comment"])

    return sentiment