"""
Feedback Manager for collecting and analyzing tool and API feedback.

This module provides functionality for collecting, storing, and analyzing feedback
on tools and APIs to improve the system's tool suggestions.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from atia.config import settings
from atia.feedback.models import (
    ToolFeedback,
    APIFeedback,
    FeedbackAnalysis,
    FeedbackSummary,
    FeedbackRating,
    FeedbackCategory
)
from atia.utils.openai_client import get_completion, get_json_completion, get_completion_with_responses_api
from atia.utils.error_handling import catch_and_log


logger = logging.getLogger(__name__)


class FeedbackManager:
    """
    Handles collection, storage, and analysis of tool and API feedback.
    """

    def __init__(self, storage_dir: str = "data/feedback"):
        """
        Initialize the feedback manager.

        Args:
            storage_dir: Directory to store feedback data
        """
        self.storage_dir = storage_dir
        self.tool_feedback_dir = os.path.join(storage_dir, "tool")
        self.api_feedback_dir = os.path.join(storage_dir, "api")
        self.analysis_dir = os.path.join(storage_dir, "analysis")

        # Create storage directories
        os.makedirs(self.tool_feedback_dir, exist_ok=True)
        os.makedirs(self.api_feedback_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # Cache for recent feedback and analysis
        self._recent_tool_feedback: List[ToolFeedback] = []
        self._recent_api_feedback: List[APIFeedback] = []
        self._analysis_cache: Dict[str, FeedbackAnalysis] = {}

        # Load recent feedback
        self._load_recent_feedback()

    @catch_and_log(component="feedback_manager")
    async def submit_tool_feedback(self, feedback: ToolFeedback) -> str:
        """
        Submit feedback for a tool.

        Args:
            feedback: The tool feedback to submit

        Returns:
            Feedback ID
        """
        # Store in memory cache
        self._recent_tool_feedback.append(feedback)
        if len(self._recent_tool_feedback) > 100:  # Keep only the 100 most recent
            self._recent_tool_feedback.pop(0)

        # Store to file
        feedback_file = os.path.join(self.tool_feedback_dir, f"{feedback.id}.json")
        with open(feedback_file, "w") as f:
            # Convert to dict manually to handle datetime serialization
            feedback_dict = feedback.model_dump()
            feedback_dict["created_at"] = feedback.created_at.isoformat()
            json.dump(feedback_dict, f, indent=2)

        # Update tool rating in registry if possible
        await self._update_tool_rating(feedback.tool_id)

        logger.info(f"Stored tool feedback {feedback.id} for tool {feedback.tool_id}")
        return feedback.id

    @catch_and_log(component="feedback_manager")
    async def submit_api_feedback(self, feedback: APIFeedback) -> str:
        """
        Submit feedback for an API.

        Args:
            feedback: The API feedback to submit

        Returns:
            Feedback ID
        """
        # Store in memory cache
        self._recent_api_feedback.append(feedback)
        if len(self._recent_api_feedback) > 100:  # Keep only the 100 most recent
            self._recent_api_feedback.pop(0)

        # Store to file
        feedback_file = os.path.join(self.api_feedback_dir, f"{feedback.id}.json")
        with open(feedback_file, "w") as f:
            # Convert to dict manually to handle datetime serialization
            feedback_dict = feedback.model_dump()
            feedback_dict["created_at"] = feedback.created_at.isoformat()
            json.dump(feedback_dict, f, indent=2)

        logger.info(f"Stored API feedback {feedback.id} for API {feedback.api_source_id}")
        return feedback.id

    @catch_and_log(component="feedback_manager")
    async def analyze_tool_feedback(self, tool_id: str) -> Optional[FeedbackAnalysis]:
        """
        Analyze feedback for a specific tool.

        Args:
            tool_id: ID of the tool to analyze

        Returns:
            Analysis of the tool feedback, or None if not enough feedback
        """
        # Check if we have a recent cached analysis
        cache_key = f"tool:{tool_id}"
        if cache_key in self._analysis_cache:
            analysis = self._analysis_cache[cache_key]
            # If analysis is less than a day old, return it
            if datetime.now() - analysis.analyzed_at < timedelta(days=1):
                return analysis

        # Collect all feedback for this tool
        all_feedback = await self._get_all_tool_feedback(tool_id)

        if len(all_feedback) < settings.feedback_min_samples:
            logger.info(f"Not enough feedback for tool {tool_id} to analyze")
            return None

        # Perform analysis
        analysis = await self._analyze_feedback("tool", tool_id, all_feedback)

        # Cache the analysis
        self._analysis_cache[cache_key] = analysis

        # Save analysis to file
        analysis_file = os.path.join(self.analysis_dir, f"tool_{tool_id}.json")
        with open(analysis_file, "w") as f:
            analysis_dict = analysis.model_dump()
            analysis_dict["analyzed_at"] = analysis.analyzed_at.isoformat()
            json.dump(analysis_dict, f, indent=2)

        return analysis

    @catch_and_log(component="feedback_manager")
    async def analyze_api_feedback(self, api_source_id: str) -> Optional[FeedbackAnalysis]:
        """
        Analyze feedback for a specific API.

        Args:
            api_source_id: ID of the API to analyze

        Returns:
            Analysis of the API feedback, or None if not enough feedback
        """
        # Check if we have a recent cached analysis
        cache_key = f"api:{api_source_id}"
        if cache_key in self._analysis_cache:
            analysis = self._analysis_cache[cache_key]
            # If analysis is less than a day old, return it
            if datetime.now() - analysis.analyzed_at < timedelta(days=1):
                return analysis

        # Collect all feedback for this API
        all_feedback = await self._get_all_api_feedback(api_source_id)

        if len(all_feedback) < settings.feedback_min_samples:
            logger.info(f"Not enough feedback for API {api_source_id} to analyze")
            return None

        # Perform analysis
        analysis = await self._analyze_feedback("api", api_source_id, all_feedback)

        # Cache the analysis
        self._analysis_cache[cache_key] = analysis

        # Save analysis to file
        analysis_file = os.path.join(self.analysis_dir, f"api_{api_source_id}.json")
        with open(analysis_file, "w") as f:
            analysis_dict = analysis.model_dump()
            analysis_dict["analyzed_at"] = analysis.analyzed_at.isoformat()
            json.dump(analysis_dict, f, indent=2)

        return analysis

    @catch_and_log(component="feedback_manager")
    async def get_tool_rating(self, tool_id: str) -> Tuple[float, int]:
        """
        Get the average rating for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            Tuple of (average_rating, number_of_ratings)
        """
        all_feedback = await self._get_all_tool_feedback(tool_id)

        if not all_feedback:
            return 0.0, 0

        # Convert ratings to numeric values
        rating_values = {
            FeedbackRating.EXCELLENT: 5.0,
            FeedbackRating.GOOD: 4.0,
            FeedbackRating.NEUTRAL: 3.0,
            FeedbackRating.POOR: 2.0,
            FeedbackRating.VERY_POOR: 1.0
        }

        total = sum(rating_values[f.rating] for f in all_feedback)
        count = len(all_feedback)

        return total / count, count

    @catch_and_log(component="feedback_manager")
    async def get_overall_summary(self) -> FeedbackSummary:
        """
        Get a summary of all feedback across tools and APIs.

        Returns:
            Summary of all feedback
        """
        # Get all tool feedback
        all_tool_feedback = await self._get_all_tool_feedback()

        # Get all API feedback
        all_api_feedback = await self._get_all_api_feedback()

        # Calculate summary statistics
        summary = FeedbackSummary(
            total_feedback=len(all_tool_feedback) + len(all_api_feedback),
            generated_at=datetime.now()
        )

        # Calculate average tool rating
        if all_tool_feedback:
            rating_values = {
                FeedbackRating.EXCELLENT: 5.0,
                FeedbackRating.GOOD: 4.0,
                FeedbackRating.NEUTRAL: 3.0,
                FeedbackRating.POOR: 2.0,
                FeedbackRating.VERY_POOR: 1.0
            }

            tool_total = sum(rating_values[f.rating] for f in all_tool_feedback)
            summary.average_tool_rating = tool_total / len(all_tool_feedback)

        # Calculate average API rating
        if all_api_feedback:
            rating_values = {
                FeedbackRating.EXCELLENT: 5.0,
                FeedbackRating.GOOD: 4.0,
                FeedbackRating.NEUTRAL: 3.0,
                FeedbackRating.POOR: 2.0,
                FeedbackRating.VERY_POOR: 1.0
            }

            api_total = sum(rating_values[f.rating] for f in all_api_feedback)
            summary.average_api_rating = api_total / len(all_api_feedback)

        # Calculate top and lowest rated tools
        tool_ratings = {}
        for feedback in all_tool_feedback:
            if feedback.tool_id not in tool_ratings:
                tool_ratings[feedback.tool_id] = {"total": 0, "count": 0}

            rating_values = {
                FeedbackRating.EXCELLENT: 5.0,
                FeedbackRating.GOOD: 4.0,
                FeedbackRating.NEUTRAL: 3.0,
                FeedbackRating.POOR: 2.0,
                FeedbackRating.VERY_POOR: 1.0
            }

            tool_ratings[feedback.tool_id]["total"] += rating_values[feedback.rating]
            tool_ratings[feedback.tool_id]["count"] += 1

        # Calculate average ratings
        for tool_id, data in tool_ratings.items():
            if data["count"] > 0:
                data["average"] = data["total"] / data["count"]

        # Sort tools by average rating
        sorted_tools = sorted(
            [{"id": k, "average": v["average"], "count": v["count"]} 
             for k, v in tool_ratings.items() if v["count"] >= 3],  # At least 3 ratings
            key=lambda x: x["average"],
            reverse=True
        )

        # Get top and lowest rated tools
        summary.top_rated_tools = sorted_tools[:5]  # Top 5
        summary.lowest_rated_tools = sorted_tools[-5:]  # Bottom 5

        # Gather common issues across all feedback
        issues = {}

        # Look for issues in tool feedback comments
        for feedback in all_tool_feedback:
            if feedback.rating in [FeedbackRating.POOR, FeedbackRating.VERY_POOR] and feedback.comment:
                for category in feedback.categories:
                    if category not in issues:
                        issues[category] = {"count": 0, "examples": []}

                    issues[category]["count"] += 1
                    if len(issues[category]["examples"]) < 3:  # Keep up to 3 examples
                        issues[category]["examples"].append(feedback.comment[:100])  # Truncate long comments

        # Look for issues in API feedback comments
        for feedback in all_api_feedback:
            if feedback.rating in [FeedbackRating.POOR, FeedbackRating.VERY_POOR] and feedback.comment:
                for category in feedback.categories:
                    if category not in issues:
                        issues[category] = {"count": 0, "examples": []}

                    issues[category]["count"] += 1
                    if len(issues[category]["examples"]) < 3:  # Keep up to 3 examples
                        issues[category]["examples"].append(feedback.comment[:100])  # Truncate long comments

        # Convert to list and sort by count
        summary.common_issues = [
            {"category": k, "count": v["count"], "examples": v["examples"]}
            for k, v in issues.items()
        ]
        summary.common_issues.sort(key=lambda x: x["count"], reverse=True)

        return summary

    async def _get_all_tool_feedback(self, tool_id: Optional[str] = None) -> List[ToolFeedback]:
        """Get all feedback for a tool or all tools."""
        # Start with cached feedback
        feedback_list = [f for f in self._recent_tool_feedback if tool_id is None or f.tool_id == tool_id]

        # Read from files
        try:
            for filename in os.listdir(self.tool_feedback_dir):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(self.tool_feedback_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                    # Skip if we're looking for a specific tool and this isn't it
                    if tool_id is not None and data.get("tool_id") != tool_id:
                        continue

                    # Convert data to ToolFeedback
                    if "created_at" in data and isinstance(data["created_at"], str):
                        data["created_at"] = datetime.fromisoformat(data["created_at"])

                    # Create ToolFeedback object
                    feedback = ToolFeedback(**data)

                    # Add to list if not already in cache
                    if feedback.id not in [f.id for f in feedback_list]:
                        feedback_list.append(feedback)
        except Exception as e:
            logger.error(f"Error reading tool feedback: {e}")

        return feedback_list

    async def _get_all_api_feedback(self, api_source_id: Optional[str] = None) -> List[APIFeedback]:
        """Get all feedback for an API or all APIs."""
        # Start with cached feedback
        feedback_list = [f for f in self._recent_api_feedback if api_source_id is None or f.api_source_id == api_source_id]

        # Read from files
        try:
            for filename in os.listdir(self.api_feedback_dir):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(self.api_feedback_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                    # Skip if we're looking for a specific API and this isn't it
                    if api_source_id is not None and data.get("api_source_id") != api_source_id:
                        continue

                    # Convert data to APIFeedback
                    if "created_at" in data and isinstance(data["created_at"], str):
                        data["created_at"] = datetime.fromisoformat(data["created_at"])

                    # Create APIFeedback object
                    feedback = APIFeedback(**data)

                    # Add to list if not already in cache
                    if feedback.id not in [f.id for f in feedback_list]:
                        feedback_list.append(feedback)
        except Exception as e:
            logger.error(f"Error reading API feedback: {e}")

        return feedback_list

    async def _analyze_feedback(self, target_type: str, target_id: str, 
                               feedback_list: Union[List[ToolFeedback], List[APIFeedback]]) -> FeedbackAnalysis:
        """Analyze feedback using LLM for insights."""
        if not feedback_list:
            raise ValueError("No feedback provided for analysis")

        # Create the analysis object
        analysis = FeedbackAnalysis(
            target_id=target_id,
            target_type=target_type,
            total_feedback=len(feedback_list),
            analyzed_at=datetime.now()
        )

        # Calculate average rating
        rating_values = {
            FeedbackRating.EXCELLENT: 5.0,
            FeedbackRating.GOOD: 4.0,
            FeedbackRating.NEUTRAL: 3.0,
            FeedbackRating.POOR: 2.0,
            FeedbackRating.VERY_POOR: 1.0
        }

        rating_sum = sum(rating_values[f.rating] for f in feedback_list)
        analysis.average_rating = rating_sum / len(feedback_list)

        # Calculate rating distribution
        for rating in FeedbackRating:
            count = sum(1 for f in feedback_list if f.rating == rating)
            analysis.rating_distribution[rating] = count

        # Calculate top categories
        category_counts = {}
        for feedback in feedback_list:
            for category in feedback.categories:
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

        analysis.top_categories = [
            {"category": k, "count": v}
            for k, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # Use LLM to analyze comments and generate insights
        # Only if we have at least some negative feedback with comments
        negative_feedback = [
            f for f in feedback_list 
            if f.rating in [FeedbackRating.POOR, FeedbackRating.VERY_POOR] 
            and f.comment
        ]

        if negative_feedback:
            analysis.common_issues, analysis.improvement_suggestions = \
                await self._get_llm_insights(target_type, target_id, negative_feedback)

        return analysis

    async def _get_llm_insights(self, target_type: str, target_id: str, 
                              negative_feedback: List[Union[ToolFeedback, APIFeedback]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Use LLM to analyze negative feedback and generate insights."""
        # Prepare negative feedback comments
        comments = []
        for i, feedback in enumerate(negative_feedback[:20], 1):  # Limit to 20 comments
            rating = str(feedback.rating).replace('FeedbackRating.', '')
            categories = [str(c).replace('FeedbackCategory.', '') for c in feedback.categories]
            comments.append(f"{i}. Rating: {rating}, Categories: {categories}, Comment: {feedback.comment}")

        comments_text = "\n".join(comments)

        prompt = f"""
        Analyze the following negative feedback for a {target_type} and provide:
        1. Common issues mentioned (3-5 issues)
        2. Concrete suggestions for improvement (3-5 actionable suggestions)

        Feedback:
        {comments_text}

        Respond with a JSON object containing:
        {{
            "common_issues": [
                {{"issue": "Description of the issue", "count": number_of_mentions}}
            ],
            "improvement_suggestions": [
                "Actionable suggestion 1",
                "Actionable suggestion 2"
            ]
        }}
        """

        system_message = (
            "You are an expert at analyzing user feedback and identifying actionable insights. "
            "Extract common issues and provide specific, concrete suggestions for improvement."
        )

        # Try to use Responses API if available
        try:
            if not settings.disable_responses_api:
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    model=settings.openai_model
                )

                # Try to extract JSON from the response
                content = response.get("content", "")
                result = json.loads(content)
            else:
                # Use standard JSON completion
                result = get_json_completion(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    max_tokens=1000,
                    model=settings.openai_model
                )

            # Extract common issues and suggestions
            common_issues = result.get("common_issues", [])
            improvement_suggestions = result.get("improvement_suggestions", [])

            return common_issues, improvement_suggestions
        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            return [], []

    async def _update_tool_rating(self, tool_id: str) -> None:
        """Update tool rating in the Tool Registry if available."""
        try:
            from atia.tool_registry import ToolRegistry

            # Try to import dynamically to avoid circular imports
            try:
                tool_registry = ToolRegistry()

                # Get average rating and count
                avg_rating, count = await self.get_tool_rating(tool_id)

                if count > 0:
                    # Update the tool's metadata with rating info
                    tool = await tool_registry.get_tool(tool_id)
                    if tool:
                        # Update tool metadata to include rating
                        if not tool.metadata:
                            tool.metadata = {}

                        tool.metadata["user_rating"] = avg_rating
                        tool.metadata["rating_count"] = count

                        # Re-register the tool with updated metadata
                        from atia.function_builder import FunctionBuilder

                        function_builder = FunctionBuilder()
                        function_def = await tool_registry._get_function_definition(tool.function_id)

                        if function_def:
                            await tool_registry.register_function(function_def, tool.metadata)
                            logger.info(f"Updated tool {tool_id} with rating {avg_rating} from {count} ratings")
            except Exception as inner_e:
                logger.warning(f"Could not update tool rating in registry: {inner_e}")
        except ImportError:
            # Registry not available
            logger.warning("Tool Registry not available, skipping rating update")

    def _load_recent_feedback(self) -> None:
        """Load recent feedback into memory cache."""
        # Load recent tool feedback
        try:
            tool_files = sorted(
                [f for f in os.listdir(self.tool_feedback_dir) if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(self.tool_feedback_dir, x)),
                reverse=True
            )

            # Load the 100 most recent
            for filename in tool_files[:100]:
                try:
                    with open(os.path.join(self.tool_feedback_dir, filename), 'r') as f:
                        data = json.load(f)

                        # Convert timestamps
                        if "created_at" in data and isinstance(data["created_at"], str):
                            data["created_at"] = datetime.fromisoformat(data["created_at"])

                        # Create ToolFeedback object
                        feedback = ToolFeedback(**data)
                        self._recent_tool_feedback.append(feedback)
                except Exception as e:
                    logger.error(f"Error loading tool feedback file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading recent tool feedback: {e}")

        # Load recent API feedback
        try:
            api_files = sorted(
                [f for f in os.listdir(self.api_feedback_dir) if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(self.api_feedback_dir, x)),
                reverse=True
            )

            # Load the 100 most recent
            for filename in api_files[:100]:
                try:
                    with open(os.path.join(self.api_feedback_dir, filename), 'r') as f:
                        data = json.load(f)

                        # Convert timestamps
                        if "created_at" in data and isinstance(data["created_at"], str):
                            data["created_at"] = datetime.fromisoformat(data["created_at"])

                        # Create APIFeedback object
                        feedback = APIFeedback(**data)
                        self._recent_api_feedback.append(feedback)
                except Exception as e:
                    logger.error(f"Error loading API feedback file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading recent API feedback: {e}")

        logger.info(f"Loaded {len(self._recent_tool_feedback)} recent tool feedback and {len(self._recent_api_feedback)} recent API feedback items")