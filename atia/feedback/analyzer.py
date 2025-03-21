"""
Feedback Analyzer for extracting insights from feedback data.

This module provides advanced analytics and NLP-based analysis of user feedback
to generate actionable insights for improving tools and APIs.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import Counter

from atia.config import settings
from atia.feedback.models import (
    ToolFeedback,
    APIFeedback,
    FeedbackAnalysis,
    FeedbackRating,
    FeedbackCategory
)
from atia.utils.openai_client import get_completion, get_json_completion, get_completion_with_responses_api


logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """
    Analyzes feedback data to extract insights and improvement suggestions.
    """

    def __init__(self):
        """Initialize the feedback analyzer."""
        pass

    async def extract_issues(self, 
                           feedback_list: List[Any], 
                           min_count: int = 2) -> List[Dict[str, Any]]:
        """
        Extract common issues from feedback comments.

        Args:
            feedback_list: List of feedback items
            min_count: Minimum number of mentions to consider an issue common

        Returns:
            List of common issues with counts
        """
        # Filter for negative feedback with comments
        negative_feedback = [
            f for f in feedback_list 
            if (hasattr(f, 'rating') and f.rating in [FeedbackRating.POOR, FeedbackRating.VERY_POOR]) 
            and hasattr(f, 'comment') and f.comment
        ]

        if not negative_feedback:
            return []

        # First, try with an LLM approach if we have enough comments
        if len(negative_feedback) >= 3:
            issues = await self._extract_issues_with_llm(negative_feedback)
            if issues:
                return issues

        # Fallback to rule-based approach
        return self._extract_issues_rule_based(negative_feedback, min_count)

    async def generate_suggestions(self, 
                                 issues: List[Dict[str, Any]], 
                                 target_type: str) -> List[str]:
        """
        Generate improvement suggestions based on identified issues.

        Args:
            issues: List of common issues
            target_type: Type of target ("tool" or "api")

        Returns:
            List of improvement suggestions
        """
        if not issues:
            return []

        # Format issues for the prompt
        issues_text = "\n".join([f"{i+1}. {issue['issue']} (mentioned {issue['count']} times)" 
                               for i, issue in enumerate(issues)])

        prompt = f"""
        Based on these common issues with a {target_type}:

        {issues_text}

        Generate 3-5 specific, actionable suggestions to address these issues. 
        Make the suggestions concrete and implementation-ready.

        Format your response as a JSON array of suggestion strings.
        """

        system_message = (
            "You are an expert at analyzing user feedback and providing actionable improvement suggestions. "
            "Be specific, concrete, and focus on technical improvements that can be implemented."
        )

        try:
            # Try to use Responses API if available
            if not settings.disable_responses_api:
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    model=settings.openai_model
                )

                # Try to extract JSON from the response
                content = response.get("content", "")

                # Handle potential leading/trailing text
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]

                content = content.strip()

                try:
                    suggestions = json.loads(content)
                    if isinstance(suggestions, list):
                        return suggestions
                except:
                    # If we can't parse as JSON, try to extract from text
                    suggestions = []
                    lines = content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and line[0].isdigit() and ". " in line:
                            suggestion = line.split(". ", 1)[1]
                            suggestions.append(suggestion)
                    return suggestions
            else:
                # Use standard JSON completion
                result = get_json_completion(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    max_tokens=1000,
                    model=settings.openai_model
                )

                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and "suggestions" in result:
                    return result["suggestions"]
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")

        # Fallback: Generate simple suggestions based on issue categories
        return self._generate_fallback_suggestions(issues, target_type)

    async def categorize_feedback(self, 
                                feedback_item: Any) -> List[FeedbackCategory]:
        """
        Automatically categorize feedback based on comment content.

        Args:
            feedback_item: Feedback item to categorize

        Returns:
            List of feedback categories
        """
        if not hasattr(feedback_item, 'comment') or not feedback_item.comment:
            # No comment to categorize
            return []

        # Check if already categorized
        if hasattr(feedback_item, 'categories') and feedback_item.categories:
            return feedback_item.categories

        comment = feedback_item.comment

        # Try with LLM if available
        if not settings.disable_responses_api:
            categories = await self._categorize_with_llm(comment)
            if categories:
                return categories

        # Fallback to rule-based categorization
        return self._categorize_rule_based(comment)

    async def extract_sentiment(self, comment: str) -> Dict[str, Any]:
        """
        Extract sentiment and key aspects from a feedback comment.

        Args:
            comment: The feedback comment text

        Returns:
            Dictionary with sentiment analysis results
        """
        if not comment:
            return {"sentiment": "neutral", "score": 0.5, "aspects": {}}

        # Try with LLM if available
        if not settings.disable_responses_api:
            try:
                prompt = f"""
                Analyze the sentiment of this feedback comment, and extract key aspects mentioned:

                "{comment}"

                Respond with a JSON object containing:
                1. Overall sentiment (positive, negative, or neutral)
                2. Sentiment score (0.0 to 1.0, where 0 is very negative, 0.5 is neutral, 1.0 is very positive)
                3. Key aspects mentioned and their sentiment (e.g., "performance": "negative")
                """

                system_message = (
                    "You are an expert at sentiment analysis and aspect extraction from feedback comments. "
                    "Extract sentiment and key aspects accurately and objectively."
                )

                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    model="gpt-4o-mini"  # Use smaller model for efficiency
                )

                # Try to extract JSON from the response
                content = response.get("content", "")
                try:
                    result = json.loads(content)
                    return {
                        "sentiment": result.get("sentiment", "neutral"),
                        "score": result.get("sentiment_score", 0.5),
                        "aspects": result.get("aspects", {})
                    }
                except:
                    pass
            except Exception as e:
                logger.error(f"Error extracting sentiment with LLM: {e}")

        # Fallback to rule-based sentiment analysis
        return self._simple_sentiment_analysis(comment)

    async def analyze_feedback_trends(self, 
                                    feedback_list: List[Any], 
                                    time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze trends in feedback over time.

        Args:
            feedback_list: List of feedback items
            time_window_days: Time window for trend analysis in days

        Returns:
            Dictionary with trend analysis results
        """
        import datetime

        if not feedback_list:
            return {
                "trend": "stable",
                "data_points": [],
                "improvement_areas": []
            }

        # Sort feedback by creation time
        sorted_feedback = sorted(feedback_list, key=lambda x: x.created_at)

        # Group feedback by day
        from collections import defaultdict
        import datetime

        daily_ratings = defaultdict(list)
        for feedback in sorted_feedback:
            date_key = feedback.created_at.date()

            # Convert rating to numeric value
            rating_values = {
                FeedbackRating.EXCELLENT: 5.0,
                FeedbackRating.GOOD: 4.0,
                FeedbackRating.NEUTRAL: 3.0,
                FeedbackRating.POOR: 2.0,
                FeedbackRating.VERY_POOR: 1.0
            }

            daily_ratings[date_key].append(rating_values[feedback.rating])

        # Calculate daily averages
        data_points = []
        for date, ratings in sorted(daily_ratings.items()):
            avg_rating = sum(ratings) / len(ratings)
            data_points.append({
                "date": date.isoformat(),
                "average_rating": avg_rating,
                "count": len(ratings)
            })

        # Analyze trend
        if len(data_points) < 2:
            trend = "insufficient_data"
        else:
            # Simple linear regression
            x = list(range(len(data_points)))
            y = [point["average_rating"] for point in data_points]

            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_xx = sum(xi * xi for xi in x)

            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

            if slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"

        # Identify improvement areas based on recent negative feedback
        recent_cutoff = datetime.datetime.now() - datetime.timedelta(days=time_window_days)
        recent_negative = [
            f for f in sorted_feedback 
            if f.created_at >= recent_cutoff 
            and f.rating in [FeedbackRating.POOR, FeedbackRating.VERY_POOR]
        ]

        improvement_areas = []
        if recent_negative:
            # Count categories
            category_counts = Counter()
            for feedback in recent_negative:
                for category in feedback.categories:
                    category_counts[category] += 1

            # Get top categories
            improvement_areas = [
                {"category": str(cat).replace("FeedbackCategory.", ""), "count": count}
                for cat, count in category_counts.most_common(3)
            ]

        return {
            "trend": trend,
            "data_points": data_points,
            "improvement_areas": improvement_areas
        }

    async def _extract_issues_with_llm(self, negative_feedback: List[Any]) -> List[Dict[str, Any]]:
        """Extract issues using LLM."""
        try:
            # Prepare comments for analysis
            comments = []
            for i, feedback in enumerate(negative_feedback[:20], 1):  # Limit to 20 comments
                rating = str(feedback.rating).replace('FeedbackRating.', '')
                categories = [str(c).replace('FeedbackCategory.', '') for c in feedback.categories]
                comments.append(f"{i}. Rating: {rating}, Categories: {categories}, Comment: {feedback.comment}")

            comments_text = "\n".join(comments)

            prompt = f"""
            Analyze these negative feedback comments and identify the common issues mentioned:

            {comments_text}

            Extract 3-5 distinct issues that appear multiple times. For each issue:
            1. Provide a clear description of the issue
            2. Count how many comments mention this issue

            Respond with a JSON array of objects with "issue" and "count" properties.
            """

            system_message = (
                "You are an expert at analyzing user feedback and identifying common issues. "
                "Extract distinct issues that appear in multiple comments."
            )

            if not settings.disable_responses_api:
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    model=settings.openai_model
                )

                # Try to extract JSON from the response
                content = response.get("content", "")
                try:
                    issues = json.loads(content)
                    if isinstance(issues, list):
                        return issues
                    elif isinstance(issues, dict) and "issues" in issues:
                        return issues["issues"]
                except:
                    pass
            else:
                # Use standard JSON completion
                result = get_json_completion(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    max_tokens=1000,
                    model=settings.openai_model
                )

                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and "issues" in result:
                    return result["issues"]
        except Exception as e:
            logger.error(f"Error extracting issues with LLM: {e}")

        return []

    def _extract_issues_rule_based(self, 
                                 negative_feedback: List[Any], 
                                 min_count: int) -> List[Dict[str, Any]]:
        """Extract issues using rule-based approach."""
        # Group feedback by category
        category_feedback = {}
        for feedback in negative_feedback:
            for category in feedback.categories:
                if category not in category_feedback:
                    category_feedback[category] = []
                category_feedback[category].append(feedback)

        # Extract common keywords for each category
        issues = []
        for category, feedback_list in category_feedback.items():
            if len(feedback_list) < min_count:
                continue

            # Get common phrases
            common_phrases = self._find_common_phrases([f.comment for f in feedback_list])

            for phrase, count in common_phrases.items():
                if count >= min_count:
                    issues.append({
                        "issue": f"{str(category).replace('FeedbackCategory.', '')}: {phrase}",
                        "count": count
                    })

        # Sort by count
        issues.sort(key=lambda x: x["count"], reverse=True)

        return issues[:5]  # Return top 5 issues

    def _find_common_phrases(self, texts: List[str]) -> Dict[str, int]:
        """Find common phrases in a list of texts."""
        # Simple implementation - could be improved with NLP techniques
        from collections import Counter

        # Tokenize and clean texts
        cleaned_texts = []
        for text in texts:
            # Convert to lowercase and split into words
            words = text.lower().split()

            # Remove common stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'as'}
            cleaned_words = [w for w in words if w not in stopwords]

            cleaned_texts.append(cleaned_words)

        # Find common bigrams (pairs of words)
        bigrams = []
        for words in cleaned_texts:
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]} {words[i+1]}")

        # Count occurrences
        bigram_counter = Counter(bigrams)
        common_phrases = {phrase: count for phrase, count in bigram_counter.items() if count > 1}

        return common_phrases

    async def _categorize_with_llm(self, comment: str) -> List[FeedbackCategory]:
        """Categorize feedback using LLM."""
        try:
            categories_list = [c.value for c in FeedbackCategory]
            categories_str = ", ".join(categories_list)

            prompt = f"""
            Categorize this feedback comment into one or more of these categories: {categories_str}

            Comment: "{comment}"

            Return only a JSON array of category strings, with no explanation.
            """

            system_message = (
                "You are an expert at categorizing feedback comments. "
                "Assign the most relevant categories from the provided list."
            )

            if not settings.disable_responses_api:
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    model="gpt-4o-mini"  # Use smaller model for efficiency
                )

                # Try to extract categories from the response
                content = response.get("content", "")
                try:
                    categories = json.loads(content)
                    if isinstance(categories, list):
                        # Convert string categories to enum
                        return [FeedbackCategory(c) for c in categories 
                               if c in categories_list]
                except:
                    pass
            else:
                # Use standard JSON completion
                result = get_json_completion(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    max_tokens=100,
                    model="gpt-4o-mini"
                )

                if isinstance(result, list):
                    # Convert string categories to enum
                    return [FeedbackCategory(c) for c in result 
                           if c in categories_list]
        except Exception as e:
            logger.error(f"Error categorizing with LLM: {e}")

        return []

    def _categorize_rule_based(self, comment: str) -> List[FeedbackCategory]:
        """Categorize feedback using rule-based approach."""
        comment_lower = comment.lower()

        categories = set()

        # Simple keyword matching for each category
        if any(word in comment_lower for word in ['accuracy', 'correct', 'wrong', 'incorrect', 'error', 'mistake']):
            categories.add(FeedbackCategory.ACCURACY)

        if any(word in comment_lower for word in ['slow', 'fast', 'performance', 'speed', 'time', 'latency', 'wait']):
            categories.add(FeedbackCategory.PERFORMANCE)

        if any(word in comment_lower for word in ['use', 'usability', 'intuitive', 'complicated', 'confusing', 'interface']):
            categories.add(FeedbackCategory.USABILITY)

        if any(word in comment_lower for word in ['reliable', 'stability', 'crash', 'consistent', 'always', 'never']):
            categories.add(FeedbackCategory.RELIABILITY)

        if any(word in comment_lower for word in ['document', 'documentation', 'example', 'explain', 'unclear', 'instruction']):
            categories.add(FeedbackCategory.DOCUMENTATION)

        if any(word in comment_lower for word in ['secure', 'security', 'privacy', 'data', 'credential', 'password', 'token']):
            categories.add(FeedbackCategory.SECURITY)

        # If no categories matched, use OTHER
        if not categories:
            categories.add(FeedbackCategory.OTHER)

        return list(categories)

    def _simple_sentiment_analysis(self, comment: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis."""
        comment_lower = comment.lower()

        # Simple positive/negative word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 
            'wonderful', 'helpful', 'useful', 'effective', 'love', 'like',
            'accurate', 'fast', 'reliable', 'intuitive', 'easy'
        }

        negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'useless', 
            'unhelpful', 'ineffective', 'hate', 'dislike', 'slow', 'buggy',
            'crashes', 'error', 'difficult', 'complicated', 'confusing'
        }

        # Count positive and negative words
        words = set(comment_lower.split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))

        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            sentiment = "neutral"
            score = 0.5
        else:
            score = positive_count / total
            if score > 0.6:
                sentiment = "positive"
            elif score < 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        # Extract aspects (very simple implementation)
        aspects = {}
        aspect_keywords = {
            "accuracy": ["accuracy", "correct", "incorrect", "error"],
            "performance": ["performance", "speed", "slow", "fast"],
            "usability": ["usability", "use", "interface", "intuitive"],
            "reliability": ["reliability", "stable", "crash", "consistent"]
        }

        for aspect, keywords in aspect_keywords.items():
            matches = [k for k in keywords if k in comment_lower]
            if matches:
                # Determine aspect sentiment based on surrounding words
                aspect_sentiment = "neutral"
                for match in matches:
                    index = comment_lower.find(match)
                    if index >= 0:
                        # Check words before and after the match
                        context = comment_lower[max(0, index-20):min(len(comment_lower), index+20)]
                        pos_count = sum(1 for word in positive_words if word in context)
                        neg_count = sum(1 for word in negative_words if word in context)

                        if pos_count > neg_count:
                            aspect_sentiment = "positive"
                        elif neg_count > pos_count:
                            aspect_sentiment = "negative"

                aspects[aspect] = aspect_sentiment

        return {
            "sentiment": sentiment,
            "score": score,
            "aspects": aspects
        }

    def _generate_fallback_suggestions(self, 
                                     issues: List[Dict[str, Any]], 
                                     target_type: str) -> List[str]:
        """Generate fallback suggestions based on issue categories."""
        suggestions = []

        # Generic suggestions for common categories
        category_suggestions = {
            "accuracy": [
                "Implement more robust validation for input data",
                "Add comprehensive error handling with specific error messages",
                "Increase test coverage for edge cases"
            ],
            "performance": [
                "Optimize critical code paths for better performance",
                "Implement caching for frequently accessed data",
                "Review and optimize database queries"
            ],
            "usability": [
                "Improve input parameter documentation with examples",
                "Add more descriptive error messages",
                "Simplify the interface by consolidating similar parameters"
            ],
            "reliability": [
                "Implement retry mechanisms for transient failures",
                "Add circuit breakers to prevent cascading failures",
                "Improve logging for better diagnostics"
            ],
            "documentation": [
                "Expand documentation with more examples and use cases",
                "Add a troubleshooting section to documentation",
                "Include parameter validation rules in the documentation"
            ],
            "security": [
                "Review and enhance authentication mechanisms",
                "Implement proper input validation to prevent injection attacks",
                "Enhance credential management and security"
            ]
        }

        # Generate suggestions based on issues
        for issue in issues:
            issue_text = issue["issue"].lower()

            # Find matching categories
            matched_suggestions = []
            for category, category_suggs in category_suggestions.items():
                if category in issue_text:
                    # Add suggestions from this category
                    matched_suggestions.extend(category_suggs)

            # Add a suggestion if we found matches
            if matched_suggestions and len(matched_suggestions) > 0:
                sugg_index = len(suggestions) % len(matched_suggestions)
                suggestions.append(matched_suggestions[sugg_index])

        # Ensure we have at least 3 suggestions
        if len(suggestions) < 3:
            if target_type == "tool":
                suggestions.extend([
                    "Improve error handling and provide clearer error messages",
                    "Enhance documentation with more examples and use cases",
                    "Optimize performance for common usage patterns"
                ])
            else:  # API
                suggestions.extend([
                    "Improve API documentation with more comprehensive examples",
                    "Enhance error responses with clearer error codes and messages",
                    "Optimize endpoint performance and response times"
                ])

        # Remove duplicates and limit to 5
        return list(dict.fromkeys(suggestions))[:5]