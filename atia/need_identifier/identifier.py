from typing import Dict, List, Optional, Union, Any
import logging
import json

from pydantic import BaseModel

from atia.config import settings
from atia.utils.openai_client import get_json_completion, get_completion_with_responses_api


class ToolNeed(BaseModel):
    """
    Represents a detected need for a tool.
    """
    category: str
    description: str
    confidence: float


class NeedIdentifier:
    """
    Detects when the agent requires external tools based on user queries or task requirements.
    Enhanced to work with Responses API.
    """

    def __init__(self, threshold: float = settings.need_identifier_threshold):
        """
        Initialize the Need Identifier.

        Args:
            threshold: Confidence threshold for identifying a tool need
        """
        self.threshold = threshold
        self.tool_categories = self._load_tool_categories()
        self.logger = logging.getLogger(__name__)

        # Responses API function definition for need identification
        self.need_identifier_function = {
            "type": "function",
            "function": {
                "name": "identify_tool_need",
                "description": "Identifies if a tool is needed based on the user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_needed": {
                            "type": "boolean",
                            "description": "Whether a tool is needed or not"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category of tool needed (if applicable)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what the tool should do"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score from 0.0 to 1.0"
                        }
                    },
                    "required": ["tool_needed"]
                }
            }
        }

    def _load_tool_categories(self) -> List[Dict]:
        """
        Load predefined tool categories.

        Returns:
            List of tool categories with descriptions
        """
        # This is a simplified version for Phase 1
        # In later phases, this could be loaded from a database or external source
        return [
            {"name": "translation", "description": "Translate text from one language to another"},
            {"name": "weather", "description": "Get weather information for a location"},
            {"name": "calendar", "description": "Manage calendar events and schedules"},
            {"name": "search", "description": "Search for information on the web"},
            {"name": "email", "description": "Send and receive emails"},
            {"name": "maps", "description": "Get directions, find locations, and calculate distances"},
            {"name": "image_generation", "description": "Generate images from text descriptions"},
            {"name": "data_visualization", "description": "Create charts and graphs from data"},
            {"name": "sentiment_analysis", "description": "Analyze sentiment in text"},
            {"name": "summarization", "description": "Summarize long texts"},
            {"name": "flight_booking", "description": "Search for and book flights"},
            {"name": "hotel_booking", "description": "Search for and book hotel accommodations"},
            {"name": "restaurant_reservation", "description": "Make restaurant reservations"},
            {"name": "payment_processing", "description": "Process payments"},
            {"name": "stock_trading", "description": "Trade stocks and get market information"},
            {"name": "social_media", "description": "Post to social media platforms"},
            {"name": "file_conversion", "description": "Convert files from one format to another"},
            {"name": "dynamic_discovery", "description": "Dynamically discover and integrate new tools"},
        ]

    def _construct_need_detection_prompt(self, query: str, context: Dict) -> str:
        """
        Construct the prompt for need detection.

        Args:
            query: The user query
            context: Additional context about the query

        Returns:
            The constructed prompt
        """
        categories_str = "\n".join([
            f"- {cat['name']}: {cat['description']}" for cat in self.tool_categories
        ])

        return f"""
        Analyze the following user query and determine if a tool is needed:

        User Query: {query}

        Context: {context}

        Tool Categories:
        {categories_str}

        Determine:
        1. Is a tool needed to fulfill this request? (yes/no)
        2. If yes, which category best fits the need?
        3. Provide a brief description of the required tool functionality
        4. Assign a confidence score (0.0 to 1.0) to your assessment

        Respond with a JSON object with the following structure:
        {{
            "tool_needed": true/false,
            "category": "category_name",
            "description": "brief_description",
            "confidence": 0.XX
        }}
        """

    async def _get_llm_evaluation_standard(self, prompt: str) -> Dict:
        """
        Get an evaluation from the LLM using standard completion.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's evaluation as a dictionary
        """
        system_message = (
            "You are an expert at determining when a user query requires an external tool. "
            "Respond with a JSON object containing your assessment."
        )

        return get_json_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.1,  # Lower temperature for more deterministic results
            model=settings.openai_model
        )

    async def _get_llm_evaluation_responses(self, prompt: str) -> Dict:
        """
        Get an evaluation using the Responses API.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's evaluation as a dictionary
        """
        system_message = (
            "You are an expert at determining when a user query requires an external tool. "
            "Use the provided function to identify if a tool is needed."
        )

        try:
            response = await get_completion_with_responses_api(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,
                tools=[self.need_identifier_function],
                model=settings.openai_model
            )

            # Check if the function was called
            if response.get("status") == "requires_action":
                tool_calls = response.get("required_action", {}).get("submit_tool_outputs", {}).get("tool_calls", [])

                for tool_call in tool_calls:
                    if tool_call.get("function", {}).get("name") == "identify_tool_need":
                        # Extract arguments from the tool call
                        try:
                            arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                            # Return the evaluation
                            return arguments
                        except Exception as e:
                            self.logger.error(f"Error parsing tool call arguments: {e}")

            # If no function was called or there was an error, parse the text response
            content = response.get("content", "")

            # Try to find a JSON object in the content
            import re
            json_match = re.search(r'{[\s\S]*}', content)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                except Exception as e:
                    self.logger.error(f"Error parsing JSON from content: {e}")

            # Fallback to a dummy response
            return {
                "tool_needed": False,
                "category": "",
                "description": "",
                "confidence": 0.0
            }

        except Exception as e:
            self.logger.error(f"Error calling Responses API for need identification: {e}")
            # Fall back to standard completion
            return await self._get_llm_evaluation_standard(prompt)

    async def identify_tool_need(self, query: str, context: Optional[Dict] = None) -> Optional[ToolNeed]:
        """
        Determines if a tool is needed based on user query and context.

        Args:
            query: The user query
            context: Additional context about the query

        Returns:
            A ToolNeed object if threshold is exceeded, None otherwise
        """
        if context is None:
            context = {}

        prompt = self._construct_need_detection_prompt(query, context)

        # Use Responses API by default if not disabled
        if not settings.disable_responses_api:
            try:
                response = await self._get_llm_evaluation_responses(prompt)
            except Exception as e:
                self.logger.warning(f"Error using Responses API: {e}, falling back to standard completion")
                response = await self._get_llm_evaluation_standard(prompt)
        else:
            # Use standard completion
            response = await self._get_llm_evaluation_standard(prompt)

        if response.get("tool_needed", False) and response.get("confidence", 0) > self.threshold:
            return ToolNeed(
                category=response.get("category", "unknown"),
                description=response.get("description", ""),
                confidence=response.get("confidence", self.threshold)
            )

        return None