from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from atia.config import settings
from atia.utils.openai_client import get_json_completion


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
    """

    def __init__(self, threshold: float = settings.need_identifier_threshold):
        """
        Initialize the Need Identifier.

        Args:
            threshold: Confidence threshold for identifying a tool need
        """
        self.threshold = threshold
        self.tool_categories = self._load_tool_categories()

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

    def _get_llm_evaluation(self, prompt: str) -> Dict:
        """
        Get an evaluation from the LLM.

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
        response = self._get_llm_evaluation(prompt)

        if response.get("tool_needed", False) and response.get("confidence", 0) > self.threshold:
            return ToolNeed(
                category=response["category"],
                description=response["description"],
                confidence=response["confidence"]
            )

        return None