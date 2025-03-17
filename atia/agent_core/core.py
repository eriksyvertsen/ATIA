from typing import Dict, List, Optional

from atia.config import settings
from atia.utils.openai_client import get_completion


class AgentCore:
    """
    Serves as the primary orchestration component for the ATIA system.
    Handles user queries and orchestrates the flow between components.
    """

    def __init__(self, name: str = "ATIA"):
        """
        Initialize the Agent Core.

        Args:
            name: The name of the agent
        """
        self.name = name
        self.system_prompt = (
            f"You are {name}, an Autonomous Tool Integration Agent capable of "
            f"identifying when you need new tools, discovering APIs, and integrating them. "
            f"You have access to a function registry and can create new functions as needed."
        )

    async def process_query(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Process a user query and return a response.

        Args:
            query: The user query
            context: Additional context about the query

        Returns:
            Agent's response
        """
        if context is None:
            context = {}

        # For Phase 1, this is a simple passthrough to the OpenAI API
        # In later phases, this will orchestrate the full pipeline
        return get_completion(
            prompt=query,
            system_message=self.system_prompt,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            model=settings.openai_model
        )

    async def call_component(self, component_name: str, *args, **kwargs):
        """
        Call a specific component with the given arguments.

        Args:
            component_name: The name of the component to call
            *args, **kwargs: Arguments to pass to the component

        Returns:
            The result of the component call
        """
        # This is a placeholder for Phase 1
        # In later phases, this will dynamically call the appropriate component
        # based on the component_name
        return f"Called {component_name} component (placeholder for Phase 1)"