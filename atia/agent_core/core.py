from typing import Dict, List, Optional

from atia.config import settings
from atia.utils.openai_client import get_completion, get_completion_with_responses_api, execute_tools_in_run
from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor


class AgentCore:
    """
    Serves as the primary orchestration component for the ATIA system.
    Handles user queries and orchestrates the flow between components.
    """

    def __init__(self, name: str = "ATIA", tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize the Agent Core.

        Args:
            name: The name of the agent
            tool_registry: Optional Tool Registry for tool lookup
        """
        self.name = name
        self.system_prompt = (
            f"You are {name}, an Autonomous Tool Integration Agent capable of "
            f"identifying when you need new tools, discovering APIs, and integrating them. "
            f"You have access to a function registry and can create new functions as needed."
        )
        self.tool_registry = tool_registry
        if tool_registry:
            self.tool_executor = ToolExecutor(tool_registry)
        else:
            self.tool_executor = None

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

    async def process_query_with_responses_api(
        self, 
        query: str, 
        tools: List[Dict] = None, 
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Process a query using the Responses API with available tools.

        Args:
            query: The user query
            tools: Optional list of tools to make available
            context: Additional context about the query

        Returns:
            Response from the Responses API
        """
        if context is None:
            context = {}

        # Enhance the prompt with context if available
        enhanced_prompt = query
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            enhanced_prompt = f"{query}\n\nContext:\n{context_str}"

        # If tools is None and we have a tool registry, get tools from there
        if tools is None and self.tool_registry:
            tools = await self.tool_registry.get_tools_for_responses_api()

        # Process the query with the Responses API
        response = await get_completion_with_responses_api(
            prompt=enhanced_prompt,
            system_message=self.system_prompt,
            temperature=settings.openai_temperature,
            tools=tools,
            model=settings.openai_model
        )

        # If the response indicates that tools need to be executed and we have a tool executor
        if response.get("status") == "requires_action" and self.tool_executor:
            # Execute tools and get the final response
            final_response = await execute_tools_in_run(
                thread_id=response["thread_id"],
                run_id=response["run_id"],
                tool_executor=self.tool_executor
            )

            return final_response

        return response