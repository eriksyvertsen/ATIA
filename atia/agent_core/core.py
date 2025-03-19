"""
Enhanced Agent Core component for Phase 4.

This version defaults to using the Responses API rather than treating it as an option.
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from atia.config import settings
from atia.utils.openai_client import (
    get_completion, 
    get_completion_with_responses_api, 
    execute_tools_in_run
)
from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor
from atia.utils.cache import ResponseCache

logger = logging.getLogger(__name__)

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
        self.cache = ResponseCache()

        if tool_registry:
            self.tool_executor = ToolExecutor(tool_registry)
        else:
            self.tool_executor = None

        # Track usage statistics
        self.usage_stats = {
            "total_queries": 0,
            "tool_executions": 0,
            "responses_api_calls": 0,
            "standard_api_calls": 0,
            "average_response_time": 0,
            "last_response_time": 0,
            "session_start": datetime.now()
        }

    async def process_query(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Process a user query and return a response.

        In Phase 4, this method still exists for backward compatibility
        but delegates to the Responses API by default.

        Args:
            query: The user query
            context: Additional context about the query

        Returns:
            Agent's response
        """
        self.usage_stats["total_queries"] += 1

        if context is None:
            context = {}

        # Check cache first
        cache_key = f"{query}:{str(context)}"
        cached_response = self.cache.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_response

        start_time = datetime.now()

        # For Phase 4, we default to using Responses API unless explicitly disabled
        if not settings.disable_responses_api:
            # Use Responses API
            try:
                response_data = await self.process_query_with_responses_api(query, context=context)
                response = response_data.get("content", "")

                # Update stats
                self.usage_stats["responses_api_calls"] += 1

                # Cache the response
                self.cache.set(cache_key, response)

                # Update timing stats
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                self._update_timing_stats(duration)

                return response
            except Exception as e:
                logger.warning(f"Error using Responses API, falling back to standard API: {e}")
                # Fall back to standard API if Responses API fails

        # Use standard API as fallback
        logger.info("Using standard completion API")
        self.usage_stats["standard_api_calls"] += 1

        response = get_completion(
            prompt=query,
            system_message=self.system_prompt,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            model=settings.openai_model
        )

        # Cache the response
        self.cache.set(cache_key, response)

        # Update timing stats
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self._update_timing_stats(duration)

        return response

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
            tools = await self.tool_registry.get_tools_for_responses_api(capability_description=query)
            logger.info(f"Retrieved {len(tools)} tools from registry for query")

        start_time = datetime.now()

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
            logger.info("Response requires tool execution")
            # Update tool execution stats
            self.usage_stats["tool_executions"] += 1

            # Execute tools and get the final response
            final_response = await execute_tools_in_run(
                thread_id=response["thread_id"],
                run_id=response["run_id"],
                tool_executor=self.tool_executor
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Query processed with tool execution in {duration:.2f} seconds")

            return final_response

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Query processed in {duration:.2f} seconds")

        return response

    def _update_timing_stats(self, duration: float) -> None:
        """Update response timing statistics."""
        # Update running average
        total_queries = self.usage_stats["total_queries"]
        current_avg = self.usage_stats["average_response_time"]

        if total_queries > 1:
            # Calculate new average
            new_avg = ((current_avg * (total_queries - 1)) + duration) / total_queries
            self.usage_stats["average_response_time"] = new_avg
        else:
            # First query
            self.usage_stats["average_response_time"] = duration

        self.usage_stats["last_response_time"] = duration

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the agent.

        Returns:
            Dictionary of usage statistics
        """
        # Calculate session duration
        session_duration = (datetime.now() - self.usage_stats["session_start"]).total_seconds()

        # Create a copy of the stats with the session duration
        stats = dict(self.usage_stats)
        stats["session_duration_seconds"] = session_duration

        return stats

    async def call_component(self, component_name: str, *args, **kwargs):
        """
        Call a specific component with the given arguments.

        Args:
            component_name: The name of the component to call
            *args, **kwargs: Arguments to pass to the component

        Returns:
            The result of the component call
        """
        # This is a placeholder for Phase 4
        # In a production system, this would dynamically call the appropriate component
        return f"Called {component_name} component (Phase 4)"