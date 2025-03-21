"""
Enhanced Agent Core component for Phase 4.

This version defaults to using the Responses API rather than treating it as an option.
"""

from typing import Dict, List, Optional, Any, Union
import logging
import time
import json
import traceback
from datetime import datetime

from atia.config import settings
from atia.utils.openai_client import (
    get_completion, 
    get_completion_with_responses_api, 
    execute_tools_in_run
)
from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor
from atia.need_identifier import NeedIdentifier, ToolNeed
from atia.api_discovery import APIDiscovery
from atia.account_manager import AccountManager
from atia.doc_processor import DocumentationProcessor
from atia.function_builder import FunctionBuilder
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
        self.tool_registry = tool_registry if tool_registry else ToolRegistry()
        self.cache = ResponseCache()

        # Initialize other components
        self.need_identifier = NeedIdentifier()
        self.api_discovery = APIDiscovery()
        self.doc_processor = DocumentationProcessor()
        self.account_manager = AccountManager()
        self.function_builder = FunctionBuilder()
        self.tool_executor = ToolExecutor(self.tool_registry, self.account_manager)

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

    async def _select_endpoint_for_need(self, endpoints: List, need: ToolNeed) -> Any:
        """Select the most appropriate endpoint for a specific need."""
        # Delegate to doc processor's endpoint selection
        return await self.doc_processor.select_endpoint_for_need(endpoints, need.description)

    async def process_query_with_tool_integration(self, query: str, context: Dict = None) -> Dict:
        """
        Process a query with full end-to-end tool integration.

        Args:
            query: User query
            context: Optional additional context

        Returns:
            Response with tool integration results
        """
        if context is None:
            context = {}

        # Track timing
        start_time = time.time()

        try:
            # 1. First, identify if a tool is needed
            tool_need = await self.need_identifier.identify_tool_need(query, context)

            if not tool_need:
                # No tool needed, process normally
                return await self.process_query_with_responses_api(query, context=context)

            logger.info(f"Tool need identified: {tool_need.category} ({tool_need.confidence:.2f})")

            # 2. Check if we have an existing tool that can handle this need
            existing_tools = await self.tool_registry.search_by_capability(tool_need.description)

            if existing_tools:
                # Use existing tools
                logger.info(f"Found {len(existing_tools)} existing tools for this need")
                tool_schemas = await self.tool_registry.get_tools_for_responses_api(tool_need.description)

                # Process the query with the existing tools
                return await self.process_query_with_responses_api(
                    query,
                    tools=tool_schemas,
                    context=context
                )

            # 3. No existing tools, so we need to find a relevant API
            logger.info("No existing tools found, searching for APIs...")
            api_candidates = await self.api_discovery.search_for_api(
                tool_need.description,
                evaluate=True
            )

            if not api_candidates:
                logger.info("No suitable APIs found")
                # No APIs found, respond with a message explaining we couldn't find an API
                return await self.process_query_with_responses_api(
                    query + "\n\nNote: I searched for APIs to help with this request but couldn't find any suitable ones.",
                    context=context
                )

            # 4. Process the top API candidate's documentation
            top_api = api_candidates[0]
            logger.info(f"Selected API: {top_api.name} (score: {top_api.relevance_score:.2f})")

            # Fetch and process documentation if needed
            doc_content = ""
            if top_api.documentation_content:
                doc_content = top_api.documentation_content
            elif top_api.documentation_url:
                try:
                    doc_content = await self.doc_processor.fetch_documentation(top_api.documentation_url)
                except Exception as e:
                    logger.error(f"Error fetching documentation: {e}")

            # Process the documentation to extract API info
            api_info = await self.doc_processor.process_documentation(
                doc_content=doc_content,
                url=top_api.documentation_url
            )

            # 5. Extract detailed endpoint information
            endpoints = await self.doc_processor.extract_endpoints(api_info)

            if not endpoints:
                logger.info("No endpoints found in documentation")
                # No endpoints found, respond normally
                return await self.process_query_with_responses_api(
                    query + f"\n\nNote: I found an API ({top_api.name}) that might help, but couldn't extract usable endpoints.",
                    context=context
                )

            # 6. Select the most appropriate endpoint for this need
            selected_endpoint = await self._select_endpoint_for_need(endpoints, tool_need)
            logger.info(f"Selected endpoint: {selected_endpoint.method} {selected_endpoint.path}")

            # 7. Get credentials for the API
            try:
                credentials = await self.account_manager.get_or_create_credentials(api_info)
                logger.info("Obtained API credentials")
            except Exception as e:
                logger.error(f"Error getting credentials: {e}")
                # Continue without credentials, the function will be created but may not work without auth
                credentials = {}

            # 8. Build a function for the endpoint
            function_def = await self.function_builder.build_function_for_api(
                api_info=api_info, 
                endpoint=selected_endpoint,
                tool_need_description=tool_need.description
            )

            # 9. Register the function as a tool
            tool = await self.tool_registry.register_function(function_def)
            logger.info(f"Registered new tool: {tool.name}")

            # 10. Generate the OpenAI tool schema
            tool_schema = self.function_builder.generate_responses_api_tool_schema(function_def)

            # 11. Use the new tool to process the query
            logger.info("Using newly created tool to process query")
            result = await self.process_query_with_responses_api(
                query,
                tools=[tool_schema],
                context=context
            )

            # 12. Handle tool execution if required
            if result.get("status") == "requires_action":
                logger.info("Tool execution required")
                final_result = await execute_tools_in_run(
                    thread_id=result["thread_id"],
                    run_id=result["run_id"],
                    tool_executor=self.tool_executor
                )
                return final_result

            return result

        except Exception as e:
            # Log the error and traceback
            logger.error(f"Error in tool integration: {e}")
            logger.debug(traceback.format_exc())

            # Return a response explaining the issue
            error_response = await self.process_query_with_responses_api(
                query + f"\n\nNote: I attempted to find and use a tool for this, but encountered an error: {str(e)}",
                context=context
            )

            # Add error information to the response
            if isinstance(error_response, dict):
                error_response["error"] = {
                    "message": str(e),
                    "type": type(e).__name__
                }

            return error_response
        finally:
            # Log total processing time
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            logger.info(f"Tool integration process completed in {duration_ms:.2f}ms")

            # Update usage stats
            self.usage_stats["total_queries"] += 1

    async def call_component(self, component_name: str, *args, **kwargs):
        """
        Call a specific component with the given arguments.

        Args:
            component_name: The name of the component to call
            *args, **kwargs: Arguments to pass to the component

        Returns:
            The result of the component call
        """
        # Map component names to actual components
        components = {
            "need_identifier": self.need_identifier,
            "api_discovery": self.api_discovery,
            "doc_processor": self.doc_processor,
            "account_manager": self.account_manager,
            "function_builder": self.function_builder,
            "tool_registry": self.tool_registry,
            "tool_executor": self.tool_executor
        }

        # Get the component
        component = components.get(component_name)
        if not component:
            raise ValueError(f"Unknown component: {component_name}")

        # Find the method to call
        method_name = kwargs.pop("method", None)
        if not method_name:
            raise ValueError("Method name is required")

        method = getattr(component, method_name, None)
        if not method or not callable(method):
            raise ValueError(f"Invalid method: {method_name}")

        # Call the method
        return await method(*args, **kwargs)