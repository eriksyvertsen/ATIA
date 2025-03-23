"""
Enhanced Agent Core component for Phase 1: Week 1.

This version integrates fully with the Responses API for autonomous tool discovery and execution.
"""

from typing import Dict, List, Optional, Any, Union
import logging
import time
import json
import traceback
import asyncio
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

        # Define the dynamic tool discovery function
        self.dynamic_tool_discovery_schema = {
            "type": "function",
            "function": {
                "name": "dynamic_tool_discovery",
                "description": "Discovers and integrates new APIs based on capability needs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "capability": {
                            "type": "string", 
                            "description": "Description of the capability needed"
                        },
                        "api_type": {
                            "type": "string",
                            "description": "Type of API to discover (e.g., 'rest', 'graphql')"
                        }
                    },
                    "required": ["capability"]
                }
            }
        }

    async def process_query(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Process a user query and return a response.

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

        # In Phase 1, always use Responses API unless explicitly disabled
        if not settings.disable_responses_api:
            # Use Responses API with dynamic tool discovery
            try:
                response_data = await self._process_with_dynamic_discovery(query, context)
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
                logger.warning(f"Error using Responses API with dynamic discovery: {e}")
                # Fall back to standard processing with Responses API
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
                except Exception as e2:
                    logger.warning(f"Error using standard Responses API: {e2}, falling back to standard API")

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

    async def _process_with_dynamic_discovery(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a query using the Responses API with dynamic tool discovery.

        Args:
            query: The user query
            context: Additional context about the query

        Returns:
            Response from the Responses API with dynamic tool discovery
        """
        logger.info("Processing query with dynamic tool discovery")

        # Get existing tools from the registry
        existing_tools = await self.tool_registry.get_tools_for_responses_api()

        # Add the dynamic tool discovery function to the available tools
        tools = existing_tools + [self.dynamic_tool_discovery_schema]

        # Enhance the prompt with context if available
        enhanced_prompt = query
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            enhanced_prompt = f"{query}\n\nContext:\n{context_str}"

        # Process with Responses API
        response = await get_completion_with_responses_api(
            prompt=enhanced_prompt,
            system_message=self.system_prompt,
            temperature=settings.openai_temperature,
            tools=tools,
            model=settings.openai_model
        )

        # Check if tool execution is required
        if response.get("status") == "requires_action":
            # Check if the dynamic_tool_discovery function was called
            tool_calls = response.get("required_action", {}).get("submit_tool_outputs", {}).get("tool_calls", [])

            for tool_call in tool_calls:
                # Check if this is a dynamic tool discovery call
                if tool_call.get("function", {}).get("name") == "dynamic_tool_discovery":
                    # Extract the capability from the tool call
                    try:
                        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                        capability = arguments.get("capability", "")

                        if capability:
                            # Log the capability
                            logger.info(f"Dynamic tool discovery requested for capability: {capability}")

                            # Process tool discovery
                            discovered_tool = await self._discover_and_create_tool(capability)

                            # Return tool discovery result
                            tool_output = {
                                "tool_call_id": tool_call.get("id"),
                                "output": json.dumps({
                                    "success": True if discovered_tool else False,
                                    "tool_name": discovered_tool.name if discovered_tool else None,
                                    "capability": capability,
                                    "message": "Tool successfully created" if discovered_tool else "No suitable API found"
                                })
                            }

                            # Submit the tool output
                            return await execute_tools_in_run(
                                thread_id=response.get("thread_id"),
                                run_id=response.get("run_id"),
                                tool_executor=self.tool_executor,
                                tool_outputs=[tool_output]
                            )
                    except Exception as e:
                        logger.error(f"Error processing dynamic tool discovery: {e}")
                        # Submit error as tool output
                        tool_output = {
                            "tool_call_id": tool_call.get("id"),
                            "output": json.dumps({
                                "success": False,
                                "error": str(e)
                            })
                        }

                        return await execute_tools_in_run(
                            thread_id=response.get("thread_id"),
                            run_id=response.get("run_id"),
                            tool_executor=self.tool_executor,
                            tool_outputs=[tool_output]
                        )

            # For other tool calls, let the tool executor handle them
            return await execute_tools_in_run(
                thread_id=response.get("thread_id"),
                run_id=response.get("run_id"),
                tool_executor=self.tool_executor
            )

        return response

    async def _discover_and_create_tool(self, capability: str):
        """
        Discover and create a tool for a specific capability.

        Args:
            capability: The capability needed

        Returns:
            Created tool or None if no suitable API found
        """
        try:
            # Create a tool need
            tool_need = ToolNeed(
                category="dynamic_discovery",
                description=capability,
                confidence=1.0
            )

            # Check if we have an existing tool that can handle this need
            existing_tools = await self.tool_registry.search_by_capability(tool_need.description)

            if existing_tools:
                # Return the most relevant existing tool
                return existing_tools[0]

            # No existing tool, search for APIs
            api_candidates = await self.api_discovery.search_for_api(
                tool_need.description,
                evaluate=True
            )

            if not api_candidates:
                logger.info("No suitable APIs found")
                return None

            # Select the top API candidate
            top_api = api_candidates[0]
            logger.info(f"Selected API: {top_api.name} (score: {top_api.relevance_score:.2f})")

            # Fetch and process documentation
            doc_content = ""
            if top_api.documentation_content:
                doc_content = top_api.documentation_content
            elif top_api.documentation_url:
                try:
                    doc_content = await self.doc_processor.fetch_documentation(top_api.documentation_url)
                except Exception as e:
                    logger.error(f"Error fetching documentation: {e}")

            # Process the documentation
            api_info = await self.doc_processor.process_documentation(
                doc_content=doc_content,
                url=top_api.documentation_url
            )

            # Extract endpoints
            endpoints = await self.doc_processor.extract_endpoints(api_info)

            if not endpoints:
                logger.info("No endpoints found in documentation")
                return None

            # Select the most appropriate endpoint
            selected_endpoint = await self._select_endpoint_for_need(endpoints, tool_need)
            logger.info(f"Selected endpoint: {selected_endpoint.method} {selected_endpoint.path}")

            # Get credentials for the API
            try:
                credentials = await self.account_manager.get_or_create_credentials(api_info)
                logger.info("Obtained API credentials")
            except Exception as e:
                logger.error(f"Error getting credentials: {e}")
                credentials = {}

            # Build a function for the endpoint
            function_def = await self.function_builder.build_function_for_api(
                api_info=api_info, 
                endpoint=selected_endpoint,
                tool_need_description=tool_need.description
            )

            # Register the function as a tool
            tool = await self.tool_registry.register_function(function_def)
            logger.info(f"Registered new tool: {tool.name}")

            return tool

        except Exception as e:
            logger.error(f"Error in _discover_and_create_tool: {e}")
            return None

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

    async def process_query_with_tool_integration(self, query: str, context: Dict = None) -> Dict:
        """
        Process a query with full end-to-end tool integration.

        Note: This method is now a wrapper around _process_with_dynamic_discovery in Phase 1.

        Args:
            query: User query
            context: Optional additional context

        Returns:
            Response with tool integration results
        """
        if context is None:
            context = {}

        # In Phase 1, we're focusing on the Responses API approach, so delegate to that method
        return await self._process_with_dynamic_discovery(query, context)

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