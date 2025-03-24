"""
Integration helper to connect the enhanced components with Agent Core.

This module provides functions to integrate the enhanced API Discovery and
Function Builder components with the Agent Core.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from atia.config import settings
from atia.api_discovery.discovery import APICandidate
from atia.doc_processor.processor import APIInfo, APIEndpoint
from atia.function_builder.models import FunctionDefinition
from atia.need_identifier.identifier import ToolNeed

logger = logging.getLogger(__name__)

async def discover_and_generate_tool(
    agent_core,
    capability: str,
    use_responses_api: bool = True
) -> Tuple[Optional[FunctionDefinition], Optional[APICandidate]]:
    """
    Discover APIs and generate tool functions in one integrated step.

    This helper function integrates the enhanced API Discovery and Function Builder
    components to create tools for the Agent Core.

    Args:
        agent_core: Instance of AgentCore
        capability: Description of the capability needed
        use_responses_api: Whether to use Responses API integration

    Returns:
        Tuple of (function_definition, api_candidate) or (None, None) if failed
    """
    try:
        logger.info(f"Starting integrated tool discovery and generation for capability: {capability}")

        # Step 1: Create tool need
        tool_need = ToolNeed(
            category="dynamic_discovery",
            description=capability,
            confidence=1.0
        )

        # Step 2: Check if we have existing tools
        existing_tools = await agent_core.tool_registry.search_by_capability(tool_need.description)
        if existing_tools:
            logger.info(f"Found existing tool: {existing_tools[0].name}")
            return None, None

        # Step 3: Search for APIs using enhanced search
        api_candidates = await agent_core.api_discovery.enhanced_api_search(
            capability_description=tool_need.description,
            num_results=5,
            use_responses_api=use_responses_api
        )

        if not api_candidates:
            logger.info("No suitable APIs found")
            return None, None

        # Step 4: Select the top API candidate
        top_api = api_candidates[0]
        logger.info(f"Selected API: {top_api.name} (score: {top_api.relevance_score:.2f})")

        # Step 5: Fetch and process documentation
        doc_content = ""
        if hasattr(top_api, 'documentation_content') and top_api.documentation_content:
            doc_content = top_api.documentation_content
        elif top_api.documentation_url:
            try:
                doc_content = await agent_core.doc_processor.fetch_documentation(top_api.documentation_url)
            except Exception as e:
                logger.error(f"Error fetching documentation: {e}")

        # Step 6: Process the documentation
        api_info = await agent_core.doc_processor.process_documentation(
            doc_content=doc_content,
            url=top_api.documentation_url
        )

        # Step 7: Extract endpoints
        endpoints = await agent_core.doc_processor.extract_endpoints(api_info)
        if not endpoints:
            logger.info("No endpoints found in documentation")
            return None, None

        # Step 8: Select the most appropriate endpoint
        selected_endpoint = await agent_core.doc_processor.select_endpoint_for_need(
            endpoints, tool_need.description
        )
        logger.info(f"Selected endpoint: {selected_endpoint.method} {selected_endpoint.path}")

        # Step 9: Build function with enhanced generation
        function_def = await agent_core.function_builder.enhanced_function_generation(
            api_info=api_info,
            endpoint=selected_endpoint,
            capability_description=tool_need.description
        )

        # Return the function definition and API candidate
        return function_def, top_api

    except Exception as e:
        logger.error(f"Error in discover_and_generate_tool: {e}")
        return None, None

async def register_function_as_tool(
    agent_core,
    function_def: FunctionDefinition,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Register a function as a tool in the Tool Registry.

    Args:
        agent_core: Instance of AgentCore
        function_def: The function definition to register
        metadata: Additional metadata for the function

    Returns:
        The registered tool
    """
    if metadata is None:
        metadata = {}

    try:
        # Register the function with the tool registry
        tool = await agent_core.tool_registry.register_function(function_def, metadata)
        logger.info(f"Registered new tool: {tool.name}")
        return tool
    except Exception as e:
        logger.error(f"Error registering function as tool: {e}")
        return None

async def prepare_tools_for_responses_api(
    agent_core,
    capability_description: Optional[str] = None
) -> List[Dict]:
    """
    Prepare all relevant tools for use with the Responses API.

    Args:
        agent_core: Instance of AgentCore
        capability_description: Optional capability description to filter tools

    Returns:
        List of tools in Responses API format
    """
    try:
        # Get tools from registry
        tools = await agent_core.tool_registry.get_tools_for_responses_api(capability_description)

        # Add dynamic tool discovery function
        tools.append(agent_core.dynamic_tool_discovery_schema)

        return tools
    except Exception as e:
        logger.error(f"Error preparing tools for Responses API: {e}")
        # Return just the dynamic tool discovery schema as fallback
        return [agent_core.dynamic_tool_discovery_schema]

async def execute_tool_integration_flow(
    agent_core,
    query: str,
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute the full tool integration flow with enhanced components.

    This function integrates all the enhanced components to process a user query
    with dynamic tool discovery and execution.

    Args:
        agent_core: Instance of AgentCore
        query: The user query
        context: Additional context about the query

    Returns:
        Response with tool integration results
    """
    if context is None:
        context = {}

    try:
        # Step 1: Identify if a tool is needed
        tool_need = await agent_core.need_identifier.identify_tool_need(query, context)

        if not tool_need:
            # No tool needed, process with standard completion
            return await agent_core.process_query_with_responses_api(query, context=context)

        # Step 2: Prepare available tools
        tools = await prepare_tools_for_responses_api(agent_core, tool_need.description)

        # Step 3: Process with Responses API and dynamic tool discovery
        response = await agent_core._process_with_dynamic_discovery(query, context)

        return response
    except Exception as e:
        logger.error(f"Error in execute_tool_integration_flow: {e}")
        # Fall back to standard completion
        try:
            return await agent_core.process_query(query, context)
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return {"content": f"Error processing your request: {str(e)}"}
