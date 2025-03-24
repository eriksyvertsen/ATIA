"""
Apply Agent Core integration for Phase 1: Week 2.

This module integrates the enhanced API Discovery and Function Builder components
with the Agent Core.
"""

import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def import_or_fail(module_name: str) -> bool:
    """Import a module or fail with useful message."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        logger.error(f"Error importing {module_name}. Make sure the module exists.")
        return False

def integrate_agent_core() -> None:
    """Integrate updated components with Agent Core."""
    logger.info("Integrating enhanced components with Agent Core...")

    # Import necessary modules
    if not import_or_fail("atia.agent_core.core"):
        return

    if not import_or_fail("atia.api_discovery.updates"):
        return

    if not import_or_fail("atia.function_builder.updates"):
        return

    if not import_or_fail("atia.utils.integration_helper"):
        return

    # Apply the updates to the real classes
    from atia.agent_core.core import AgentCore
    from atia.api_discovery.discovery import APIDiscovery
    from atia.function_builder.builder import FunctionBuilder

    # Apply API Discovery updates
    from atia.api_discovery.updates import (
        search_with_responses_api,
        evaluate_candidates_with_responses_api,
        enhanced_api_search
    )

    APIDiscovery.search_with_responses_api = search_with_responses_api
    APIDiscovery.evaluate_candidates_with_responses_api = evaluate_candidates_with_responses_api
    APIDiscovery.enhanced_api_search = enhanced_api_search

    # Apply Function Builder updates
    from atia.function_builder.updates import (
        enhanced_responses_api_tool_schema,
        enhanced_function_generation
    )

    FunctionBuilder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema
    FunctionBuilder.enhanced_function_generation = enhanced_function_generation

    # Extend AgentCore to use the integration helper
    from atia.utils.integration_helper import (
        discover_and_generate_tool,
        register_function_as_tool,
        prepare_tools_for_responses_api,
        execute_tool_integration_flow
    )

    # Add the new methods to AgentCore
    AgentCore.discover_and_generate_tool = discover_and_generate_tool
    AgentCore.register_function_as_tool = register_function_as_tool
    AgentCore.prepare_tools_for_responses_api = prepare_tools_for_responses_api
    AgentCore.execute_tool_integration_flow = execute_tool_integration_flow

    logger.info("Integration complete. Enhanced components now available in AgentCore.")

if __name__ == "__main__":
    integrate_agent_core()
