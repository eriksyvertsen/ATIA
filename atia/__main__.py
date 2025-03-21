#!/usr/bin/env python

"""
Command-line interface for the ATIA system with full tool integration.
"""

import argparse
import asyncio
import sys
import json
import os
import logging
from typing import Dict, Optional, Any

from atia.agent_core import AgentCore
from atia.tool_registry import ToolRegistry
from atia.config import settings


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def process_query(query: str, context: Optional[Dict] = None, use_tool_integration: bool = True) -> None:
    """
    Process a user query and display the results.

    Args:
        query: The user query
        context: Additional context about the query
        use_tool_integration: Whether to use full tool integration flow
    """
    if context is None:
        context = {}

    # Create data directories if they don't exist
    os.makedirs("data/tools", exist_ok=True)
    os.makedirs("data/functions", exist_ok=True)
    os.makedirs("data/credentials", exist_ok=True)
    os.makedirs("data/analytics", exist_ok=True)

    # Initialize components
    logger.info("Initializing ATIA components...")
    tool_registry = ToolRegistry()
    agent = AgentCore(tool_registry=tool_registry)

    print("\n🤖 ATIA - Autonomous Tool Integration Agent\n")
    print(f"Query: {query}\n")

    # Process the query
    start_time = asyncio.get_event_loop().time()

    if use_tool_integration:
        # Use the full tool integration process
        result = await agent.process_query_with_tool_integration(query, context)

        # Extract the response text or error
        if isinstance(result, dict) and "content" in result:
            response_text = result.get("content", "")
        elif isinstance(result, dict) and "error" in result:
            response_text = f"Error: {result['error'].get('message', 'Unknown error')}"
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = "No response generated"

        print(f"Response: {response_text}\n")
    else:
        # Use legacy processing without tool integration
        response = await agent.process_query(query, context)
        print(f"Response: {response}\n")

        # Identify if a tool is needed
        print("🔍 Identifying tool needs...")
        tool_need = await agent.need_identifier.identify_tool_need(query, context)

        if tool_need:
            print(f"Tool need identified: {tool_need.category}")
            print(f"Description: {tool_need.description}")
            print(f"Confidence: {tool_need.confidence:.2f}\n")

            # Discover APIs for the identified need
            print("🔎 Searching for relevant APIs...")
            api_candidates = await agent.api_discovery.search_for_api(tool_need.description)

            if api_candidates:
                top_api = api_candidates[0]
                print(f"Top API candidate: {top_api.name}")
                print(f"Provider: {top_api.provider}")
                print(f"Description: {top_api.description}")
                print(f"Documentation URL: {top_api.documentation_url}")
                print(f"Relevance score: {top_api.relevance_score:.2f}\n")

    # Calculate and print execution time
    end_time = asyncio.get_event_loop().time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Processed in {execution_time:.2f}ms")


async def interactive_mode(use_tool_integration: bool = True) -> None:
    """
    Run in interactive mode, processing queries from the user.

    Args:
        use_tool_integration: Whether to use full tool integration flow
    """
    print("🤖 ATIA - Autonomous Tool Integration Agent")
    print(f"{'Using full tool integration' if use_tool_integration else 'Using standard mode'}")
    print("Type 'exit' or 'quit' to exit.")

    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ["exit", "quit"]:
            break

        await process_query(query, use_tool_integration=use_tool_integration)


async def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="ATIA - Autonomous Tool Integration Agent")
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--standard", "-s", action="store_true", help="Use standard processing (without tool integration)")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine whether to use tool integration
    use_tool_integration = not args.standard

    if args.interactive:
        await interactive_mode(use_tool_integration)
    elif args.query:
        if args.json:
            # Initialize agent for JSON output
            tool_registry = ToolRegistry()
            agent = AgentCore(tool_registry=tool_registry)

            # Process query and output as JSON
            result = await agent.process_query_with_tool_integration(args.query) if use_tool_integration else await agent.process_query(args.query)
            print(json.dumps(result, indent=2, default=str))
        else:
            # Process query with human-readable output
            await process_query(args.query, use_tool_integration=use_tool_integration)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)