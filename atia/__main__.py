#!/usr/bin/env python

"""
Command-line interface for the ATIA system.

This module provides a simple CLI for interacting with the ATIA system.
"""

import argparse
import asyncio
import sys
from typing import Dict, Optional

from atia.agent_core import AgentCore
from atia.need_identifier import NeedIdentifier
from atia.api_discovery import APIDiscovery
from atia.doc_processor import DocumentationProcessor


async def process_query(query: str, context: Optional[Dict] = None) -> None:
    """
    Process a user query and display the results.

    Args:
        query: The user query
        context: Additional context about the query
    """
    if context is None:
        context = {}

    # Initialize components
    agent = AgentCore()
    need_identifier = NeedIdentifier()
    api_discovery = APIDiscovery()
    doc_processor = DocumentationProcessor()

    print("\n🤖 ATIA - Autonomous Tool Integration Agent\n")
    print(f"Query: {query}\n")

    # Step 1: Process the query with the agent
    print("🧠 Processing query...")
    agent_response = await agent.process_query(query, context)
    print(f"Initial response: {agent_response}\n")

    # Step 2: Identify if a tool is needed
    print("🔍 Identifying tool needs...")
    tool_need = await need_identifier.identify_tool_need(query, context)

    if tool_need:
        print(f"Tool need identified: {tool_need.category}")
        print(f"Description: {tool_need.description}")
        print(f"Confidence: {tool_need.confidence:.2f}\n")

        # Step 3: Discover APIs for the identified need
        print("🔎 Searching for relevant APIs...")
        api_candidates = await api_discovery.search_for_api(tool_need.description)

        if api_candidates:
            top_api = api_candidates[0]
            print(f"Top API candidate: {top_api.name}")
            print(f"Provider: {top_api.provider}")
            print(f"Description: {top_api.description}")
            print(f"Documentation URL: {top_api.documentation_url}")
            print(f"Relevance score: {top_api.relevance_score:.2f}\n")

            # For Phase 1, we'll just demonstrate fetching the documentation
            print("📝 Fetching API documentation...")
            try:
                print(f"API documentation would be processed here in later phases.")
                print("For Phase 1, this is a placeholder for documentation processing.")
            except Exception as e:
                print(f"Error fetching documentation: {e}")
        else:
            print("No suitable APIs found for this need.")
    else:
        print("No tool need identified for this query.")


async def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="ATIA - Autonomous Tool Integration Agent")
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        print("🤖 ATIA - Autonomous Tool Integration Agent")
        print("Type 'exit' or 'quit' to exit.")

        while True:
            query = input("\nEnter your query: ")

            if query.lower() in ["exit", "quit"]:
                break

            await process_query(query)
    elif args.query:
        await process_query(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)