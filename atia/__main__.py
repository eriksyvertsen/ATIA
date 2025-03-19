#!/usr/bin/env python

"""
Enhanced CLI entry point for ATIA Phase 4.

This module provides an improved command-line interface with support for
all Phase 4 features, analytics dashboard, and improved error handling.
"""

import argparse
import asyncio
import sys
import json
import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from atia.agent_core import AgentCore
from atia.need_identifier import NeedIdentifier
from atia.api_discovery import APIDiscovery
from atia.doc_processor import DocumentationProcessor
from atia.account_manager import AccountManager
from atia.tool_registry import ToolRegistry
from atia.function_builder import FunctionBuilder
from atia.tool_executor import ToolExecutor
from atia.utils.analytics import dashboard, track_agent_query
from atia.utils.error_handling import ATIAError, ErrorContext, AsyncErrorContext
from atia.config import settings, print_settings


logger = logging.getLogger(__name__)


class ATIASession:
    """
    Session manager for ATIA components.

    Handles initialization and coordination of all ATIA components.
    """

    def __init__(self):
        """Initialize the ATIA session."""
        self.session_id = f"s-{int(time.time())}"
        self.logger = logging.getLogger(__name__)

        # Track component init times
        self.init_times = {}

        # Initialize components
        self.logger.info("Initializing ATIA components...")
        self.tool_registry = self._timed_init("tool_registry", self._init_tool_registry)
        self.tool_executor = self._timed_init("tool_executor", self._init_tool_executor)
        self.agent_core = self._timed_init("agent_core", self._init_agent_core)
        self.need_identifier = self._timed_init("need_identifier", self._init_need_identifier)
        self.api_discovery = self._timed_init("api_discovery", self._init_api_discovery)
        self.doc_processor = self._timed_init("doc_processor", self._init_doc_processor)
        self.account_manager = self._timed_init("account_manager", self._init_account_manager)
        self.function_builder = self._timed_init("function_builder", self._init_function_builder)

        # Log initialization results
        total_init_time = sum(self.init_times.values())
        self.logger.info(f"ATIA initialized in {total_init_time:.2f}ms")
        for component, time_ms in self.init_times.items():
            self.logger.debug(f"  {component}: {time_ms:.2f}ms ({(time_ms/total_init_time)*100:.1f}%)")

    def _timed_init(self, component_name: str, init_func: callable) -> Any:
        """
        Time the initialization of a component.

        Args:
            component_name: Name of the component
            init_func: Initialization function

        Returns:
            Initialized component
        """
        start_time = time.time()
        try:
            component = init_func()
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            self.init_times[component_name] = duration_ms
            return component
        except Exception as e:
            self.logger.error(f"Error initializing {component_name}: {e}")
            self.init_times[component_name] = 0
            raise

    def _init_tool_registry(self) -> ToolRegistry:
        """Initialize the tool registry."""
        return ToolRegistry()

    def _init_tool_executor(self) -> ToolExecutor:
        """Initialize the tool executor."""
        return ToolExecutor(self.tool_registry)

    def _init_agent_core(self) -> AgentCore:
        """Initialize the agent core."""
        return AgentCore(tool_registry=self.tool_registry)

    def _init_need_identifier(self) -> NeedIdentifier:
        """Initialize the need identifier."""
        return NeedIdentifier()

    def _init_api_discovery(self) -> APIDiscovery:
        """Initialize the API discovery component."""
        return APIDiscovery()

    def _init_doc_processor(self) -> DocumentationProcessor:
        """Initialize the documentation processor."""
        return DocumentationProcessor()

    def _init_account_manager(self) -> AccountManager:
        """Initialize the account manager."""
        return AccountManager()

    def _init_function_builder(self) -> FunctionBuilder:
        """Initialize the function builder."""
        return FunctionBuilder()

    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a user query and return a detailed response.

        Args:
            query: The user query
            context: Additional context about the query

        Returns:
            Dictionary with response details
        """
        if context is None:
            context = {}

        start_time = time.time()

        # Create response object
        response = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "response": None,
            "tool_need": None,
            "api_candidate": None,
            "processing_details": {},
            "error": None
        }

        try:
            self.logger.info(f"Processing query: {query}")

            # Process with agent core
            agent_response = await self.agent_core.process_query(query, context)
            response["response"] = agent_response

            # Identify if a tool is needed
            self.logger.info("Identifying tool needs...")
            tool_need = await self.need_identifier.identify_tool_need(query, context)

            if tool_need:
                self.logger.info(f"Tool need identified: {tool_need.category}")

                # Add tool need to response
                response["tool_need"] = {
                    "category": tool_need.category,
                    "description": tool_need.description,
                    "confidence": tool_need.confidence
                }

                # Discover APIs for the identified need
                self.logger.info("Searching for relevant APIs...")
                api_candidates = await self.api_discovery.search_for_api(
                    tool_need.description, 
                    evaluate=True
                )

                if api_candidates:
                    # Add top API candidate to response
                    top_api = api_candidates[0]
                    response["api_candidate"] = {
                        "name": top_api.name,
                        "provider": top_api.provider,
                        "description": top_api.description,
                        "documentation_url": top_api.documentation_url,
                        "relevance_score": top_api.relevance_score
                    }

                    # Process API documentation if needed for more advanced actions
                    # This part would be expanded in a full implementation of Phase 4

            # Track query for analytics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            track_agent_query(
                query=query,
                duration_ms=duration_ms,
                session_id=self.session_id,
                metadata={
                    "has_tool_need": tool_need is not None,
                    "response_length": len(response["response"]) if response["response"] else 0
                }
            )

            # Add performance details
            response["processing_details"] = {
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            }

            return response

        except ATIAError as e:
            # Handle ATIA errors gracefully
            self.logger.error(f"ATIA error processing query: {e}")

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            response["error"] = {
                "type": e.__class__.__name__,
                "message": str(e),
                "component": e.component,
                "details": e.details
            }

            response["processing_details"] = {
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

            return response

        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error processing query: {e}")
            traceback.print_exc()

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            response["error"] = {
                "type": "UnexpectedException",
                "message": str(e),
                "details": {
                    "traceback": traceback.format_exc()
                }
            }

            response["processing_details"] = {
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

            return response


async def process_query_command(query: str) -> None:
    """
    Process a single query and display the results.

    Args:
        query: The query to process
    """
    session = ATIASession()
    result = await session.process_query(query)

    # Print the query and response
    print("\n🤖 ATIA - Autonomous Tool Integration Agent\n")
    print(f"Query: {query}\n")

    # Print the response
    if result["error"]:
        print("❌ Error processing query:")
        print(f"  Type: {result['error']['type']}")
        print(f"  Message: {result['error']['message']}")
        if result["error"].get("component"):
            print(f"  Component: {result['error']['component']}")
    else:
        print(f"Response: {result['response']}\n")

        # Print tool need if present
        if result["tool_need"]:
            print("\n🔍 Tool Need Identified:")
            print(f"  Category: {result['tool_need']['category']}")
            print(f"  Description: {result['tool_need']['description']}")
            print(f"  Confidence: {result['tool_need']['confidence']:.2f}\n")

            # Print API candidate if present
            if result["api_candidate"]:
                print("🔎 Top API Candidate:")
                print(f"  Name: {result['api_candidate']['name']}")
                print(f"  Provider: {result['api_candidate']['provider']}")
                print(f"  Description: {result['api_candidate']['description']}")
                print(f"  Documentation URL: {result['api_candidate']['documentation_url']}")
                print(f"  Relevance score: {result['api_candidate']['relevance_score']:.2f}\n")

    # Print processing details
    duration = result["processing_details"].get("duration_ms", 0)
    print(f"\nProcessed in {duration:.2f}ms")


async def interactive_mode() -> None:
    """
    Run in interactive mode, processing queries from the user.
    """
    session = ATIASession()

    print("🤖 ATIA - Autonomous Tool Integration Agent (Phase 4)")
    print("Type 'exit' or 'quit' to exit, 'help' for commands.")

    while True:
        try:
            query = input("\nEnter your query: ")

            # Check for exit command
            if query.lower() in ["exit", "quit"]:
                break

            # Check for help command
            if query.lower() in ["help", "commands", "?"]:
                print("\nAvailable commands:")
                print("  help: Show this help message")
                print("  exit, quit: Exit the program")
                print("  !stats: Show usage statistics")
                print("  !dashboard: Generate analytics dashboard")
                print("  !settings: Show current settings")
                print("  !clear: Clear the screen")
                continue

            # Check for special commands
            if query.startswith("!"):
                handle_special_command(query)
                continue

            # Process the query
            result = await session.process_query(query)

            # Print the response
            if result["error"]:
                print("❌ Error processing query:")
                print(f"  Type: {result['error']['type']}")
                print(f"  Message: {result['error']['message']}")
                if result["error"].get("component"):
                    print(f"  Component: {result['error']['component']}")
            else:
                print(f"\nResponse: {result['response']}")

                # Print tool need if present
                if result["tool_need"]:
                    print("\n🔍 Tool Need Identified:")
                    print(f"  Category: {result['tool_need']['category']}")
                    print(f"  Description: {result['tool_need']['description']}")
                    print(f"  Confidence: {result['tool_need']['confidence']:.2f}")

                    # Print API candidate if present
                    if result["api_candidate"]:
                        print("\n🔎 Top API Candidate:")
                        print(f"  Name: {result['api_candidate']['name']}")
                        print(f"  Provider: {result['api_candidate']['provider']}")
                        print(f"  Description: {result['api_candidate']['description']}")
                        print(f"  Documentation URL: {result['api_candidate']['documentation_url']}")
                        print(f"  Relevance score: {result['api_candidate']['relevance_score']:.2f}")

            # Print processing time
            duration = result["processing_details"].get("duration_ms", 0)
            print(f"\nProcessed in {duration:.2f}ms")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def handle_special_command(command: str) -> None:
    """
    Handle special commands in interactive mode.

    Args:
        command: The command to handle
    """
    command = command.lower()

    if command == "!stats":
        # Show usage statistics
        metrics = dashboard.get_metrics()

        print("\n📊 Usage Statistics:")
        print(f"  Total queries: {metrics.total_queries}")
        print(f"  Total tool executions: {metrics.total_tool_executions}")
        print(f"  Average query time: {metrics.avg_query_duration_ms:.2f}ms")
        print(f"  Session started: {metrics.session_start.strftime('%Y-%m-%d %H:%M:%S')}")

    elif command == "!dashboard":
        # Generate analytics dashboard
        from atia.utils.analytics import save_dashboard_html

        dashboard_path = "data/analytics/dashboard.html"
        save_dashboard_html(dashboard_path)

        print(f"\n📈 Dashboard saved to {dashboard_path}")
        print("Open this file in a web browser to view the dashboard.")

    elif command == "!settings":
        # Show current settings
        print("\n⚙️ " + print_settings())

    elif command == "!clear":
        # Clear the screen
        import os
        os.system('cls' if os.name == 'nt' else 'clear')


async def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="ATIA - Autonomous Tool Integration Agent (Phase 4)")
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    parser.add_argument("--dashboard", "-d", action="store_true", help="Generate analytics dashboard")
    parser.add_argument("--settings", "-s", action="store_true", help="Show current settings")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")

    args = parser.parse_args()

    # Handle debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    # Handle quiet mode
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Handle dashboard generation
    if args.dashboard:
        from atia.utils.analytics import save_dashboard_html

        dashboard_path = "data/analytics/dashboard.html"
        save_dashboard_html(dashboard_path)

        print(f"Dashboard saved to {dashboard_path}")
        return

    # Handle settings display
    if args.settings:
        print(print_settings())
        return

    # Handle interactive mode
    if args.interactive:
        await interactive_mode()
    elif args.query:
        # Process a single query
        if args.json:
            # JSON output mode
            session = ATIASession()
            result = await session.process_query(args.query)
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output mode
            await process_query_command(args.query)
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
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            traceback.print_exc()
        sys.exit(1)