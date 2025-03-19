#!/usr/bin/env python

"""
Test script for ATIA Phase 3 Responses API implementation.
"""

import asyncio
import json
import os
from typing import Dict, Any

from atia.agent_core import AgentCore
from atia.tool_registry import ToolRegistry
from atia.function_builder import FunctionBuilder, FunctionDefinition
from atia.function_builder.models import ApiType, ParameterType, FunctionParameter
from atia.tool_executor import ToolExecutor
from atia.config import settings


async def register_test_function(tool_registry):
    """Register a test function in the tool registry."""
    # Create a function builder
    function_builder = FunctionBuilder()

    # Create a simple echo function
    function_def = FunctionDefinition(
        name="echo_function",
        description="Echo back the input message",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="message",
                param_type=ParameterType.STRING,
                description="The message to echo back",
                required=True
            )
        ],
        code="""
async def echo_function(message: str):
    \"\"\"
    Echo back the input message.

    Args:
        message: The message to echo back

    Returns:
        Dict with the echoed message
    \"\"\"
    return {"message": message, "status": "success"}
""",
        endpoint="/echo",
        method="GET",
        tags=["test", "echo"]
    )

    # Register the function
    tool = await tool_registry.register_function(function_def)
    print(f"✅ Registered tool: {tool.name} (ID: {tool.id})")
    return tool


async def test_responses_api_flow():
    """Test the complete Responses API flow."""
    print("\n🧪 Testing Responses API Flow")
    print("============================\n")

    # Create the tool registry
    tool_registry = ToolRegistry()

    # Register a test function
    tool = await register_test_function(tool_registry)

    # Create a tool executor
    tool_executor = ToolExecutor(tool_registry)

    # Create an agent with the tool registry
    agent = AgentCore(tool_registry=tool_registry)
    agent.tool_executor = tool_executor

    # Get the tool schema for Responses API
    function_builder = FunctionBuilder()
    function_def = await tool_registry._get_function_definition(tool.function_id)
    tool_schema = function_builder.generate_responses_api_tool_schema(function_def)

    print(f"Generated tool schema: {json.dumps(tool_schema, indent=2)}")

    # Test a query that should use the echo function
    query = "Can you echo the following message: 'Hello, ATIA Phase 3!'"

    print(f"\nProcessing query: {query}")
    try:
        # Process the query with Responses API
        response = await agent.process_query_with_responses_api(
            query=query,
            tools=[tool_schema]
        )

        print(f"\nResponse from Responses API: {json.dumps(response, indent=2)}")

        # Try to increment the usage counter
        usage_updated = await tool_registry.increment_usage(tool.id)
        print(f"Tool usage incremented: {usage_updated}")

        # Get the updated tool
        updated_tool = await tool_registry.get_tool(tool.id)
        print(f"Tool usage count: {updated_tool.usage_count}")

        return response
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        return None


async def test_backward_compatibility():
    """Test backward compatibility with previous phases."""
    print("\n🧪 Testing Backward Compatibility")
    print("===============================\n")

    # Create an agent without tool registry (simulating Phase 1-2)
    agent = AgentCore()

    query = "What's the weather like in Paris?"

    print(f"Processing query without Responses API: {query}")
    try:
        # Process the query with standard API
        response = await agent.process_query(query)

        print(f"\nResponse from standard API: {response[:200]}...")

        return response
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        return None


async def main():
    """Main test function."""
    print("🔍 ATIA Phase 3 Testing Tool")
    print("==========================\n")

    # Check if we have an OpenAI API key
    if not settings.openai_api_key:
        print("❌ No OpenAI API key found. Please set OPENAI_API_KEY in .env file.")
        return

    # Check Responses API settings
    print("Responses API Settings:")
    print(f"- Enabled: {settings.openai_responses_enabled}")
    print(f"- Assistant ID: {settings.openai_assistant_id or 'Not set'}")

    # Test backward compatibility
    await test_backward_compatibility()

    # Test Responses API
    await test_responses_api_flow()

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())