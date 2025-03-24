#!/usr/bin/env python

"""
Demo script for Phase 1: Week 2 - Function Builder & API Discovery enhancements.

This script demonstrates the enhanced API Discovery and Function Builder components.
"""

import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure that the needed directories exist
os.makedirs("data/tools", exist_ok=True)
os.makedirs("data/functions", exist_ok=True)
os.makedirs("data/credentials", exist_ok=True)

async def api_discovery_demo():
    """Demonstrate the enhanced API Discovery component."""
    from atia.api_discovery.discovery import APIDiscovery
    from atia.api_discovery.updates import enhanced_api_search

    # Create API Discovery instance
    discovery = APIDiscovery()

    # Apply the enhanced_api_search method
    discovery.enhanced_api_search = enhanced_api_search.__get__(discovery, APIDiscovery)

    # Test capabilities
    capabilities = [
        "Get current weather for a city",
        "Translate text from English to French",
        "Find nearby restaurants"
    ]

    for capability in capabilities:
        print(f"\n📋 Testing API discovery for: {capability}")
        try:
            # Search for APIs with enhanced search
            apis = await discovery.enhanced_api_search(capability)

            # Display results
            if apis:
                print(f"✅ Found {len(apis)} potential APIs:")
                for i, api in enumerate(apis[:3], 1):
                    print(f"  {i}. {api.name} by {api.provider}")
                    print(f"     Score: {api.relevance_score:.2f}")
                    print(f"     Description: {api.description[:100]}...")
                    print(f"     Documentation: {api.documentation_url}")
                    if hasattr(api, 'capabilities') and api.capabilities:
                        print(f"     Capabilities: {', '.join(api.capabilities[:3])}")
                    print()
            else:
                print("❌ No APIs found")
        except Exception as e:
            print(f"❌ Error in API discovery: {e}")

async def function_builder_demo():
    """Demonstrate the enhanced Function Builder component."""
    from atia.function_builder.builder import FunctionBuilder
    from atia.function_builder.updates import enhanced_responses_api_tool_schema
    from atia.function_builder.models import (
        FunctionDefinition, ApiType, ParameterType, FunctionParameter
    )
    from atia.doc_processor.processor import APIEndpoint, APIInfo

    # Create Function Builder instance
    builder = FunctionBuilder()

    # Apply the enhanced method
    builder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema.__get__(builder, FunctionBuilder)

    # Create a sample function definition
    function_def = FunctionDefinition(
        name="get_weather",
        description="Get current weather for a location",
        api_source_id="weather_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="location",
                param_type=ParameterType.STRING,
                description="City name or coordinates",
                required=True
            ),
            FunctionParameter(
                name="units",
                param_type=ParameterType.STRING,
                description="Temperature units (metric, imperial)",
                required=False,
                default_value="metric"
            ),
            FunctionParameter(
                name="credentials",
                param_type=ParameterType.OBJECT,
                description="API credentials",
                required=False
            )
        ],
        code="""
async def get_weather(location: str, units: str = "metric", credentials: Dict[str, Any] = None):
    \"\"\"
    Get current weather for a location.

    Args:
        location: City name or coordinates
        units: Temperature units (metric, imperial)
        credentials: API credentials

    Returns:
        Weather data
    \"\"\"
    url = "https://api.weatherapi.com/v1/current.json"
    headers = {"Content-Type": "application/json"}

    # Add API key if provided
    if credentials and "api_key" in credentials:
        headers["X-API-Key"] = credentials["api_key"]

    params = {"q": location, "units": units}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                raise Exception(f"Request failed with status {response.status}: {error_text}")

            result = await response.json()
            return result
""",
        endpoint="/v1/current.json",
        method="GET",
        response_format={},
        tags=["weather", "current", "api_key"]
    )

    print("\n📋 Testing Function Builder enhancement")

    # Generate Responses API tool schema
    try:
        schema = builder.enhanced_responses_api_tool_schema(function_def)

        # Display the schema
        import json
        print("✅ Generated Responses API tool schema:")
        print(json.dumps(schema, indent=2))

        # Show that authentication is properly handled
        if "authentication" in schema:
            print("\n✅ Authentication included in schema:")
            print(json.dumps(schema["authentication"], indent=2))
        else:
            print("\n❌ Authentication not included in schema")

    except Exception as e:
        print(f"❌ Error generating schema: {e}")

async def integration_demo():
    """Demonstrate the integration capabilities."""
    from atia.agent_core.core import AgentCore
    from atia.tool_registry import ToolRegistry
    from atia.utils.integration_helper import discover_and_generate_tool, register_function_as_tool

    print("\n📋 Testing Integration Helper")

    try:
        # Create needed components
        tool_registry = ToolRegistry()
        agent = AgentCore(tool_registry=tool_registry)

        # Install the method on the AgentCore instance
        from atia.utils.integration_helper import discover_and_generate_tool
        agent.discover_and_generate_tool = discover_and_generate_tool.__get__(agent, AgentCore)

        # Demonstrate the tool discovery and generation flow
        capability = "Get weather forecast for a city"
        print(f"🔍 Searching for APIs with capability: {capability}")

        # This would normally call all the components, but we'll simulate for the demo
        print("✅ Integration flow successfully set up")
        print("   In a real environment, this would discover and generate a tool")
        print("   using the enhanced API Discovery and Function Builder components.")

    except Exception as e:
        print(f"❌ Error in integration demo: {e}")

async def main():
    """Run all demos."""
    print("🚀 ATIA Phase 1: Week 2 Demo - Function Builder & API Discovery Enhancements")

    print("\n==== API Discovery Demo ====")
    await api_discovery_demo()

    print("\n==== Function Builder Demo ====")
    await function_builder_demo()

    print("\n==== Integration Demo ====")
    await integration_demo()

    print("\n✨ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())