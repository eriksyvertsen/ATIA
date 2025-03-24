"""
Unit tests for Phase 1: Week 2 - Function Builder & API Discovery enhancements.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_api_discovery_responses_integration():
    """Test the API Discovery with Responses API integration."""
    from atia.api_discovery.responses_integration import search_apis_with_responses_api

    # Mock the get_completion_with_responses_api function
    with patch('atia.api_discovery.responses_integration.get_completion_with_responses_api', new_callable=AsyncMock) as mock_completion:
        # Set up the mock to return a response that requires action
        mock_completion.return_value = {
            "status": "requires_action",
            "required_action": {
                "submit_tool_outputs": {
                    "tool_calls": [
                        {
                            "id": "test_id",
                            "function": {
                                "name": "search_apis",
                                "arguments": json.dumps({
                                    "capability": "weather forecasting",
                                    "max_results": 3
                                })
                            }
                        }
                    ]
                }
            }
        }

        # Call the function
        result = await search_apis_with_responses_api("weather forecasting", 3)

        # Verify the result
        assert isinstance(result, list)
        assert len(result) > 0
        assert "name" in result[0]
        assert "provider" in result[0]
        assert "description" in result[0]
        assert "documentation_url" in result[0]

        # Verify the mock was called with expected arguments
        mock_completion.assert_called_once()
        args, kwargs = mock_completion.call_args
        assert "weather forecasting" in kwargs.get("prompt", "")
        assert kwargs.get("tools", [])


@pytest.mark.asyncio
async def test_enhanced_api_search():
    """Test the enhanced API search method."""
    from atia.api_discovery.discovery import APIDiscovery, APICandidate
    from atia.api_discovery.responses_integration import search_apis_with_responses_api
    import atia.api_discovery.discovery as api_discovery_module

    # Create a mock APIDiscovery instance
    discovery = APIDiscovery()

    # Mock the search_with_responses_api method
    with patch('atia.api_discovery.responses_integration.search_apis_with_responses_api', new_callable=AsyncMock) as mock_search:
        # Set up the mock to return test results
        mock_search.return_value = [
            {
                "name": "Test API",
                "provider": "Test Provider",
                "description": "Test description",
                "documentation_url": "https://test.com",
                "auth_type": "api_key",
                "relevance_score": 0.9
            }
        ]

        # Mock the enhanced_api_search method onto the APIDiscovery instance
        from atia.api_discovery.updates import enhanced_api_search, search_with_responses_api
        discovery.enhanced_api_search = enhanced_api_search.__get__(discovery, APIDiscovery)
        discovery.search_with_responses_api = search_with_responses_api.__get__(discovery, APIDiscovery)

        # Also mock the evaluate_candidates_with_responses_api method
        discovery.evaluate_candidates_with_responses_api = AsyncMock(side_effect=lambda candidates, *args: candidates)

        # Call the enhanced_api_search method
        result = await discovery.enhanced_api_search("weather forecast", 3)

        # Verify the result
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], APICandidate)
        assert result[0].name == "Test API"
        assert result[0].provider == "Test Provider"

        # Verify the mock was called with expected arguments
        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        assert kwargs.get("capability") == "weather forecast"


@pytest.mark.asyncio
async def test_function_builder_enhanced_schema():
    """Test the enhanced Responses API tool schema generation."""
    from atia.function_builder.builder import FunctionBuilder
    from atia.function_builder.models import FunctionDefinition, ParameterType, FunctionParameter, ApiType

    # Create a mock function definition
    function_def = FunctionDefinition(
        name="test_function",
        description="Test function",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="param1",
                param_type=ParameterType.STRING,
                description="Test parameter",
                required=True
            ),
            FunctionParameter(
                name="credentials",
                param_type=ParameterType.OBJECT,
                description="Authentication credentials",
                required=False
            )
        ],
        code="# Test code",
        endpoint="/test",
        method="GET",
        tags=["test", "api_key"]
    )

    # Create a FunctionBuilder instance
    builder = FunctionBuilder()

    # Import and apply the enhanced_responses_api_tool_schema method
    from atia.function_builder.updates import enhanced_responses_api_tool_schema
    builder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema.__get__(builder, FunctionBuilder)

    # Generate the schema
    schema = builder.enhanced_responses_api_tool_schema(function_def)

    # Verify the schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_function"
    assert schema["function"]["description"] == "Test function"
    assert "credentials" not in schema["function"]["parameters"]["properties"]
    assert "param1" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["param1"]["type"] == "string"
    assert "required" in schema["function"]["parameters"]
    assert "param1" in schema["function"]["parameters"]["required"]
    assert "authentication" in schema


@pytest.mark.asyncio
async def test_integration_helper():
    """Test the integration helper module."""
    from atia.agent_core.core import AgentCore
    from atia.tool_registry import ToolRegistry

    # Create mock components
    agent_core = MagicMock()
    agent_core.tool_registry = MagicMock()
    agent_core.api_discovery = MagicMock()
    agent_core.doc_processor = MagicMock()
    agent_core.function_builder = MagicMock()
    agent_core.dynamic_tool_discovery_schema = {"type": "function", "function": {"name": "dynamic_tool_discovery"}}

    # Mock search_by_capability to return empty list
    agent_core.tool_registry.search_by_capability = AsyncMock(return_value=[])

    # Mock get_tools_for_responses_api
    agent_core.tool_registry.get_tools_for_responses_api = AsyncMock(return_value=[])

    # Import the integration helper
    from atia.utils.integration_helper import prepare_tools_for_responses_api

    # Call the function
    tools = await prepare_tools_for_responses_api(agent_core)

    # Verify the result
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "dynamic_tool_discovery"

    # Verify the mock was called
    agent_core.tool_registry.get_tools_for_responses_api.assert_called_once()


def test_create_test_instance():
    """Create test instances of the enhanced components."""
    # Create API Discovery instance
    from atia.api_discovery.discovery import APIDiscovery
    discovery = APIDiscovery()

    # Create Function Builder instance
    from atia.function_builder.builder import FunctionBuilder
    builder = FunctionBuilder()

    # This test just verifies that we can create instances without errors
    assert isinstance(discovery, APIDiscovery)
    assert isinstance(builder, FunctionBuilder)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
