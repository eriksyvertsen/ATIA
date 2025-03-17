"""
Tests for the Tool Registry component.
"""

import pytest
from unittest.mock import patch, MagicMock

from atia.tool_registry import ToolRegistry, ToolRegistration
from atia.function_builder.models import (
    FunctionDefinition, ApiType, ParameterType, FunctionParameter
)


@pytest.fixture
def tool_registry():
    """Create a ToolRegistry instance for testing."""
    with patch('atia.tool_registry.registry.get_embedding', return_value=[0.1, 0.2, 0.3]):
        registry = ToolRegistry()
        # Mock Pinecone initialization to avoid actual API calls
        registry._pinecone_initialized = False
        return registry


@pytest.fixture
def sample_function_def():
    """Create a sample FunctionDefinition for testing."""
    return FunctionDefinition(
        name="test_function",
        description="A test function",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="test_param",
                param_type=ParameterType.STRING,
                description="A test parameter",
                required=True
            )
        ],
        code="async def test_function(test_param: str): return test_param",
        endpoint="/test",
        method="GET",
        tags=["test", "example"]
    )


@pytest.mark.asyncio
async def test_register_function(tool_registry, sample_function_def):
    """Test registering a function in the tool registry."""
    tool = await tool_registry.register_function(sample_function_def)

    assert isinstance(tool, ToolRegistration)
    assert tool.name == "test_function"
    assert tool.description == "A test function"
    assert tool.function_id == sample_function_def.id
    assert tool.api_source_id == "test_api"
    assert len(tool.capability_tags) == 2
    assert tool.usage_count == 0


@pytest.mark.asyncio
async def test_get_tool(tool_registry, sample_function_def):
    """Test retrieving a tool from the registry."""
    tool = await tool_registry.register_function(sample_function_def)

    retrieved_tool = await tool_registry.get_tool(tool.id)

    assert retrieved_tool is not None
    assert retrieved_tool.id == tool.id
    assert retrieved_tool.name == "test_function"


@pytest.mark.asyncio
async def test_increment_usage(tool_registry, sample_function_def):
    """Test incrementing the usage count for a tool."""
    tool = await tool_registry.register_function(sample_function_def)

    # Initial usage count should be 0
    assert tool.usage_count == 0

    # Increment the usage count
    result = await tool_registry.increment_usage(tool.id)

    assert result is True

    # Get the updated tool
    updated_tool = await tool_registry.get_tool(tool.id)

    assert updated_tool.usage_count == 1
    assert updated_tool.last_used is not None


@pytest.mark.asyncio
async def test_search_by_capability(tool_registry, sample_function_def):
    """Test searching for tools by capability."""
    with patch('atia.tool_registry.registry.get_embedding', return_value=[0.1, 0.2, 0.3]):
        tool = await tool_registry.register_function(sample_function_def)

        # Search for tools with matching capability
        matching_tools = await tool_registry.search_by_capability("test example")

        assert len(matching_tools) == 1
        assert matching_tools[0].id == tool.id

        # Search for tools with non-matching capability
        non_matching_tools = await tool_registry.search_by_capability("non-matching")

        assert len(non_matching_tools) == 0


@pytest.mark.asyncio
async def test_generate_embedding(tool_registry, sample_function_def):
    """Test generating an embedding for a function definition."""
    with patch('atia.tool_registry.registry.get_embedding', return_value=[0.1, 0.2, 0.3]) as mock_get_embedding:
        embedding = await tool_registry._generate_embedding(sample_function_def)

        assert embedding == [0.1, 0.2, 0.3]
        mock_get_embedding.assert_called_once()