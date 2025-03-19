"""
Test error handling in the Responses API implementation.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor
from atia.utils.openai_client import execute_tools_in_run
from atia.function_builder.models import FunctionDefinition


@pytest.mark.asyncio
async def test_invalid_tool_name_handling():
    """Test handling of invalid tool names."""
    # Create mock tool registry
    tool_registry = MagicMock(spec=ToolRegistry)
    tool_registry.get_tool_by_name = AsyncMock(return_value=None)

    # Create tool executor
    tool_executor = ToolExecutor(tool_registry)

    # Create an invalid tool call
    tool_call = {
        "id": "call_123",
        "function": {
            "name": "nonexistent_function",
            "arguments": '{"param1": "value1"}'
        }
    }

    # Execute the tool
    result = await tool_executor.execute_tool(tool_call)

    # Verify error handling
    assert result["tool_call_id"] == "call_123"
    assert "output" in result
    parsed_output = json.loads(result["output"])
    assert "error" in parsed_output
    assert "not found" in parsed_output["error"]


@pytest.mark.asyncio
async def test_invalid_arguments_handling():
    """Test handling of invalid JSON arguments."""
    # Create mock tool registry
    tool_registry = MagicMock(spec=ToolRegistry)

    # Create tool executor
    tool_executor = ToolExecutor(tool_registry)

    # Create a tool call with invalid JSON
    tool_call = {
        "id": "call_123",
        "function": {
            "name": "some_function",
            "arguments": '{invalid json}'
        }
    }

    # Execute the tool
    result = await tool_executor.execute_tool(tool_call)

    # Verify error handling
    assert result["tool_call_id"] == "call_123"
    assert "output" in result
    parsed_output = json.loads(result["output"])
    assert "error" in parsed_output
    assert "Invalid JSON" in parsed_output["error"]


@pytest.mark.asyncio
async def test_missing_required_parameter():
    """Test handling of missing required parameters."""
    # Create a mock function definition
    function_def = MagicMock(spec=FunctionDefinition)
    function_def.name = "test_function"
    function_def.code = """
async def test_function(required_param: str):
    return {"result": required_param}
"""

    # Create mock tool registry
    mock_tool = MagicMock()
    mock_tool.id = "tool_123"
    mock_tool.name = "test_function"
    mock_tool.function_id = "function_123"

    tool_registry = MagicMock(spec=ToolRegistry)
    tool_registry.get_tool_by_name = AsyncMock(return_value=mock_tool)
    tool_registry._get_function_definition = AsyncMock(return_value=function_def)
    tool_registry.increment_usage = AsyncMock(return_value=True)

    # Create tool executor
    tool_executor = ToolExecutor(tool_registry)

    # Create a tool call missing a required parameter
    tool_call = {
        "id": "call_123",
        "function": {
            "name": "test_function",
            "arguments": '{}'  # Missing required_param
        }
    }

    # Execute the tool with mocked exec
    with patch("atia.tool_executor.executor.exec", side_effect=exec):
        result = await tool_executor.execute_tool(tool_call)

    # Verify error handling
    assert result["tool_call_id"] == "call_123"
    assert "output" in result
    parsed_output = json.loads(result["output"])
    assert "error" in parsed_output
    assert "required_param" in parsed_output["error"] or "Required argument" in parsed_output["error"]


@pytest.mark.asyncio
async def test_api_error_handling():
    """Test handling of API errors during tool execution."""
    # Mock a run that fails
    mock_failed_run = MagicMock()
    mock_failed_run.status = "failed"
    mock_failed_run.error = {"message": "API rate limit exceeded"}

    # Mock the client
    with patch("atia.utils.openai_client.client") as mock_client:
        mock_client.beta.threads.runs.retrieve.return_value = mock_failed_run

        # Mock tool executor
        tool_executor = MagicMock()

        # Try to execute tools
        result = await execute_tools_in_run(
            thread_id="thread_123",
            run_id="run_123",
            tool_executor=tool_executor
        )

        # Verify error handling
        assert "error" in result
        assert "failed" in result["error"]