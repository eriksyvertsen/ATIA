"""
Tests for the OpenAI Responses API integration.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from atia.agent_core import AgentCore
from atia.utils.openai_client import get_completion_with_responses_api, execute_tools_in_run
from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor
from atia.function_builder import FunctionBuilder, FunctionDefinition
from atia.function_builder.models import ApiType, ParameterType, FunctionParameter


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing Responses API."""
    with patch("atia.utils.openai_client.client") as mock_client:
        # Mock thread creation
        mock_thread = MagicMock()
        mock_thread.id = "thread_123"
        mock_client.beta.threads.create.return_value = mock_thread

        # Mock message creation
        mock_message = MagicMock()
        mock_client.beta.threads.messages.create.return_value = mock_message

        # Mock run creation
        mock_run = MagicMock()
        mock_run.id = "run_123"
        mock_client.beta.threads.runs.create.return_value = mock_run

        # Mock run retrieval
        mock_completed_run = MagicMock()
        mock_completed_run.status = "completed"
        mock_client.beta.threads.runs.retrieve.return_value = mock_completed_run

        # Mock message listing
        mock_messages = MagicMock()
        mock_assistant_message = MagicMock()
        mock_assistant_message.role = "assistant"
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = MagicMock()
        mock_content.text.value = "This is a test response"
        mock_assistant_message.content = [mock_content]
        mock_messages.data = [mock_assistant_message]
        mock_client.beta.threads.messages.list.return_value = mock_messages

        yield mock_client


@pytest.fixture
def mock_tool_registry():
    """Create a mock ToolRegistry."""
    return MagicMock(spec=ToolRegistry)


@pytest.fixture
def mock_tool_executor():
    """Create a mock ToolExecutor."""
    mock_executor = MagicMock(spec=ToolExecutor)
    mock_executor.execute_tool = AsyncMock(return_value={
        "tool_call_id": "call_123",
        "output": '{"result": "Test result"}'
    })
    return mock_executor


@pytest.mark.asyncio
async def test_get_completion_with_responses_api(mock_openai_client):
    """Test getting a completion with the Responses API."""
    with patch("atia.utils.openai_client._wait_for_run_completion", new_callable=AsyncMock) as mock_wait:
        mock_wait.return_value = {
            "content": "This is a test response",
            "thread_id": "thread_123",
            "run_id": "run_123"
        }

        response = await get_completion_with_responses_api(
            prompt="Test prompt",
            system_message="Test system message"
        )

        assert "content" in response
        assert response["content"] == "This is a test response"
        assert "thread_id" in response
        assert "run_id" in response


@pytest.mark.asyncio
async def test_execute_tools_in_run(mock_openai_client, mock_tool_executor):
    """Test executing tools in a run."""
    # Mock run that requires action first, then completes
    mock_requires_action_run = MagicMock()
    mock_requires_action_run.status = "requires_action"
    mock_requires_action_run.required_action = MagicMock()
    mock_requires_action_run.required_action.submit_tool_outputs = MagicMock()

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = "test_function"
    mock_tool_call.function.arguments = '{"param1": "test"}'

    mock_requires_action_run.required_action.submit_tool_outputs.tool_calls = [mock_tool_call]

    mock_completed_run = MagicMock()
    mock_completed_run.status = "completed"

    # Return requires_action run on first call, completed run on second call
    mock_openai_client.beta.threads.runs.retrieve.side_effect = [
        mock_requires_action_run,
        mock_completed_run
    ]

    # Mock submitting tool outputs
    mock_openai_client.beta.threads.runs.submit_tool_outputs = MagicMock()

    # Mock message listing
    mock_messages = MagicMock()
    mock_assistant_message = MagicMock()
    mock_assistant_message.role = "assistant"
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = MagicMock()
    mock_content.text.value = "Final response after tool execution"
    mock_assistant_message.content = [mock_content]
    mock_messages.data = [mock_assistant_message]
    mock_openai_client.beta.threads.messages.list.return_value = mock_messages

    result = await execute_tools_in_run(
        thread_id="thread_123",
        run_id="run_123",
        tool_executor=mock_tool_executor
    )

    assert "content" in result
    assert result["content"] == "Final response after tool execution"
    mock_tool_executor.execute_tool.assert_called_once()
    mock_openai_client.beta.threads.runs.submit_tool_outputs.assert_called_once()


@pytest.mark.asyncio
async def test_function_builder_generate_responses_api_tool_schema():
    """Test generating a Responses API tool schema from a function definition."""
    # Create a function builder
    function_builder = FunctionBuilder()

    # Create a function definition
    function_def = FunctionDefinition(
        name="test_function",
        description="A test function",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="param1",
                param_type=ParameterType.STRING,
                description="A test parameter",
                required=True
            ),
            FunctionParameter(
                name="param2",
                param_type=ParameterType.INTEGER,
                description="Another test parameter",
                required=False,
                default_value=42
            )
        ],
        code="async def test_function(param1: str, param2: int = 42): return {'param1': param1, 'param2': param2}",
        endpoint="/test",
        method="GET"
    )

    # Generate a tool schema
    tool_schema = function_builder.generate_responses_api_tool_schema(function_def)

    # Verify the schema
    assert tool_schema["type"] == "function"
    assert tool_schema["function"]["name"] == "test_function"
    assert tool_schema["function"]["description"] == "A test function"
    assert "parameters" in tool_schema["function"]
    assert tool_schema["function"]["parameters"]["type"] == "object"
    assert "param1" in tool_schema["function"]["parameters"]["properties"]
    assert tool_schema["function"]["parameters"]["properties"]["param1"]["type"] == "string"
    assert "param2" in tool_schema["function"]["parameters"]["properties"]
    assert tool_schema["function"]["parameters"]["properties"]["param2"]["type"] == "integer"
    assert tool_schema["function"]["parameters"]["properties"]["param2"]["default"] == 42
    assert "required" in tool_schema["function"]["parameters"]
    assert "param1" in tool_schema["function"]["parameters"]["required"]
    assert "param2" not in tool_schema["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_tool_executor(mock_tool_registry):
    """Test the ToolExecutor."""
    from atia.tool_executor.executor import ToolExecutor

    # Set up the mock tool registry
    mock_tool = MagicMock()
    mock_tool.id = "tool_123"
    mock_tool.name = "test_function"
    mock_tool.function_id = "function_123"

    mock_tool_registry.get_tool_by_name = AsyncMock(return_value=mock_tool)

    # Create a function definition
    function_def = FunctionDefinition(
        id="function_123",
        name="test_function",
        description="A test function",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="param1",
                param_type=ParameterType.STRING,
                description="A test parameter",
                required=True
            )
        ],
        code="async def test_function(param1: str): return {'result': param1}",
        endpoint="/test",
        method="GET"
    )

    mock_tool_registry._get_function_definition = AsyncMock(return_value=function_def)
    mock_tool_registry.increment_usage = AsyncMock(return_value=True)

    # Create the tool executor
    tool_executor = ToolExecutor(mock_tool_registry)

    # Create a tool call
    tool_call = {
        "id": "call_123",
        "function": {
            "name": "test_function",
            "arguments": '{"param1": "test value"}'
        }
    }

    # Execute the tool
    with patch("atia.tool_executor.executor.exec", wraps=exec) as mock_exec:
        result = await tool_executor.execute_tool(tool_call)

    # Verify the result
    assert result["tool_call_id"] == "call_123"
    assert "output" in result
    assert '"result": "test value"' in result["output"]

    # Verify the mock calls
    mock_tool_registry.get_tool_by_name.assert_called_once_with("test_function")
    mock_tool_registry._get_function_definition.assert_called_once_with("function_123")
    mock_tool_registry.increment_usage.assert_called_once_with("tool_123")
    mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_agent_core_with_responses_api(mock_tool_registry, mock_tool_executor):
    """Test Agent Core with Responses API."""
    # Set up the mock tool registry
    mock_tool_registry.get_tools_for_responses_api = AsyncMock(return_value=[])

    # Patch the Responses API functions
    with patch("atia.agent_core.core.get_completion_with_responses_api", new_callable=AsyncMock) as mock_get_completion:
        mock_get_completion.return_value = {
            "content": "Test response",
            "thread_id": "thread_123",
            "run_id": "run_123"
        }

        # Patch the execute_tools_in_run function
        with patch("atia.agent_core.core.execute_tools_in_run", new_callable=AsyncMock) as mock_execute_tools:
            mock_execute_tools.return_value = {
                "content": "Final response after tool execution",
                "thread_id": "thread_123",
                "run_id": "run_123"
            }

            # Create the Agent Core
            agent = AgentCore(tool_registry=mock_tool_registry)
            agent.tool_executor = mock_tool_executor

            # Process a query with the Responses API
            response = await agent.process_query_with_responses_api(
                query="Test query",
                tools=[]
            )

            # Check the result
            assert "content" in response
            assert response["content"] == "Test response"

            # Test with a response that requires tool execution
            mock_get_completion.return_value = {
                "status": "requires_action",
                "thread_id": "thread_123",
                "run_id": "run_123"
            }

            response = await agent.process_query_with_responses_api(
                query="Test query that requires tools",
                tools=[]
            )

            # Check the result
            assert "content" in response
            assert response["content"] == "Final response after tool execution"

            # Verify the expected functions were called
            mock_get_completion.assert_called()
            mock_execute_tools.assert_called_once()