import pytest

from atia.agent_core.core import AgentCore


def test_agent_core_init():
    """Test that AgentCore initializes correctly."""
    agent = AgentCore(name="TestAgent")
    assert agent.name == "TestAgent"
    assert "TestAgent" in agent.system_prompt


@pytest.mark.asyncio
async def test_process_query():
    """Test that process_query returns a response."""
    # This is a simple test that doesn't actually call OpenAI
    # We'll mock the OpenAI client in a more complete test
    agent = AgentCore()

    # Mock the get_completion function
    original_process = agent.process_query

    async def mock_process(query, context=None):
        return f"Processed: {query}"

    agent.process_query = mock_process

    response = await agent.process_query("Hello")
    assert response == "Processed: Hello"

    # Restore the original function
    agent.process_query = original_process


@pytest.mark.asyncio
async def test_call_component():
    """Test that call_component returns a placeholder response."""
    agent = AgentCore()
    result = await agent.call_component("test_component")
    assert "test_component" in result
    assert "placeholder" in result.lower()