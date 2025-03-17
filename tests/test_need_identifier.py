import pytest
from unittest.mock import patch

from atia.need_identifier.identifier import NeedIdentifier, ToolNeed


def test_need_identifier_init():
    """Test that NeedIdentifier initializes correctly."""
    identifier = NeedIdentifier(threshold=0.8)
    assert identifier.threshold == 0.8
    assert len(identifier.tool_categories) > 0


def test_construct_need_detection_prompt():
    """Test that the need detection prompt is constructed correctly."""
    identifier = NeedIdentifier()
    prompt = identifier._construct_need_detection_prompt("Translate this to French", {})

    assert "Translate this to French" in prompt
    assert "Tool Categories" in prompt
    assert "translation" in prompt
    assert "JSON" in prompt


@pytest.mark.asyncio
@patch("atia.need_identifier.identifier.get_json_completion")
async def test_identify_tool_need_positive(mock_get_json_completion):
    """Test that a tool need is correctly identified."""
    mock_get_json_completion.return_value = {
        "tool_needed": True,
        "category": "translation",
        "description": "Translate text from English to French",
        "confidence": 0.95
    }

    identifier = NeedIdentifier(threshold=0.7)
    result = await identifier.identify_tool_need("Translate this text to French")

    assert isinstance(result, ToolNeed)
    assert result.category == "translation"
    assert result.confidence == 0.95
    assert "French" in result.description


@pytest.mark.asyncio
@patch("atia.need_identifier.identifier.get_json_completion")
async def test_identify_tool_need_negative(mock_get_json_completion):
    """Test that no tool need is identified when confidence is below threshold."""
    mock_get_json_completion.return_value = {
        "tool_needed": True,
        "category": "translation",
        "description": "Translate text from English to French",
        "confidence": 0.6
    }

    identifier = NeedIdentifier(threshold=0.7)
    result = await identifier.identify_tool_need("Translate this text to French")

    assert result is None


@pytest.mark.asyncio
@patch("atia.need_identifier.identifier.get_json_completion")
async def test_identify_tool_need_not_needed(mock_get_json_completion):
    """Test that no tool need is identified when no tool is needed."""
    mock_get_json_completion.return_value = {
        "tool_needed": False,
        "category": "",
        "description": "",
        "confidence": 0.9
    }

    identifier = NeedIdentifier(threshold=0.7)
    result = await identifier.identify_tool_need("What is the capital of France?")

    assert result is None