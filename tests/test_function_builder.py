"""
Tests for the Function Builder component.
"""

import pytest
from unittest.mock import MagicMock

from atia.function_builder import (
    FunctionBuilder, FunctionDefinition, ApiType, 
    ParameterType, FunctionParameter
)
from atia.doc_processor.processor import APIEndpoint, APIInfo


@pytest.fixture
def function_builder():
    """Create a FunctionBuilder instance for testing."""
    return FunctionBuilder()


@pytest.fixture
def sample_api_info():
    """Create a sample APIInfo for testing."""
    return APIInfo(
        base_url="https://api.example.com",
        endpoints=[],
        auth_methods=[],
        description="Example API for testing"
    )


@pytest.fixture
def sample_endpoint():
    """Create a sample APIEndpoint for testing."""
    return APIEndpoint(
        path="/users",
        method="GET",
        description="Get a list of users",
        parameters=[
            {
                "name": "limit",
                "parameter_type": "integer",
                "required": False,
                "description": "Maximum number of users to return",
                "default": 10
            },
            {
                "name": "offset",
                "parameter_type": "integer",
                "required": False,
                "description": "Number of users to skip",
                "default": 0
            }
        ],
        response_format={"type": "json"}
    )


@pytest.mark.asyncio
async def test_generate_function(function_builder, sample_api_info, sample_endpoint):
    """Test generating a function from an API endpoint."""
    function_def = await function_builder.generate_function(sample_api_info, sample_endpoint)

    assert isinstance(function_def, FunctionDefinition)
    assert "example_users" in function_def.name
    assert function_def.description == "Get a list of users"
    assert function_def.api_type == ApiType.REST
    assert len(function_def.parameters) == 2
    assert function_def.method == "GET"
    assert "aiohttp.ClientSession" in function_def.code
    assert len(function_def.test_cases) > 0


@pytest.mark.asyncio
async def test_validate_function(function_builder, sample_api_info, sample_endpoint):
    """Test validating a function definition."""
    function_def = await function_builder.generate_function(sample_api_info, sample_endpoint)

    is_valid, messages = await function_builder.validate_function(function_def)
    assert is_valid is True
    assert len(messages) == 0

    # Test invalid function
    invalid_function = FunctionDefinition(
        name="",
        description="",
        api_source_id="test",
        code="this is not valid python code ->><-",
        endpoint="/test",
        method="GET"
    )

    is_valid, messages = await function_builder.validate_function(invalid_function)
    assert is_valid is False
    assert len(messages) > 0


def test_generate_function_name(function_builder, sample_endpoint):
    """Test generating a function name."""
    base_url = "https://api.example.com"
    function_name = function_builder._generate_function_name(base_url, sample_endpoint)

    assert "example" in function_name
    assert "users" in function_name


def test_generate_parameters(function_builder, sample_endpoint):
    """Test generating function parameters."""
    parameters = function_builder._generate_parameters(sample_endpoint)

    assert len(parameters) == 2
    assert parameters[0].name == "limit"
    assert parameters[0].param_type == ParameterType.INTEGER
    assert parameters[0].required is False


def test_generate_tests(function_builder, sample_api_info, sample_endpoint):
    """Test generating tests for a function."""
    function_def = MagicMock()
    function_def.name = "example_users"
    function_def.method = "GET"
    function_def.parameters = [
        FunctionParameter(
            name="limit",
            param_type=ParameterType.INTEGER,
            description="Maximum number of users to return",
            required=False
        )
    ]

    test_code = function_builder.generate_tests(function_def)

    assert "test_example_users_success" in test_code
    assert "test_example_users_error" in test_code
    assert "limit = 1" in test_code
    assert "@pytest.mark.asyncio" in test_code