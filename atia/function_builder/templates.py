"""
Code generation templates for different API types.
"""

# Template for REST GET requests
REST_GET_TEMPLATE = """
async def {function_name}({parameters}):
    \"\"\"
    {description}

    Args:
    {parameter_descriptions}

    Returns:
    {return_description}

    Raises:
    APIError: If the API request fails
    \"\"\"
    url = "{full_url}"
    headers = {{
        "Authorization": "Bearer $AUTH_TOKEN",
        "Content-Type": "application/json"
    }}
    params = {params_dict}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                raise APIError(f"Request failed with status {{response.status}}: {{error_text}}")

            result = await response.json()
            return result
"""

# Template for REST POST requests
REST_POST_TEMPLATE = """
async def {function_name}({parameters}):
    \"\"\"
    {description}

    Args:
    {parameter_descriptions}

    Returns:
    {return_description}

    Raises:
    APIError: If the API request fails
    \"\"\"
    url = "{full_url}"
    headers = {{
        "Authorization": "Bearer $AUTH_TOKEN",
        "Content-Type": "application/json"
    }}
    payload = {params_dict}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                raise APIError(f"Request failed with status {{response.status}}: {{error_text}}")

            result = await response.json()
            return result
"""

# Template for GraphQL queries
GRAPHQL_TEMPLATE = """
async def {function_name}({parameters}):
    \"\"\"
    {description}

    Args:
    {parameter_descriptions}

    Returns:
    {return_description}

    Raises:
    APIError: If the API request fails
    \"\"\"
    url = "{full_url}"
    headers = {{
        "Authorization": "Bearer $AUTH_TOKEN",
        "Content-Type": "application/json"
    }}

    query = \"\"\"
    query {{
        {method} {{
            # Fields would be replaced in actual implementation
            id
            name
        }}
    }}
    \"\"\"

    variables = {params_dict}

    payload = {{
        "query": query,
        "variables": variables
    }}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                raise APIError(f"Request failed with status {{response.status}}: {{error_text}}")

            result = await response.json()
            return result
"""

# Generic template for other API types
GENERIC_API_TEMPLATE = """
async def {function_name}({parameters}):
    \"\"\"
    {description}

    Args:
    {parameter_descriptions}

    Returns:
    {return_description}

    Raises:
    APIError: If the API request fails
    \"\"\"
    url = "{full_url}"
    headers = {{
        "Authorization": "Bearer $AUTH_TOKEN",
        "Content-Type": "application/json"
    }}

    # This function needs to be customized based on the specific API requirements
    # The code below is a placeholder

    payload = {params_dict}

    async with aiohttp.ClientSession() as session:
        async with session.{method.lower()}(url, headers=headers, json=payload) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                raise APIError(f"Request failed with status {{response.status}}: {{error_text}}")

            result = await response.json()
            return result
"""

# Template for function test generation
FUNCTION_TEST_TEMPLATE = """
import pytest
import aiohttp
from unittest.mock import patch, MagicMock

from atia.functions import {function_name}


@pytest.mark.asyncio
async def test_{function_name}_success():
    \"\"\"Test successful API call to {function_name}.\"\"\"
    # Mock response data
    mock_response = {{"result": "success", "data": {{"id": "test123"}}}}

    # Create mock for the aiohttp ClientSession
    mock_session = MagicMock()
    mock_response_obj = MagicMock()
    mock_response_obj.status = 200
    mock_response_obj.json = MagicMock(return_value=mock_response)
    mock_response_obj.__aenter__.return_value = mock_response_obj
    mock_session.get.return_value = mock_response_obj
    mock_session.__aenter__.return_value = mock_session

    # Patch the aiohttp.ClientSession
    with patch('aiohttp.ClientSession', return_value=mock_session):
        # Call the function with test parameters
        {test_params}
        result = await {function_name}({param_values})

        # Assertions
        assert result == mock_response
        getattr(mock_session, method.lower()).assert_called_once()


@pytest.mark.asyncio
async def test_{function_name}_error():
    \"\"\"Test error handling in {function_name}.\"\"\"
    # Create mock for the aiohttp ClientSession
    mock_session = MagicMock()
    mock_response_obj = MagicMock()
    mock_response_obj.status = 404
    mock_response_obj.text = MagicMock(return_value="Not Found")
    mock_response_obj.__aenter__.return_value = mock_response_obj
    mock_session.get.return_value = mock_response_obj
    mock_session.__aenter__.return_value = mock_session

    # Patch the aiohttp.ClientSession
    with patch('aiohttp.ClientSession', return_value=mock_session):
        # Call the function with test parameters
        {test_params}

        # Test that it raises an exception
        with pytest.raises(APIError):
            await {function_name}({param_values})
"""