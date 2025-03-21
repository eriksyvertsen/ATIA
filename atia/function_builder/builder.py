"""
Function Builder implementation for generating API functions.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from atia.function_builder.models import (
    FunctionDefinition, ApiType, ParameterType, FunctionParameter
)
from atia.function_builder.templates import (
    REST_GET_TEMPLATE, REST_POST_TEMPLATE, 
    GRAPHQL_TEMPLATE, GENERIC_API_TEMPLATE,
    FUNCTION_TEST_TEMPLATE
)
from atia.doc_processor.processor import APIEndpoint, APIInfo


logger = logging.getLogger(__name__)


class APIError(Exception):
    """Exception raised for API errors."""
    pass


class FunctionBuilder:
    """
    Generates reusable function definitions based on API documentation.
    """

    def __init__(self):
        """Initialize the Function Builder."""
        self.templates = {
            ApiType.REST: {
                "GET": REST_GET_TEMPLATE,
                "POST": REST_POST_TEMPLATE,
                "PUT": REST_POST_TEMPLATE,  # Reuse POST template for PUT
                "DELETE": REST_GET_TEMPLATE,  # Reuse GET template for DELETE
                "PATCH": REST_POST_TEMPLATE,  # Reuse POST template for PATCH
            },
            ApiType.GRAPHQL: GRAPHQL_TEMPLATE,
            ApiType.GENERIC: GENERIC_API_TEMPLATE
        }
        # Make sure the test template is properly initialized
        self.test_template = FUNCTION_TEST_TEMPLATE

        # Authentication code templates
        self.auth_templates = {
            "api_key": {
                "header": '    headers["{key_name}"] = "{key_prefix}{key_value}"\n',
                "query": '    params["{key_name}"] = "{key_value}"\n',
                "cookie": '    cookies["{key_name}"] = "{key_value}"\n'
            },
            "bearer": '    headers["Authorization"] = "Bearer {token}"\n',
            "basic": '    import base64\n    auth_str = base64.b64encode(f"{username}:{password}".encode()).decode()\n    headers["Authorization"] = f"Basic {auth_str}"\n',
            "oauth": '    headers["Authorization"] = "Bearer {access_token}"\n',
            "none": ""
        }

        # Verify that the template doesn't have any formatting issues
        try:
            # Test the template with sample values to catch any formatting errors
            self.test_template.format(
                function_name="test_func",
                test_params="param = 'value'",
                param_values="param=param",
                method="get"
            )
        except Exception as e:
            # Fall back to a simpler template if there's a formatting issue
            import logging
            logging.warning(f"Error with test template: {e}. Using simplified template.")
            self.test_template = """
import pytest
import aiohttp
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_{function_name}_success():
    \"\"\"Test successful API call.\"\"\"
    # Test parameters
    {test_params}

    # Mock response
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json.return_value = {{"success": True}}

    with patch('aiohttp.ClientSession', return_value=mock_session):
        # Call function with parameters
        result = await {function_name}({param_values})
        assert result is not None
"""

    async def generate_function(self, 
                              api_info: APIInfo, 
                              endpoint: APIEndpoint) -> FunctionDefinition:
        """
        Generate a function for an API endpoint.

        Args:
            api_info: Information about the API
            endpoint: The specific endpoint to generate a function for

        Returns:
            Generated function definition
        """
        # Determine API type
        api_type = ApiType.REST  # Default to REST

        # Generate function name
        function_name = self._generate_function_name(api_info.base_url, endpoint)

        # Generate parameters
        parameters = self._generate_parameters(endpoint)

        # Select template
        template = self._select_template(api_type, endpoint.method)

        # Generate the function code
        code = self._generate_function_code(
            template, 
            function_name,
            api_info.base_url,
            endpoint,
            parameters,
            auth_methods=api_info.auth_methods
        )

        # Generate test cases
        test_cases = self._generate_test_cases(endpoint)

        # Create function definition
        function_def = FunctionDefinition(
            name=function_name,
            description=endpoint.description,
            api_source_id=getattr(api_info, 'source_id', str(hash(api_info.base_url))),
            api_type=api_type,
            parameters=parameters,
            code=code,
            endpoint=endpoint.path,
            method=endpoint.method,
            response_format=endpoint.response_format or {},
            tags=self._generate_tags(api_info, endpoint),
            test_cases=test_cases
        )

        return function_def

    def _generate_function_name(self, base_url: str, endpoint: APIEndpoint) -> str:
        """Generate a function name based on the endpoint."""
        # Extract domain from base_url
        from urllib.parse import urlparse
        netloc = urlparse(base_url).netloc
        domain_parts = netloc.split('.')

        # Use the main domain name part (usually the second part like "example" in "api.example.com")
        if len(domain_parts) >= 2:
            domain = domain_parts[1]  # Extract "example" from "api.example.com"
        else:
            domain = domain_parts[0]

        # Clean up the path to create a function name
        path_parts = [p for p in endpoint.path.split('/') if p and not p.startswith('{')]

        # If there are no usable path parts, use the method and a generic name
        if not path_parts:
            name_parts = [domain, endpoint.method.lower(), "endpoint"]
        else:
            # Use domain and all path parts
            name_parts = [domain] + path_parts

            # If the endpoint is complex with no distinct parts, simplify based on purpose
            if len(name_parts) <= 1 or (len(name_parts) == 2 and name_parts[1] in ['api', 'v1', 'v2']):
                # Try to extract action from description
                action_words = ['get', 'search', 'find', 'create', 'update', 'delete', 'list', 'query']
                for word in action_words:
                    if word in endpoint.description.lower():
                        name_parts.append(word)
                        break

                # If no action found, use method and a descriptor from description
                if len(name_parts) <= 2:
                    description_words = re.findall(r'\b\w+\b', endpoint.description.lower())
                    relevant_words = [w for w in description_words if len(w) > 3 and w not in ['with', 'from', 'this', 'that', 'will', 'have']]
                    if relevant_words:
                        name_parts.append(relevant_words[0])

        # Create snake_case function name
        function_name = '_'.join(name_parts).lower()

        # Replace any remaining special characters with underscores
        function_name = re.sub(r'[^a-z0-9_]', '_', function_name)

        # Remove repeated underscores
        function_name = re.sub(r'_+', '_', function_name)

        # Make sure it doesn't start with a number
        if function_name[0].isdigit():
            function_name = 'f_' + function_name

        # Limit length to 50 characters
        if len(function_name) > 50:
            function_name = function_name[:50]

        # Remove trailing underscores
        function_name = function_name.rstrip('_')

        return function_name

    def _generate_parameters(self, endpoint: APIEndpoint) -> List[FunctionParameter]:
        """Generate function parameters from endpoint parameters."""
        result = []

        # Track param names to avoid duplicates
        param_names = set()

        for param in endpoint.parameters:
            # Handle both dict and object cases
            if isinstance(param, dict):
                name = param.get('name', '')
                param_type = self._map_parameter_type(param.get('parameter_type', 'string'))
                description = param.get('description', '')
                required = param.get('required', False)
                default_value = param.get('default', None)
            else:
                name = getattr(param, 'name', '')
                param_type = self._map_parameter_type(getattr(param, 'parameter_type', 'string'))
                description = getattr(param, 'description', '')
                required = getattr(param, 'required', False)
                default_value = getattr(param, 'default_value', None)

            # Skip empty names or duplicates
            if not name or name in param_names:
                continue

            param_names.add(name)

            # Clean param name - replace special chars with underscore
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

            # Make sure it doesn't start with a number
            if clean_name[0].isdigit():
                clean_name = 'p_' + clean_name

            # Use original name in metadata but clean name for function param
            result.append(FunctionParameter(
                name=clean_name,
                param_type=param_type,
                description=description,
                required=required,
                default_value=default_value,
                metadata={"original_name": name}
            ))

        return result

    def _map_parameter_type(self, param_type: str) -> ParameterType:
        """Map API parameter type to function parameter type."""
        type_map = {
            'string': ParameterType.STRING,
            'integer': ParameterType.INTEGER,
            'number': ParameterType.FLOAT,
            'boolean': ParameterType.BOOLEAN,
            'array': ParameterType.ARRAY,
            'object': ParameterType.OBJECT
        }

        return type_map.get(param_type.lower(), ParameterType.STRING)

    def _select_template(self, api_type: ApiType, method: str) -> str:
        """Select the appropriate template for the API type and method."""
        if api_type == ApiType.REST:
            return self.templates[api_type].get(method.upper(), 
                                               self.templates[api_type].get("GET"))

        return self.templates.get(api_type, self.templates[ApiType.GENERIC])

    def _generate_function_code(self, 
                              template: str, 
                              function_name: str,
                              base_url: str,
                              endpoint: APIEndpoint,
                              parameters: List[FunctionParameter],
                              auth_methods: List[Dict] = None) -> str:
        """Generate the function code using the template."""
        # Generate parameter string for function signature
        param_str = ', '.join([
            f"{p.name}: {p.param_type.value}" + (
                "" if p.required else f" = {json.dumps(p.default_value) if p.default_value is not None else 'None'}"
            ) for p in parameters
        ])

        # Add credentials parameter if auth required
        auth_required = True if auth_methods and len(auth_methods) > 0 else False
        if auth_required:
            if param_str:
                param_str += ", "
            param_str += "credentials: Dict[str, Any] = None"

        # Generate parameter descriptions for docstring
        param_desc = '\n    '.join([f"{p.name}: {p.description}" for p in parameters])
        if auth_required:
            param_desc += "\n    credentials: Authentication credentials for the API"

        if not param_desc:
            param_desc = "No parameters required."

        # Generate authentication code
        auth_code = self._generate_auth_code(auth_methods)

        # Analyze endpoint path for path parameters
        path_params = re.findall(r'{([^}]+)}', endpoint.path)
        path_param_mapping = {}

        # Map function parameters to path parameters
        for path_param in path_params:
            # Find matching function parameter
            for param in parameters:
                if param.name == path_param or (hasattr(param, 'metadata') and param.metadata.get('original_name') == path_param):
                    path_param_mapping[path_param] = param.name
                    break

        # Build final path with path parameters substituted
        final_path = endpoint.path
        for path_param, func_param in path_param_mapping.items():
            final_path = final_path.replace(f"{{{path_param}}}", f"{{{func_param}}}")

        # Full URL with proper joining
        if base_url.endswith('/') and final_path.startswith('/'):
            full_url = f"{base_url}{final_path[1:]}"
        elif not base_url.endswith('/') and not final_path.startswith('/'):
            full_url = f"{base_url}/{final_path}"
        else:
            full_url = f"{base_url}{final_path}"

        # Determine parameter locations (query, header, path, body)
        query_params = []
        header_params = []
        body_params = []

        for param in parameters:
            # Skip path parameters as they're handled separately
            if param.name in path_param_mapping.values():
                continue

            # Look for location in parameter metadata or guess based on method
            location = "query"  # Default location
            if hasattr(param, 'metadata') and param.metadata.get('location'):
                location = param.metadata.get('location').lower()
            elif endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
                location = "body"

            if location == "query":
                query_params.append(param.name)
            elif location == "header":
                header_params.append(param.name)
            elif location in ["body", "json"]:
                body_params.append(param.name)

        # Generate params dictionary for requests
        query_dict = '{'
        if query_params:
            query_dict += ', '.join([f'"{p}": {p}' for p in query_params])
        query_dict += '}'

        # Generate headers dictionary
        headers_dict = '{\n        "Content-Type": "application/json"\n    }'

        # Generate body dictionary
        if endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
            if body_params:
                body_dict = '{'
                body_dict += ', '.join([f'"{p}": {p}' for p in body_params])
                body_dict += '}'
            else:
                # If no explicit body params but it's a body method, use all non-path params
                body_dict = '{'
                body_dict += ', '.join([f'"{p.name}": {p.name}' for p in parameters 
                                      if p.name not in path_param_mapping.values() and p.name not in query_params])
                body_dict += '}'
        else:
            body_dict = '{}'

        # Determine which dictionary to use based on method
        if endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
            params_dict = body_dict
        else:
            params_dict = query_dict

        # Generate error handling code
        error_handling = """
        # Handle error responses
        if response.status < 200 or response.status >= 300:
            try:
                error_json = await response.json()
                error_message = error_json.get("message", str(error_json))
            except:
                error_text = await response.text()
                error_message = f"HTTP {response.status}: {error_text}"

            raise APIError(error_message)
        """

        # Add imports
        imports = """import aiohttp
import json
from typing import Dict, List, Optional, Any

"""

        # Replace template placeholders
        code = template.format(
            function_name=function_name,
            parameters=param_str,
            parameter_descriptions=param_desc,
            base_url=base_url,
            path=final_path,
            method=endpoint.method.upper(),
            params_dict=params_dict,
            full_url=full_url,
            description=endpoint.description,
            return_description="The API response data."
        )

        # Insert auth code and error handling
        code = code.replace("# Auth code will be inserted here", auth_code)

        # Add imports at the top
        code = imports + code

        return code

    def _generate_auth_code(self, auth_methods: List[Dict]) -> str:
        """Generate authentication code based on auth methods."""
        if not auth_methods or len(auth_methods) == 0:
            return "    # No authentication required\n"

        # Determine auth type from first method
        auth_type = auth_methods[0].get("type", "").lower()

        # Generate auth code based on type
        if auth_type == "apikey":
            key_location = auth_methods[0].get("in", "header").lower()
            key_name = auth_methods[0].get("name", "X-API-Key")
            key_prefix = auth_methods[0].get("prefix", "")

            if key_location in self.auth_templates["api_key"]:
                return self.auth_templates["api_key"][key_location].format(
                    key_name=key_name,
                    key_prefix=key_prefix + " " if key_prefix else "",
                    key_value="credentials.get('api_key', '')" if key_prefix else "credentials.get('api_key', '')"
                )
            else:
                return self.auth_templates["api_key"]["header"].format(
                    key_name=key_name,
                    key_prefix=key_prefix + " " if key_prefix else "",
                    key_value="credentials.get('api_key', '')"
                )

        elif auth_type == "oauth2" or auth_type == "oauth":
            return self.auth_templates["oauth"].format(
                access_token="credentials.get('access_token', '')"
            )

        elif auth_type == "http":
            scheme = auth_methods[0].get("scheme", "").lower()
            if scheme == "bearer":
                return self.auth_templates["bearer"].format(
                    token="credentials.get('token', '')"
                )
            elif scheme == "basic":
                return self.auth_templates["basic"].format(
                    username="credentials.get('username', '')",
                    password="credentials.get('password', '')"
                )

        # Default to Bearer token if unsure
        return """    # Add authentication if credentials provided
    if credentials:
        if "api_key" in credentials:
            headers["X-API-Key"] = credentials["api_key"]
        elif "token" in credentials:
            headers["Authorization"] = f"Bearer {credentials['token']}"
        elif "access_token" in credentials:
            headers["Authorization"] = f"Bearer {credentials['access_token']}"
    """

    def _generate_tags(self, api_info: APIInfo, endpoint: APIEndpoint) -> List[str]:
        """Generate capability tags for the function."""
        tags = []

        # Add domain tag if available
        if hasattr(api_info, 'domain'):
            tags.append(api_info.domain)
        else:
            # Extract domain from base URL
            from urllib.parse import urlparse
            netloc = urlparse(api_info.base_url).netloc
            domain_parts = netloc.split('.')
            if len(domain_parts) >= 2:
                tags.append(domain_parts[1])  # e.g. "example" from "api.example.com"

        # Add method tag
        tags.append(endpoint.method.lower())

        # Add tags based on endpoint description
        if endpoint.description:
            # Get key actions and concepts
            actions = ["search", "create", "update", "delete", "list", "get", "find", "query", "check"]
            concepts = ["weather", "translation", "image", "text", "audio", "video", "data", "user", "account", "payment"]

            for action in actions:
                if action in endpoint.description.lower():
                    tags.append(action)

            for concept in concepts:
                if concept in endpoint.description.lower():
                    tags.append(concept)

        # Add tags based on path
        path_parts = [p for p in endpoint.path.split('/') if p and not p.startswith('{')]
        for part in path_parts:
            if part not in ['v1', 'v2', 'v3', 'api']:
                tags.append(part)

        # Add tags based on parameters
        for param in endpoint.parameters:
            param_name = param.get('name', '') if isinstance(param, dict) else getattr(param, 'name', '')
            if param_name and param_name not in ['api_key', 'token']:
                tags.append(f"param:{param_name}")

        # Deduplicate and return
        return list(set(tags))

    def _generate_test_cases(self, endpoint: APIEndpoint) -> List[Dict[str, Any]]:
        """Generate test cases for the function."""
        test_cases = []

        # Generate mock params
        mock_params = {}
        for param in endpoint.parameters:
            if isinstance(param, dict):
                param_type = param.get('parameter_type', 'string')
                param_name = param.get('name', '')
            else:
                param_type = getattr(param, 'parameter_type', 'string')
                param_name = getattr(param, 'name', '')

            if not param_name:
                continue

            if param_type == 'string':
                mock_params[param_name] = f"test_{param_name}"
            elif param_type in ('integer', 'number'):
                mock_params[param_name] = 1
            elif param_type == 'boolean':
                mock_params[param_name] = True
            else:
                mock_params[param_name] = {}

        # Create test cases
        test_cases.append({
            "name": "Basic functionality test",
            "params": mock_params,
            "expected_status": 200
        })

        # Add error test case
        test_cases.append({
            "name": "Error handling test",
            "params": mock_params,
            "expected_status": 404,
            "expected_error": True
        })

        return test_cases

    def generate_tests(self, function_def: FunctionDefinition) -> str:
        """
        Generate test code for a function.

        Args:
            function_def: Function definition to generate tests for

        Returns:
            Test code as string
        """
        # Get the test parameters
        test_params = []
        param_values = []

        for param in function_def.parameters:
            if param.param_type == ParameterType.STRING:
                test_params.append(f"{param.name} = 'test_{param.name}'")
                param_values.append(f"{param.name}={param.name}")
            elif param.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
                test_params.append(f"{param.name} = 1")
                param_values.append(f"{param.name}={param.name}")
            elif param.param_type == ParameterType.BOOLEAN:
                test_params.append(f"{param.name} = True")
                param_values.append(f"{param.name}={param.name}")
            elif param.param_type == ParameterType.ARRAY:
                test_params.append(f"{param.name} = []")
                param_values.append(f"{param.name}={param.name}")
            else:
                test_params.append(f"{param.name} = {{}}")
                param_values.append(f"{param.name}={param.name}")

        test_params.append("credentials = {'api_key': 'test_key'}")
        param_values.append("credentials=credentials")

        test_params_str = "\n        ".join(test_params)
        param_values_str = ", ".join(param_values)

        # Make sure method is lowercase for HTTP methods
        method = function_def.method.lower() if hasattr(function_def.method, 'lower') else 'get'

        # Generate the test code
        test_code = self.test_template.format(
            function_name=function_def.name,
            test_params=test_params_str,
            param_values=param_values_str,
            method=method
        )

        return test_code

    async def validate_function(self, function_def: FunctionDefinition) -> Tuple[bool, List[str]]:
        """
        Validate a function definition.

        Args:
            function_def: Function definition to validate

        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        validation_messages = []

        # Check for required fields
        if not function_def.name:
            validation_messages.append("Function name is required")

        if not function_def.description:
            validation_messages.append("Function description is required")

        if not function_def.code:
            validation_messages.append("Function code is required")

        # Validate parameter names (no spaces or special characters)
        for param in function_def.parameters:
            if not re.match(r'^[a-zA-Z0-9_]+$', param.name):
                validation_messages.append(f"Parameter name '{param.name}' contains invalid characters")

        # Check for syntax errors in the function code
        try:
            # This is a simple check - in a production system we would use ast.parse
            compile(function_def.code, '<string>', 'exec')
        except SyntaxError as e:
            validation_messages.append(f"Syntax error in function code: {str(e)}")

        return len(validation_messages) == 0, validation_messages

    def generate_responses_api_tool_schema(self, function_def: FunctionDefinition) -> Dict:
        """
        Convert a FunctionDefinition to a Responses API compatible tool schema.

        Args:
            function_def: The function definition to convert

        Returns:
            Tool schema compatible with Responses API
        """
        tool_schema = {
            "type": "function",
            "function": {
                "name": function_def.name,
                "description": function_def.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        # Convert parameters to OpenAI's tool schema format
        for param in function_def.parameters:
            param_type = param.param_type.value

            # Skip credentials parameter
            if param.name == "credentials":
                continue

            # Map Pydantic enum values to JSON Schema types
            type_mapping = {
                "string": "string",
                "integer": "integer",
                "float": "number",
                "boolean": "boolean",
                "array": "array",
                "object": "object"
            }

            json_type = type_mapping.get(param_type, "string")

            param_schema = {
                "type": json_type,
                "description": param.description
            }

            if param.default_value is not None:
                param_schema["default"] = param.default_value

            tool_schema["function"]["parameters"]["properties"][param.name] = param_schema

            if param.required:
                tool_schema["function"]["parameters"]["required"].append(param.name)

        return tool_schema

async def build_function_for_api(self, api_info: APIInfo, endpoint: APIEndpoint, tool_need_description: str = None) -> FunctionDefinition:
    """
    Build a complete function for an API based on documentation and endpoint.

    Args:
        api_info: API information
        endpoint: Specific endpoint to build function for
        tool_need_description: Optional description of the tool need for better function customization

    Returns:
        Complete function definition
    """
    # Generate function name
    function_name = self._generate_function_name(api_info.base_url, endpoint)

    # Map parameters from endpoint
    parameters = self._generate_parameters(endpoint)

    # Select appropriate template based on API type and method
    api_type = ApiType.REST  # Default to REST for now
    template = self._select_template(api_type, endpoint.method)

    # Generate function code with authentication
    code = self._generate_function_code(
        template=template,
        function_name=function_name,
        base_url=api_info.base_url,
        endpoint=endpoint,
        parameters=parameters,
        auth_methods=api_info.auth_methods
    )

    # Generate tags (with tool need description if provided)
    tags = self._generate_tags(api_info, endpoint)
    if tool_need_description:
        # Extract key terms from tool need
        need_words = re.findall(r'\b\w+\b', tool_need_description.lower())
        for word in need_words:
            if len(word) > 3 and word not in ['with', 'from', 'this', 'that', 'will', 'have', 'about', 'what']:
                tags.append(word)

    # Create function definition
    function_def = FunctionDefinition(
        name=function_name,
        description=endpoint.description or f"{endpoint.method} {endpoint.path}",
        api_source_id=api_info.source_id,
        api_type=api_type,
        parameters=parameters,
        code=code,
        endpoint=endpoint.path,
        method=endpoint.method,
        response_format=endpoint.response_format or {},
        tags=list(set(tags)),  # Deduplicate tags
        test_cases=self._generate_test_cases(endpoint)
    )

    # Validate function
    is_valid, messages = await self.validate_function(function_def)
    if not is_valid:
        logger.warning(f"Generated function has validation issues: {', '.join(messages)}")

    return function_def