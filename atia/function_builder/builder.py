"""
Function Builder implementation for generating API functions.
"""

import json
import logging
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
        self.test_template = FUNCTION_TEST_TEMPLATE

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
            parameters
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
        domain = urlparse(base_url).netloc.split('.')[0]

        # Clean up the path to create a function name
        path_parts = [p for p in endpoint.path.split('/') if p and not p.startswith('{')]
        if path_parts:
            name_parts = [domain] + path_parts
        else:
            name_parts = [domain, endpoint.method.lower()]

        # Create snake_case function name
        function_name = '_'.join(name_parts).lower()

        return function_name

    def _generate_parameters(self, endpoint: APIEndpoint) -> List[FunctionParameter]:
        """Generate function parameters from endpoint parameters."""
        result = []
        for param in endpoint.parameters:
            param_type = self._map_parameter_type(param.get('parameter_type', 'string'))

            result.append(FunctionParameter(
                name=param.get('name', ''),
                param_type=param_type,
                description=param.get('description', ''),
                required=param.get('required', False),
                default_value=param.get('default', None)
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
                              parameters: List[FunctionParameter]) -> str:
        """Generate the function code using the template."""
        # Generate parameter string for function signature
        param_str = ', '.join([f"{p.name}: {p.param_type.value}" for p in parameters])
        if param_str:
            param_str = param_str

        # Generate parameter descriptions for docstring
        param_desc = '\n    '.join([f"{p.name}: {p.description}" for p in parameters])
        if not param_desc:
            param_desc = "No parameters required."

        # Generate params dictionary for requests
        params_dict = '{' + ', '.join([f'"{p.name}": {p.name}' for p in parameters]) + '}'
        if not parameters:
            params_dict = '{}'

        # Full URL
        full_url = f"{base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"

        # Replace template placeholders
        code = template.format(
            function_name=function_name,
            parameters=param_str,
            parameter_descriptions=param_desc,
            base_url=base_url,
            path=endpoint.path,
            method=endpoint.method.upper(),
            params_dict=params_dict,
            full_url=full_url,
            description=endpoint.description,
            return_description="The API response data."
        )

        return code

    def _generate_tags(self, api_info: APIInfo, endpoint: APIEndpoint) -> List[str]:
        """Generate capability tags for the function."""
        tags = []

        # Add domain tag if available
        if hasattr(api_info, 'domain'):
            tags.append(api_info.domain)

        # Add tags based on endpoint description
        if endpoint.description:
            # In a real implementation, we would use NLP to extract key concepts
            # For Phase 2, we'll use a simple approach
            keywords = ["search", "create", "update", "delete", "list", "get"]
            for keyword in keywords:
                if keyword in endpoint.description.lower():
                    tags.append(keyword)

        return tags

    def _generate_test_cases(self, endpoint: APIEndpoint) -> List[Dict[str, Any]]:
        """Generate test cases for the function."""
        test_cases = []

        # Generate mock params
        mock_params = {}
        for param in endpoint.parameters:
            param_type = param.get('parameter_type', 'string')
            param_name = param.get('name', '')

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

        test_params_str = "\n        ".join(test_params)
        param_values_str = ", ".join(param_values)

        # Generate the test code
        test_code = self.test_template.format(
            function_name=function_def.name,
            test_params=test_params_str,
            param_values=param_values_str,
            method=function_def.method
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
            if not param.name.isalnum() and '_' not in param.name:
                validation_messages.append(f"Parameter name '{param.name}' contains invalid characters")

        # Check for syntax errors in the function code
        try:
            # This is a simple check - in a production system we would use ast.parse
            compile(function_def.code, '<string>', 'exec')
        except SyntaxError as e:
            validation_messages.append(f"Syntax error in function code: {str(e)}")

        return len(validation_messages) == 0, validation_messages