"""
Function Builder updates for Phase 1: Week 2.

This update enhances the Function Builder component with better Responses API integration
and improved authentication handling.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

from atia.config import settings
from atia.function_builder.builder import FunctionBuilder
from atia.function_builder.models import (
    FunctionDefinition, 
    ApiType, 
    ParameterType, 
    FunctionParameter
)
from atia.doc_processor.processor import APIEndpoint, APIInfo

logger = logging.getLogger(__name__)

# Enhanced Responses API schema generation
def enhanced_responses_api_tool_schema(
    self, 
    function_def: FunctionDefinition,
    include_auth: bool = True
) -> Dict:
    """
    Enhanced version of generate_responses_api_tool_schema with better authentication support.

    Args:
        function_def: The function definition to convert
        include_auth: Whether to include authentication information

    Returns:
        Enhanced tool schema compatible with Responses API
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

    # Enhanced parameter conversion with better type handling
    for param in function_def.parameters:
        param_name = param.name

        # Skip credentials parameter which will be handled separately
        if param_name == "credentials":
            continue

        # Map parameter types to JSON Schema types with enhanced support
        type_mapping = {
            "string": {
                "type": "string",
                "description": param.description
            },
            "integer": {
                "type": "integer",
                "description": param.description
            },
            "float": {
                "type": "number",
                "description": param.description
            },
            "boolean": {
                "type": "boolean",
                "description": param.description
            },
            "array": {
                "type": "array",
                "description": param.description,
                "items": {"type": "string"}  # Default to string items
            },
            "object": {
                "type": "object",
                "description": param.description,
                "properties": {}  # Empty properties by default
            }
        }

        # Get the appropriate schema for this parameter type
        param_schema = type_mapping.get(param.param_type.value, {"type": "string", "description": param.description})

        # Add default value if specified
        if param.default_value is not None:
            param_schema["default"] = param.default_value

        # Add enum if parameter has validation constraints (simple implementation)
        if hasattr(param, "metadata") and param.metadata and "enum" in param.metadata:
            param_schema["enum"] = param.metadata["enum"]

        # Add the parameter to the properties
        tool_schema["function"]["parameters"]["properties"][param_name] = param_schema

        # Add to required list if parameter is required
        if param.required:
            tool_schema["function"]["parameters"]["required"].append(param_name)

    # Add authentication information if requested
    if include_auth:
        # Extract auth type from tags or infer from endpoint method
        auth_type = next((tag for tag in function_def.tags if tag in ["api_key", "oauth", "bearer", "basic"]), None)

        if auth_type or "credentials" in [p.name for p in function_def.parameters]:
            # Add auth metadata to the tool schema
            tool_schema["authentication"] = {
                "type": auth_type or "api_key",
                "required": True
            }

            # Add additional auth information for specific types
            if auth_type == "oauth":
                tool_schema["authentication"]["scope"] = "read write"  # Default scope

    return tool_schema

# Enhanced function generation with improved authentication and error handling
async def enhanced_function_generation(
    self, 
    api_info: APIInfo, 
    endpoint: APIEndpoint,
    capability_description: Optional[str] = None
) -> FunctionDefinition:
    """
    Enhanced function generation with better authentication and error handling.

    Args:
        api_info: Information about the API
        endpoint: The specific endpoint to generate a function for
        capability_description: Optional description for better function customization

    Returns:
        Generated function definition
    """
    logger.info(f"Generating enhanced function for endpoint: {endpoint.method} {endpoint.path}")

    try:
        # Use the existing function generation code as a base
        function_def = await self.build_function_for_api(
            api_info=api_info,
            endpoint=endpoint,
            tool_need_description=capability_description
        )

        # Enhance the generated function with better authentication handling
        function_def = _enhance_authentication(function_def, api_info)

        # Add better error handling to the function code
        function_def.code = _add_enhanced_error_handling(function_def.code)

        # Add standard credential parameter if not present
        _ensure_credential_parameter(function_def)

        # Update function tags with authentication information
        _update_function_tags(function_def, api_info)

        # Validate the enhanced function
        is_valid, messages = await self.validate_function(function_def)
        if not is_valid:
            logger.warning(f"Generated function has validation issues: {messages}")
            # Try to fix common issues
            function_def = _fix_common_validation_issues(function_def, messages)

        return function_def

    except Exception as e:
        logger.error(f"Error in enhanced function generation: {e}")
        # Return a simplified function as fallback
        return _create_fallback_function(api_info, endpoint, capability_description)

# Helper functions for enhanced function generation

def _enhance_authentication(function_def: FunctionDefinition, api_info: APIInfo) -> FunctionDefinition:
    """Add enhanced authentication handling to the function."""
    # Determine authentication type from API info
    auth_type = "api_key"  # Default
    if api_info.auth_methods:
        auth_method = api_info.auth_methods[0]
        auth_type = auth_method.get("type", "").lower()
        if auth_type == "http":
            scheme = auth_method.get("scheme", "").lower()
            if scheme == "bearer":
                auth_type = "bearer"
            elif scheme == "basic":
                auth_type = "basic"

    # Add authentication type to function's metadata
    if not hasattr(function_def, "metadata"):
        function_def.metadata = {}
    function_def.metadata["auth_type"] = auth_type

    return function_def

def _add_enhanced_error_handling(code: str) -> str:
    """Add enhanced error handling to the function code."""
    # Check if error handling is already present
    if "APIError" in code and "response.status" in code:
        # Already has error handling, make sure it's comprehensive
        if "try:" not in code:
            # Add try-except block around the main function body
            code_lines = code.split("\n")
            function_def_line = next((i for i, line in enumerate(code_lines) if line.startswith("async def")), -1)

            if function_def_line >= 0:
                # Find the function body indentation
                body_start = function_def_line + 1
                while body_start < len(code_lines) and (not code_lines[body_start].strip() or code_lines[body_start].lstrip().startswith('"""')):
                    body_start += 1

                if body_start < len(code_lines):
                    # Get indentation
                    indent = len(code_lines[body_start]) - len(code_lines[body_start].lstrip())
                    indent_str = " " * indent

                    # Find where to insert try-except
                    # Skip past docstring if present
                    if '"""' in code_lines[body_start:function_def_line+10]:
                        docstring_start = next((i for i, line in enumerate(code_lines[body_start:function_def_line+10], body_start) if '"""' in line), -1)
                        if docstring_start >= 0:
                            docstring_end = next((i for i, line in enumerate(code_lines[docstring_start+1:], docstring_start+1) if '"""' in line), -1)
                            if docstring_end >= 0:
                                body_start = docstring_end + 1

                    # Insert try block
                    code_lines.insert(body_start, indent_str + "try:")

                    # Insert except block at the end
                    code_lines.append(indent_str + "except Exception as e:")
                    code_lines.append(indent_str + "    error_msg = f\"Error executing function: {str(e)}\"")
                    code_lines.append(indent_str + "    raise APIError(error_msg) from e")

                    code = "\n".join(code_lines)

    # Enhance the error handling for API responses
    if "response.status" in code and "error_text = await response.text()" not in code:
        # Add better error handling for API responses
        code = code.replace(
            "if response.status < 200 or response.status >= 300:",
            """if response.status < 200 or response.status >= 300:
                # Try to get error details from response
                try:
                    error_json = await response.json()
                    error_message = error_json.get("message", error_json.get("error", ""))
                    if error_message:
                        raise APIError(f"API Error ({response.status}): {error_message}")
                except:
                    # Fallback to text response
                    error_text = await response.text()
                    raise APIError(f"Request failed with status {response.status}: {error_text}")"""
        )

    return code

def _ensure_credential_parameter(function_def: FunctionDefinition) -> None:
    """Ensure the function has a credentials parameter if authentication is needed."""
    # Check if credentials parameter already exists
    has_credentials = any(param.name == "credentials" for param in function_def.parameters)

    if not has_credentials:
        # Add credentials parameter
        function_def.parameters.append(
            FunctionParameter(
                name="credentials",
                param_type=ParameterType.OBJECT,
                description="Authentication credentials for the API",
                required=False,
                default_value=None
            )
        )

def _update_function_tags(function_def: FunctionDefinition, api_info: APIInfo) -> None:
    """Update function tags with authentication information."""
    auth_tags = set()

    # Extract auth type from API info
    if api_info.auth_methods:
        for auth_method in api_info.auth_methods:
            auth_type = auth_method.get("type", "").lower()
            auth_tags.add(auth_type)

            if auth_type == "http":
                scheme = auth_method.get("scheme", "").lower()
                if scheme:
                    auth_tags.add(scheme)

    # Add auth tags to function tags
    function_def.tags = list(set(function_def.tags).union(auth_tags))

def _fix_common_validation_issues(function_def: FunctionDefinition, messages: List[str]) -> FunctionDefinition:
    """Fix common validation issues in the function definition."""
    # Fix invalid parameter names
    for message in messages:
        if "Parameter name" in message and "contains invalid characters" in message:
            # Extract parameter name
            param_name_match = re.search(r"Parameter name '(\w+)'", message)
            if param_name_match:
                invalid_param = param_name_match.group(1)
                # Find and fix the parameter
                for param in function_def.parameters:
                    if param.name == invalid_param:
                        # Clean the parameter name
                        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', param.name)
                        param.name = clean_name
                        # Update parameter reference in code
                        function_def.code = function_def.code.replace(invalid_param, clean_name)

    # Fix syntax errors in function code
    for message in messages:
        if "Syntax error" in message:
            # Simple fixes for common syntax errors
            if "EOF while scanning" in message or "unexpected indent" in message:
                # Try to fix indentation or missing closing quotes/brackets
                function_def.code = _fix_syntax_errors(function_def.code)

    return function_def

def _fix_syntax_errors(code: str) -> str:
    """Simple fixes for common syntax errors."""
    # Check for unbalanced brackets/parentheses
    brackets = {'(': ')', '{': '}', '[': ']'}
    stack = []

    for char in code:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                # Unbalanced bracket found
                if char == ')':
                    code += ')'
                elif char == '}':
                    code += '}'
                elif char == ']':
                    code += ']'

    # Add any missing closing brackets
    while stack:
        bracket = stack.pop()
        code += brackets[bracket]

    # Check for unclosed string literals
    lines = code.split('\n')
    fixed_lines = []

    for line in lines:
        # Count quotes
        single_quotes = line.count("'")
        double_quotes = line.count('"')

        # If odd number of quotes, add a closing quote
        if single_quotes % 2 == 1:
            line += "'"
        if double_quotes % 2 == 1:
            line += '"'

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def _create_fallback_function(api_info: APIInfo, endpoint: APIEndpoint, capability_description: Optional[str] = None) -> FunctionDefinition:
    """Create a simplified fallback function when normal generation fails."""
    function_name = f"fallback_{endpoint.method.lower()}_{api_info.source_id[:6]}"

    # Create basic parameters based on endpoint
    parameters = []
    if endpoint.parameters:
        for i, param in enumerate(endpoint.parameters[:5]):  # Limit to 5 parameters
            param_name = f"param{i+1}"
            if isinstance(param, dict):
                description = param.get("description", f"Parameter {i+1}")
                required = param.get("required", False)
            else:
                description = getattr(param, "description", f"Parameter {i+1}")
                required = getattr(param, "required", False)

            parameters.append(
                FunctionParameter(
                    name=param_name,
                    param_type=ParameterType.STRING,
                    description=description,
                    required=required
                )
            )

    # Add credentials parameter
    parameters.append(
        FunctionParameter(
            name="credentials",
            param_type=ParameterType.OBJECT,
            description="Authentication credentials for the API",
            required=False
        )
    )

    # Create a simple function code
    code = f"""
async def {function_name}({', '.join([p.name for p in parameters])}):
    \"\"\"
    {endpoint.description or f"{endpoint.method} {endpoint.path}"}

    Args:
        {chr(10)+'        '.join([f"{p.name}: {p.description}" for p in parameters])}

    Returns:
        API response
    \"\"\"
    import aiohttp
    import json

    url = "{api_info.base_url}{endpoint.path}"
    headers = {{\"Content-Type\": \"application/json\"}}

    # Add authentication if credentials provided
    if credentials:
        if \"api_key\" in credentials:
            headers[\"X-API-Key\"] = credentials[\"api_key\"]
        elif \"token\" in credentials:
            headers[\"Authorization\"] = f\"Bearer {{credentials['token']}}\"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.{endpoint.method.lower()}(url, headers=headers) as response:
                if response.status < 200 or response.status >= 300:
                    error_text = await response.text()
                    raise Exception(f\"Request failed with status {{response.status}}: {{error_text}}\")

                result = await response.json()
                return result
    except Exception as e:
        raise Exception(f\"Error calling API: {{str(e)}}\")
"""

    # Create basic function definition
    function_def = FunctionDefinition(
        name=function_name,
        description=endpoint.description or f"{endpoint.method} {endpoint.path}",
        api_source_id=api_info.source_id,
        api_type=ApiType.REST,
        parameters=parameters,
        code=code,
        endpoint=endpoint.path,
        method=endpoint.method,
        tags=["fallback", endpoint.method.lower(), "auto_generated"],
        response_format={}
    )

    return function_def

# Add the enhanced methods to the FunctionBuilder class
FunctionBuilder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema
FunctionBuilder.enhanced_function_generation = enhanced_function_generation
