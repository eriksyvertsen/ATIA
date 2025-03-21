"""
Tool Executor implementation for handling Responses API tool calls.
"""

import json
import logging
import importlib
import inspect
import asyncio
import traceback
from typing import Dict, Any, Optional, List, Callable

from atia.tool_registry import ToolRegistry
from atia.function_builder.models import FunctionDefinition
from atia.account_manager import AccountManager

logger = logging.getLogger(__name__)

class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    pass

class ToolExecutor:
    """
    Handles execution of tools requested by the Responses API.
    """

    def __init__(self, tool_registry: ToolRegistry, account_manager: Optional[AccountManager] = None):
        """
        Initialize the Tool Executor.

        Args:
            tool_registry: The Tool Registry for looking up tools
            account_manager: Optional Account Manager for handling credentials
        """
        self.tool_registry = tool_registry
        self.account_manager = account_manager

        # Cache for dynamically created functions
        self._function_cache = {}

    async def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on a Responses API tool call.

        Args:
            tool_call: Tool call from the Responses API

        Returns:
            Tool execution result in the format expected by Responses API
        """
        function_name = tool_call["function"]["name"]
        tool_call_id = tool_call.get("id", "unknown")

        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in arguments: {tool_call['function']['arguments']}")
            return {
                "tool_call_id": tool_call_id,
                "output": json.dumps({"error": f"Invalid JSON in arguments: {tool_call['function']['arguments']}"})
            }

        # Find the tool in the registry
        tool = await self.tool_registry.get_tool_by_name(function_name)
        if not tool:
            logger.error(f"Tool '{function_name}' not found")
            return {
                "tool_call_id": tool_call_id,
                "output": json.dumps({"error": f"Tool '{function_name}' not found"})
            }

        try:
            # Get the function definition
            function_def = await self.tool_registry._get_function_definition(tool.function_id)

            if not function_def:
                logger.error(f"Function definition for '{function_name}' not found")
                return {
                    "tool_call_id": tool_call_id,
                    "output": json.dumps({"error": f"Function definition for '{function_name}' not found"})
                }

            # Get credentials if we have an account manager
            credentials = None
            if self.account_manager and hasattr(tool, 'api_source_id'):
                try:
                    credentials = await self.account_manager.get_credentials_for_api(tool.api_source_id)
                    logger.info(f"Using credentials for API {tool.api_source_id}")
                except Exception as e:
                    logger.error(f"Error getting credentials: {e}")

            # Add credentials to arguments if the function expects them
            if credentials:
                function_sig = self._get_function_signature(function_def)
                if 'credentials' in function_sig:
                    arguments['credentials'] = credentials

            # Execute the function
            result = await self._execute_function(function_def, arguments)

            # Update usage statistics
            await self.tool_registry.increment_usage(tool.id)

            # Format the result for Responses API
            return {
                "tool_call_id": tool_call_id,
                "output": json.dumps(result)
            }
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {str(e)}")
            logger.debug(traceback.format_exc())

            # Provide a more detailed error response
            error_details = {
                "error": f"Error executing tool: {str(e)}",
                "error_type": type(e).__name__,
                "tool_name": function_name
            }

            return {
                "tool_call_id": tool_call_id,
                "output": json.dumps(error_details)
            }

    def _get_function_signature(self, function_def: FunctionDefinition) -> Dict[str, Any]:
        """
        Get parameter names and defaults for a function.

        Args:
            function_def: Function definition

        Returns:
            Dictionary of parameter information
        """
        # Extract parameters from function definition
        function_sig = {}

        for param in function_def.parameters:
            function_sig[param.name] = {
                'required': param.required,
                'default': param.default_value,
                'type': param.param_type.value
            }

        return function_sig

    async def _execute_function(self, function_def: FunctionDefinition, arguments: Dict[str, Any]) -> Any:
        """
        Execute a function with the given arguments.

        Args:
            function_def: Function definition
            arguments: Function arguments

        Returns:
            Function result
        """
        # Check if we already have this function in cache
        if function_def.id in self._function_cache:
            func = self._function_cache[function_def.id]
        else:
            # Create a namespace for the function
            namespace = {}

            # Add necessary imports
            exec_globals = {
                "asyncio": importlib.import_module("asyncio"),
                "aiohttp": importlib.import_module("aiohttp"),
                "json": importlib.import_module("json"),
                "APIError": type("APIError", (Exception,), {})
            }

            # Execute the code in the namespace
            try:
                exec(function_def.code, exec_globals, namespace)
            except Exception as e:
                logger.error(f"Error executing function code: {e}")
                raise ToolExecutionError(f"Error compiling function: {e}")

            # Get the function object
            if function_def.name not in namespace:
                logger.error(f"Function '{function_def.name}' not found in the generated code")
                raise ToolExecutionError(f"Function '{function_def.name}' not found in the generated code")

            func = namespace[function_def.name]

            # Cache the function
            self._function_cache[function_def.id] = func

        # Validate arguments against function signature
        sig = inspect.signature(func)
        valid_args = {}

        for param_name, param in sig.parameters.items():
            if param_name in arguments:
                # Type checking and conversion
                param_type = next((p.param_type.value for p in function_def.parameters if p.name == param_name), None)
                if param_type:
                    # Convert if needed
                    valid_args[param_name] = self._convert_argument(arguments[param_name], param_type)
                else:
                    valid_args[param_name] = arguments[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Use default value if argument not provided
                continue
            else:
                logger.error(f"Required argument '{param_name}' not provided")
                raise ToolExecutionError(f"Required argument '{param_name}' not provided")

        try:
            # Call the function with arguments
            result = await func(**valid_args)
            return result
        except Exception as e:
            logger.error(f"Error calling function {function_def.name}: {e}")
            logger.debug(traceback.format_exc())
            raise ToolExecutionError(f"Error calling function: {e}")

    def _convert_argument(self, value: Any, param_type: str) -> Any:
        """
        Convert an argument to the appropriate type.

        Args:
            value: Argument value
            param_type: Parameter type string

        Returns:
            Converted value
        """
        try:
            if param_type == "string":
                return str(value)
            elif param_type == "integer":
                return int(value)
            elif param_type == "float":
                return float(value)
            elif param_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ['true', 'yes', '1', 'y']
                return bool(value)
            else:
                # For arrays and objects, just return as is
                return value
        except (ValueError, TypeError):
            # If conversion fails, return original value
            return value

    async def execute_tools_for_run(self, thread_id: str, run_id: str, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls for a Responses API run.

        Args:
            thread_id: Thread ID
            run_id: Run ID
            tool_calls: List of tool calls

        Returns:
            List of tool outputs
        """
        tool_outputs = []

        # Execute each tool call
        for tool_call in tool_calls:
            result = await self.execute_tool(tool_call)
            tool_outputs.append(result)

        return tool_outputs