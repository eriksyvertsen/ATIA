"""
Tool Executor implementation for handling Responses API tool calls.
"""

import json
import logging
import importlib
import inspect
from typing import Dict, Any, Optional, List

from atia.tool_registry import ToolRegistry
from atia.function_builder.models import FunctionDefinition

logger = logging.getLogger(__name__)

class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    pass

class ToolExecutor:
    """
    Handles execution of tools requested by the Responses API.
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the Tool Executor.

        Args:
            tool_registry: The Tool Registry for looking up tools
        """
        self.tool_registry = tool_registry

    async def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on a Responses API tool call.

        Args:
            tool_call: Tool call from the Responses API

        Returns:
            Tool execution result in the format expected by Responses API
        """
        function_name = tool_call["function"]["name"]
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in arguments: {tool_call['function']['arguments']}")
            return {
                "tool_call_id": tool_call["id"],
                "output": json.dumps({"error": f"Invalid JSON in arguments: {tool_call['function']['arguments']}"})
            }

        # Find the tool in the registry
        tool = await self.tool_registry.get_tool_by_name(function_name)
        if not tool:
            logger.error(f"Tool '{function_name}' not found")
            return {
                "tool_call_id": tool_call["id"],
                "output": json.dumps({"error": f"Tool '{function_name}' not found"})
            }

        try:
            # Get the function definition
            function_def = await self.tool_registry._get_function_definition(tool.function_id)

            if not function_def:
                logger.error(f"Function definition for '{function_name}' not found")
                return {
                    "tool_call_id": tool_call["id"],
                    "output": json.dumps({"error": f"Function definition for '{function_name}' not found"})
                }

            # Execute the function
            result = await self._execute_function(function_def, arguments)

            # Update usage statistics
            await self.tool_registry.increment_usage(tool.id)

            return {
                "tool_call_id": tool_call["id"],
                "output": json.dumps(result)
            }
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {str(e)}")
            return {
                "tool_call_id": tool_call["id"],
                "output": json.dumps({"error": f"Error executing tool: {str(e)}"})
            }

    async def _execute_function(self, function_def: FunctionDefinition, arguments: Dict[str, Any]) -> Any:
        """
        Execute a function with the given arguments.

        Args:
            function_def: Function definition
            arguments: Function arguments

        Returns:
            Function result
        """
        try:
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
            exec(function_def.code, exec_globals, namespace)

            # Get the function object
            if function_def.name not in namespace:
                logger.error(f"Function '{function_def.name}' not found in the generated code")
                raise ToolExecutionError(f"Function '{function_def.name}' not found in the generated code")

            func = namespace[function_def.name]

            # Validate arguments against function signature
            sig = inspect.signature(func)
            valid_args = {}

            for param_name, param in sig.parameters.items():
                if param_name in arguments:
                    valid_args[param_name] = arguments[param_name]
                elif param.default is not inspect.Parameter.empty:
                    # Use default value if argument not provided
                    continue
                else:
                    logger.error(f"Required argument '{param_name}' not provided")
                    raise ToolExecutionError(f"Required argument '{param_name}' not provided")

            # Call the function with arguments
            result = await func(**valid_args)

            return result
        except Exception as e:
            logger.error(f"Error executing function {function_def.name}: {str(e)}")
            raise ToolExecutionError(f"Error executing function: {str(e)}")