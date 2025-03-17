#!/bin/bash

# ATIA Phase 3 Implementation Setup Script
echo "Setting up ATIA Phase 3 implementation..."

# Create necessary directories
echo "Creating directories..."
mkdir -p atia/tool_executor
mkdir -p data/functions
mkdir -p documentation/phase3

# Create new files
echo "Creating new files..."

# 1. Create tool_executor module
cat > atia/tool_executor/__init__.py << 'EOF'
"""
Tool Executor component.

Handles execution of tools requested by the Responses API.
"""

from atia.tool_executor.executor import ToolExecutor

__all__ = ["ToolExecutor"]
EOF

cat > atia/tool_executor/executor.py << 'EOF'
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
EOF

# 2. Create updated OpenAI client utilities
cat > atia/utils/openai_client_responses.py << 'EOF'
"""
Updated OpenAI client utilities with Responses API support.

This extends the existing client with parallel implementations using the Responses API
while maintaining backward compatibility.
"""

from typing import Dict, List, Optional, Union, Any
import json
import time
import logging

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from atia.config import settings

# Initialize the OpenAI client
client = OpenAI(api_key=settings.openai_api_key)
logger = logging.getLogger(__name__)

# New Responses API functions
@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_completion_with_responses_api(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    temperature: float = settings.openai_temperature,
    tools: List[Dict] = None,
    model: str = settings.openai_model,
) -> Dict:
    """
    Get a completion using the Responses API.

    Args:
        prompt: The user prompt
        system_message: The system message
        temperature: Controls randomness (0-1)
        tools: List of tools to make available to the model
        model: The OpenAI model to use

    Returns:
        The response from the Responses API
    """
    try:
        # Create a new thread
        thread = client.beta.threads.create()
        logger.info(f"Created thread: {thread.id}")

        # Add a message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        logger.info(f"Added message to thread {thread.id}")

        # Create the run with tools if provided
        run_params = {
            "thread_id": thread.id,
            "model": model,
            "instructions": system_message,
            "temperature": temperature,
        }

        if tools:
            run_params["tools"] = tools

        run = client.beta.threads.runs.create(**run_params)
        logger.info(f"Created run: {run.id} for thread: {thread.id}")

        # Wait for the run to complete
        result = await _wait_for_run_completion(thread.id, run.id)

        return result
    except Exception as e:
        logger.error(f"Error calling OpenAI Responses API: {e}")
        raise


async def _wait_for_run_completion(thread_id: str, run_id: str) -> Dict:
    """
    Wait for a run to complete and process the results.

    Args:
        thread_id: The thread ID
        run_id: The run ID

    Returns:
        The processed response
    """
    max_polling_attempts = 60  # Prevent infinite loops
    polling_attempts = 0

    while polling_attempts < max_polling_attempts:
        polling_attempts += 1

        try:
            # Check the run status
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )

            if run.status == "completed":
                logger.info(f"Run {run_id} completed successfully")
                # Get the messages from the thread
                messages = client.beta.threads.messages.list(
                    thread_id=thread_id
                )

                # Process the messages to get the final result
                # (Get the last assistant message)
                assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]

                if assistant_messages:
                    latest_msg = assistant_messages[0]

                    # Extract text content
                    content = ""
                    for content_part in latest_msg.content:
                        if content_part.type == "text":
                            content += content_part.text.value

                    return {
                        "content": content,
                        "thread_id": thread_id,
                        "run_id": run_id
                    }

                return {"content": "", "thread_id": thread_id, "run_id": run_id}

            elif run.status == "requires_action":
                logger.info(f"Run {run_id} requires action (tool calls)")
                # This will be handled by execute_tools_in_run
                return {
                    "status": "requires_action",
                    "thread_id": thread_id,
                    "run_id": run_id
                }

            elif run.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Run {run_id} ended with status: {run.status}")
                return {
                    "error": f"Run ended with status: {run.status}",
                    "thread_id": thread_id,
                    "run_id": run_id
                }
            else:
                # Still running, wait before checking again
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error checking run status: {e}")
            return {"error": str(e), "thread_id": thread_id, "run_id": run_id}

    logger.error(f"Max polling attempts reached for run {run_id}")
    return {"error": "Max polling attempts reached", "thread_id": thread_id, "run_id": run_id}


async def execute_tools_in_run(
    thread_id: str, 
    run_id: str, 
    tool_executor,
    max_iterations: int = 10
) -> Dict:
    """
    Execute tools for a run and return the final result.

    Args:
        thread_id: The thread ID
        run_id: The run ID
        tool_executor: The ToolExecutor instance
        max_iterations: Maximum number of tool execution iterations

    Returns:
        The final response after tool execution
    """
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Tool execution iteration {iteration} for run {run_id}")

        try:
            # Check the run status
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )

            if run.status == "completed":
                logger.info(f"Run {run_id} completed")
                # Get the messages from the thread
                messages = client.beta.threads.messages.list(
                    thread_id=thread_id
                )

                # Process the messages to get the final result
                assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]

                if assistant_messages:
                    latest_msg = assistant_messages[0]

                    # Extract text content
                    content = ""
                    for content_part in latest_msg.content:
                        if content_part.type == "text":
                            content += content_part.text.value

                    return {
                        "content": content,
                        "thread_id": thread_id,
                        "run_id": run_id
                    }

                return {"content": "", "thread_id": thread_id, "run_id": run_id}

            elif run.status == "requires_action":
                logger.info(f"Run {run_id} requires action - executing tools")
                # Handle tool calls
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    logger.info(f"Executing tool: {function_name} with args: {function_args}")

                    # Execute the tool using the ToolExecutor
                    tool_result = await tool_executor.execute_tool({
                        "id": tool_call.id,
                        "function": {
                            "name": function_name,
                            "arguments": function_args
                        }
                    })

                    tool_outputs.append(tool_result)

                logger.info(f"Submitting {len(tool_outputs)} tool outputs for run {run_id}")

                # Submit the tool outputs
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )

                # Wait a moment before checking again
                time.sleep(1)

            elif run.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Run {run_id} ended with status: {run.status}")
                return {"error": f"Run ended with status: {run.status}", "thread_id": thread_id, "run_id": run_id}
            else:
                # Wait before checking again
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error during tool execution: {e}")
            return {"error": str(e), "thread_id": thread_id, "run_id": run_id}

    logger.error(f"Max iterations reached for run {run_id}")
    return {"error": "Max iterations reached", "thread_id": thread_id, "run_id": run_id}
EOF

# 3. Create function builder extension
cat > atia/function_builder/responses_api_extension.py << 'EOF'
"""
Extension to the Function Builder to support Responses API tool schemas.
Add this to the FunctionBuilder class in atia/function_builder/builder.py
"""

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
EOF

# 4. Create tool registry extension
cat > atia/tool_registry/responses_api_extension.py << 'EOF'
"""
Extension to the Tool Registry to support Responses API tools.
Add these methods to the ToolRegistry class in atia/tool_registry/registry.py
"""

async def get_tool_by_name(self, name: str) -> Optional[ToolRegistration]:
    """
    Get a tool by name.

    Args:
        name: Name of the tool

    Returns:
        Tool registration or None if not found
    """
    for tool in self._tools.values():
        if tool.name == name:
            return tool

    return None

async def get_tools_for_responses_api(self, capability_description: str = None) -> List[Dict]:
    """
    Get tools in Responses API format.

    Args:
        capability_description: Optional capability description to filter tools

    Returns:
        List of tools in Responses API format
    """
    # Import here to avoid circular imports
    from atia.function_builder.builder import FunctionBuilder

    if capability_description:
        # Search for tools matching the capability
        tools = await self.search_by_capability(capability_description)
    else:
        # Get all tools
        tools = list(self._tools.values())

    # Convert to Responses API format
    responses_api_tools = []
    function_builder = FunctionBuilder()

    for tool in tools:
        # Get the function definition
        function_def = await self._get_function_definition(tool.function_id)

        # Convert to Responses API format
        if function_def:
            tool_schema = function_builder.generate_responses_api_tool_schema(function_def)
            responses_api_tools.append(tool_schema)

    return responses_api_tools

async def _get_function_definition(self, function_id: str) -> Optional[FunctionDefinition]:
    """
    Get a function definition by ID.

    Args:
        function_id: ID of the function

    Returns:
        Function definition or None if not found
    """
    # In a real implementation, this would retrieve from a database
    # For Phase 3, we'll implement a basic file-based lookup

    # Import here to avoid circular imports
    from atia.function_builder.models import FunctionDefinition, ApiType, ParameterType, FunctionParameter
    import os
    import json

    # Check if a file with this function_id exists in the tools directory
    tools_dir = "data/functions"
    os.makedirs(tools_dir, exist_ok=True)
    function_file = f"{tools_dir}/{function_id}.json"

    if os.path.exists(function_file):
        try:
            with open(function_file, "r") as f:
                func_data = json.load(f)

            # Convert parameters from dict to FunctionParameter objects
            parameters = []
            for param_data in func_data.get("parameters", []):
                parameters.append(FunctionParameter(
                    name=param_data.get("name", ""),
                    param_type=ParameterType(param_data.get("param_type", "string")),
                    description=param_data.get("description", ""),
                    required=param_data.get("required", False),
                    default_value=param_data.get("default_value")
                ))

            return FunctionDefinition(
                id=func_data.get("id", function_id),
                name=func_data.get("name", ""),
                description=func_data.get("description", ""),
                api_source_id=func_data.get("api_source_id", ""),
                api_type=ApiType(func_data.get("api_type", "rest")),
                parameters=parameters,
                code=func_data.get("code", ""),
                endpoint=func_data.get("endpoint", ""),
                method=func_data.get("method", "GET"),
                tags=func_data.get("tags", [])
            )
        except Exception as e:
            logger.error(f"Error loading function definition: {e}")

    # If no file exists or there was an error, return a mock function definition for development
    logger.warning(f"No function definition found for {function_id}, using mock implementation")
    return FunctionDefinition(
        id=function_id,
        name=f"function_{function_id[:8]}",
        description=f"Mock function for {function_id}",
        api_source_id="mock_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="param1",
                param_type=ParameterType.STRING,
                description="A test parameter",
                required=True
            )
        ],
        code=f"""
async def function_{function_id[:8]}(param1: str):
    \"\"\"
    Mock function for {function_id}

    Args:
        param1: A test parameter

    Returns:
        A mock response
    \"\"\"
    return {{"result": param1, "function_id": "{function_id}"}}
""",
        endpoint="/test",
        method="GET"
    )
EOF

# 5. Create agent core extension
cat > atia/agent_core/responses_api_extension.py << 'EOF'
"""
Extension to the Agent Core to support the Responses API.
Add this method to the AgentCore class in atia/agent_core/core.py
"""

from typing import Dict, List, Optional, Any
from atia.utils.openai_client import get_completion, get_completion_with_responses_api, execute_tools_in_run
from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor

# Update the __init__ method to accept a tool_registry parameter
def __init__(self, name: str = "ATIA", tool_registry: Optional[ToolRegistry] = None):
    """
    Initialize the Agent Core.

    Args:
        name: The name of the agent
        tool_registry: Optional Tool Registry for tool lookup
    """
    self.name = name
    self.system_prompt = (
        f"You are {name}, an Autonomous Tool Integration Agent capable of "
        f"identifying when you need new tools, discovering APIs, and integrating them. "
        f"You have access to a function registry and can create new functions as needed."
    )
    self.tool_registry = tool_registry
    if tool_registry:
        self.tool_executor = ToolExecutor(tool_registry)
    else:
        self.tool_executor = None

# New method for processing queries with the Responses API
async def process_query_with_responses_api(
    self, 
    query: str, 
    tools: List[Dict] = None, 
    context: Optional[Dict] = None
) -> Dict:
    """
    Process a query using the Responses API with available tools.

    Args:
        query: The user query
        tools: Optional list of tools to make available
        context: Additional context about the query

    Returns:
        Response from the Responses API
    """
    if context is None:
        context = {}

    # Enhance the prompt with context if available
    enhanced_prompt = query
    if context:
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        enhanced_prompt = f"{query}\n\nContext:\n{context_str}"

    # If tools is None and we have a tool registry, get tools from there
    if tools is None and self.tool_registry:
        tools = await self.tool_registry.get_tools_for_responses_api()

    # Process the query with the Responses API
    response = await get_completion_with_responses_api(
        prompt=enhanced_prompt,
        system_message=self.system_prompt,
        temperature=settings.openai_temperature,
        tools=tools,
        model=settings.openai_model
    )

    # If the response indicates that tools need to be executed and we have a tool executor
    if response.get("status") == "requires_action" and self.tool_executor:
        # Execute tools and get the final response
        final_response = await execute_tools_in_run(
            thread_id=response["thread_id"],
            run_id=response["run_id"],
            tool_executor=self.tool_executor
        )

        return final_response

    return response
EOF

# 6. Create configuration extension
cat > atia/config/responses_api_settings.py << 'EOF'
"""
Extension to the Settings class in atia/config.py to include Responses API settings.
Add these to the Settings class.
"""

# OpenAI Responses API configuration
openai_assistant_id: Optional[str] = os.getenv("OPENAI_ASSISTANT_ID", "")
openai_responses_enabled: bool = os.getenv("OPENAI_RESPONSES_ENABLED", "False").lower() == "true"
EOF

# 7. Create main CLI update
cat > main_with_responses_api.py << 'EOF'
#!/usr/bin/env python

"""
Updated main.py to support the Responses API.
"""

import argparse
import asyncio
import sys
import json
from typing import Dict, Optional

from atia.agent_core import AgentCore
from atia.need_identifier import NeedIdentifier
from atia.api_discovery import APIDiscovery
from atia.doc_processor import DocumentationProcessor
from atia.account_manager import AccountManager
from atia.tool_registry import ToolRegistry
from atia.config import settings


async def process_query(query: str, context: Optional[Dict] = None, use_responses_api: bool = False) -> None:
    """
    Process a user query and display the results.

    Args:
        query: The user query
        context: Additional context about the query
        use_responses_api: Whether to use the Responses API
    """
    if context is None:
        context = {}

    # Initialize components
    tool_registry = ToolRegistry()
    agent = AgentCore(tool_registry=tool_registry)
    need_identifier = NeedIdentifier()
    api_discovery = APIDiscovery()
    doc_processor = DocumentationProcessor()
    account_manager = AccountManager()

    print("\n🤖 ATIA - Autonomous Tool Integration Agent\n")
    print(f"Query: {query}\n")

    # Step 1: Process the query with the agent
    print("🧠 Processing query...")

    if use_responses_api or settings.openai_responses_enabled:
        print("Using Responses API...")
        agent_response = await agent.process_query_with_responses_api(query, context=context)
        print(f"Initial response: {agent_response.get('content', '')}\n")
    else:
        agent_response = await agent.process_query(query, context)
        print(f"Initial response: {agent_response}\n")

    # Step 2: Identify if a tool is needed
    print("🔍 Identifying tool needs...")
    tool_need = await need_identifier.identify_tool_need(query, context)

    if tool_need:
        print(f"Tool need identified: {tool_need.category}")
        print(f"Description: {tool_need.description}")
        print(f"Confidence: {tool_need.confidence:.2f}\n")

        # Step 3: Discover APIs for the identified need
        print("🔎 Searching for relevant APIs...")
        api_candidates = await api_discovery.search_for_api(tool_need.description)

        if api_candidates:
            top_api = api_candidates[0]
            print(f"Top API candidate: {top_api.name}")
            print(f"Provider: {top_api.provider}")
            print(f"Description: {top_api.description}")
            print(f"Documentation URL: {top_api.documentation_url}")
            print(f"Relevance score: {top_api.relevance_score:.2f}\n")

            # Step 4: Process API documentation
            print("📝 Fetching API documentation...")
            try:
                # For Phase 2-3, we can attempt to fetch and process the documentation
                print("Processing API documentation...")

                # Convert the API candidate to a format suitable for account registration
                api_info = {
                    "id": f"api_{hash(top_api.name)}",
                    "name": top_api.name,
                    "provider": top_api.provider,
                    "description": top_api.description,
                    "documentation_url": top_api.documentation_url,
                    "auth_type": top_api.auth_type if hasattr(top_api, "auth_type") else "api_key",
                    "requires_auth": top_api.requires_auth
                }

                # Step 5: Register with the Account Manager
                print("🔑 Handling authentication...")
                credential = await account_manager.register_api(api_info)

                print(f"Successfully registered {top_api.name} with authentication type: {credential.auth_type}")
                print(f"Credential ID: {credential.id}")

                # In a full implementation, we would use this credential to make API calls
                print("Ready to use the API with secure authentication.")

            except Exception as e:
                print(f"Error processing API: {e}")
        else:
            print("No suitable APIs found for this need.")
    else:
        print("No tool need identified for this query.")


async def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="ATIA - Autonomous Tool Integration Agent")
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--responses-api", "-r", action="store_true", help="Use the Responses API")

    args = parser.parse_args()

    if args.interactive:
        print("🤖 ATIA - Autonomous Tool Integration Agent")
        print(f"{'Using Responses API' if args.responses_api or settings.openai_responses_enabled else 'Using standard API'}")
        print("Type 'exit' or 'quit' to exit.")

        while True:
            query = input("\nEnter your query: ")

            if query.lower() in ["exit", "quit"]:
                break

            await process_query(query, use_responses_api=args.responses_api)
    elif args.query:
        await process_query(args.query, use_responses_api=args.responses_api)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
EOF

# 8. Create test file
cat > tests/test_responses_api.py << 'EOF'
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
EOF

# 9. Create implementation guide
cat > documentation/phase3/implementation_guide.md << 'EOF'
# ATIA Phase 3: Responses API Integration

This guide provides instructions for implementing Phase 3 of the ATIA project, which focuses on integrating OpenAI's Responses API while maintaining backward compatibility with the existing implementation.

## Overview

Phase 3 adds the following capabilities to ATIA:

1. Parallel implementation of OpenAI client utilities using Responses API
2. Tool schema generation for Responses API compatibility
3. Tool execution framework for handling Responses API tool calls
4. Extended Tool Registry functionality to support Responses API

## Implementation Steps

### 1. Create ToolExecutor component

The ToolExecutor component is responsible for executing tools requested by the Responses API:

```bash
# Copy the files created by this script
mkdir -p atia/tool_executor
cp atia/tool_executor/*.py atia/tool_executor/
```

### 2. Update OpenAI client utilities

Add the Responses API utility functions to the OpenAI client:

```python
# Add the content of openai_client_responses.py to atia/utils/openai_client.py
# Make sure to keep all existing functions
```

### 3. Update Function Builder

Add the `generate_responses_api_tool_schema` method to the FunctionBuilder class:

```python
# Add the content of function_builder/responses_api_extension.py to atia/function_builder/builder.py
# Add it as a method to the FunctionBuilder class
```

### 4. Update Tool Registry

Add the Responses API support methods to the ToolRegistry class:

```python
# Add the content of tool_registry/responses_api_extension.py to atia/tool_registry/registry.py
# Add these as methods to the ToolRegistry class
```

### 5. Update Agent Core

Update the Agent Core class to support the Responses API:

```python
# Update atia/agent_core/core.py with the content of agent_core/responses_api_extension.py
# Update the __init__ method and add the process_query_with_responses_api method
```

### 6. Update Configuration

Add Responses API settings to the Settings class:

```python
# Add the content of config/responses_api_settings.py to atia/config.py
# Add these settings to the Settings class
```

### 7. Update Main CLI

Update the main.py script to support the Responses API:

```python
# Update main.py with the content of main_with_responses_api.py
```

### 8. Create directories for function storage

```bash
mkdir -p data/functions
```

### 9. Update .env file

Add the following environment variables to your `.env` file:

```
# OpenAI Responses API configuration
OPENAI_ASSISTANT_ID=your_assistant_id
OPENAI_RESPONSES_ENABLED=false
```

### 10. Run tests

Run the tests to ensure that the Responses API integration is working correctly:

```bash
python -m pytest tests/test_responses_api.py -v
```

## Backward Compatibility

The implementation maintains backward compatibility with the existing API. The Responses API is only used when explicitly requested through:

1. Command-line argument: `--responses-api` or `-r`
2. Environment variable: `OPENAI_RESPONSES_ENABLED=true`

## Usage

Use the updated CLI with the `--responses-api` flag:

```bash
python -m atia --responses-api "Your query here"
```

Or run in interactive mode with Responses API:

```bash
python -m atia --interactive --responses-api
```
EOF

# 10. Create instruction file for integration
cat > documentation/phase3/integration_instructions.md << 'EOF'
# Phase 3 Integration Instructions

This document provides instructions for integrating the Phase 3 components into the existing codebase.

## Steps for Integration

1. **Create Tool Executor Component**

```bash
# Create directory and copy files
mkdir -p atia/tool_executor
cp atia/tool_executor/__init__.py atia/tool_executor/
cp atia/tool_executor/executor.py atia/tool_executor/
```

2. **Update OpenAI Client**

Open `atia/utils/openai_client.py` and add all the new functions from `atia/utils/openai_client_responses.py`. Make sure to keep all existing functions and imports.

3. **Update Function Builder**

Open `atia/function_builder/builder.py` and add the `generate_responses_api_tool_schema` method from `atia/function_builder/responses_api_extension.py` to the `FunctionBuilder` class.

4. **Update Tool Registry**

Open `atia/tool_registry/registry.py` and add the new methods from `atia/tool_registry/responses_api_extension.py` to the `ToolRegistry` class.

5. **Update Agent Core**

Open `atia/agent_core/core.py` and:
- Update the `__init__` method to accept a `tool_registry` parameter
- Add the `process_query_with_responses_api` method from `atia/agent_core/responses_api_extension.py`

6. **Update Configuration**

Open `atia/config.py` and add the Responses API settings from `atia/config/responses_api_settings.py` to the `Settings` class.

7. **Update Main CLI**

Update `main.py` with the content from `main_with_responses_api.py`.

8. **Add Tests**

Copy the Responses API tests:

```bash
cp tests/test_responses_api.py tests/
```

9. **Create Required Directories**

```bash
mkdir -p data/functions
```

10. **Update Environment Variables**

Add these lines to your `.env` file:

```
# OpenAI Responses API configuration
OPENAI_ASSISTANT_ID=your_assistant_id
OPENAI_RESPONSES_ENABLED=false
```

11. **Run Tests**

Run the tests to verify the integration:

```bash
python -m pytest tests/test_responses_api.py -v
```

12. **Test the CLI**

Try running the CLI with the Responses API option:

```bash
python -m atia --responses-api "Hello, what can you do?"
```

## Additional Notes

- Maintain backward compatibility: Ensure all existing functionality continues to work without the Responses API.
- Error handling: Make sure error handling is in place for Responses API calls.
- Validation: Test thoroughly with and without the Responses API enabled.
EOF

# Create a test helper script
cat > test_phase3.py << 'EOF'
#!/usr/bin/env python

"""
Helper script to test Phase 3 implementation.
"""

import asyncio
import argparse
import sys

from atia.agent_core import AgentCore
from atia.tool_registry import ToolRegistry
from atia.tool_executor import ToolExecutor
from atia.function_builder import FunctionBuilder, FunctionDefinition
from atia.function_builder.models import ApiType, ParameterType, FunctionParameter


async def test_responses_api():
    """Test the Responses API implementation."""
    print("Testing Responses API implementation...\n")

    # Create a tool registry
    tool_registry = ToolRegistry()

    # Create an agent
    agent = AgentCore(tool_registry=tool_registry)

    # Create a function builder
    function_builder = FunctionBuilder()

    # Create a test function definition
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
            )
        ],
        code="""
async def test_function(param1: str):
    \"\"\"
    A test function

    Args:
        param1: A test parameter

    Returns:
        A test result
    \"\"\"
    return {"result": f"You said: {param1}"}
""",
        endpoint="/test",
        method="GET"
    )

    # Register the function in the tool registry
    print("Registering test function...")
    tool = await tool_registry.register_function(function_def)
    print(f"Registered tool: {tool.name}\n")

    # Generate a Responses API tool schema
    print("Generating Responses API tool schema...")
    tool_schema = function_builder.generate_responses_api_tool_schema(function_def)
    print(f"Tool schema generated: {tool_schema}\n")

    # Get tools for Responses API
    print("Getting tools for Responses API...")
    tools = await tool_registry.get_tools_for_responses_api()
    print(f"Got {len(tools)} tools for Responses API\n")

    # Process a query with the Responses API
    print("Processing query with Responses API...")
    query = "Please use the test_function with param1='Hello, world!'"

    try:
        response = await agent.process_query_with_responses_api(
            query=query,
            tools=tools
        )
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error processing query: {e}\n")

    print("Responses API testing complete!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test ATIA Phase 3 implementation")
    parser.add_argument("--responses-api", action="store_true", help="Test Responses API implementation")

    args = parser.parse_args()

    if args.responses_api:
        await test_responses_api()
    else:
        print("Please specify a test to run. Use --responses-api to test the Responses API implementation.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
EOF

# Create .env extension file
cat > .env.phase3 << 'EOF'
# OpenAI Responses API configuration
OPENAI_ASSISTANT_ID=
OPENAI_RESPONSES_ENABLED=false
EOF

echo "Making test script executable..."
chmod +x test_phase3.py

echo "Setup complete! Follow the integration instructions in documentation/phase3/integration_instructions.md to integrate the Phase 3 components into the existing codebase."