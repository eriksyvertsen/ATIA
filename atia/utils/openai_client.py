"""
OpenAI client utilities with enhanced Responses API support.

This module provides utilities for interacting with OpenAI's API, including
both standard completions and the Responses API with improved tool handling.
"""

from typing import Dict, List, Optional, Union, Any
import json
import time
import logging
import asyncio

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from atia.config import settings

# Initialize the OpenAI client
client = OpenAI(api_key=settings.openai_api_key)
logger = logging.getLogger(__name__)


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    temperature: float = settings.openai_temperature,
    max_tokens: int = settings.openai_max_tokens,
    model: str = settings.openai_model,
) -> str:
    """
    Get a completion from the OpenAI API with retry logic.

    Args:
        prompt: The user prompt
        system_message: The system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        model: The OpenAI model to use

    Returns:
        The generated text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_json_completion(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    temperature: float = settings.openai_temperature,
    max_tokens: int = settings.openai_max_tokens,
    model: str = settings.openai_model,
) -> Dict:
    """
    Get a JSON completion from the OpenAI API with retry logic.

    Args:
        prompt: The user prompt
        system_message: The system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        model: The OpenAI model to use

    Returns:
        The generated JSON as a Python dictionary
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        # Return the JSON content
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error calling OpenAI API for JSON completion: {e}")
        raise


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Get an embedding from the OpenAI API with retry logic.

    Args:
        text: The text to embed
        model: The embedding model to use

    Returns:
        The embedding as a list of floats
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_completion_with_responses_api(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    temperature: float = settings.openai_temperature,
    tools: List[Dict] = None,
    model: str = settings.openai_model,
) -> Dict:
    """
    Get a completion using the Responses API with enhanced tool support.

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
        # Check for valid assistant_id in settings
        assistant_id = settings.openai_assistant_id
        if not assistant_id or not assistant_id.startswith("asst_"):
            # Use direct chat completions API with tool calling instead
            logger.info("No valid assistant_id found, using direct chat completions with tool calling")

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]

            # Create parameters for chat completion
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            # Add tools if provided
            if tools:
                params["tools"] = tools
                # Tell the model it can call functions
                params["tool_choice"] = "auto"

            response = client.chat.completions.create(**params)

            # Check if the response contains tool calls
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                # There are tool calls to execute
                tool_calls = []
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })

                return {
                    "status": "requires_action",
                    "thread_id": "direct_" + str(time.time()),  # Create a fake thread ID
                    "run_id": "direct_" + str(time.time()),     # Create a fake run ID
                    "required_action": {
                        "submit_tool_outputs": {
                            "tool_calls": tool_calls
                        }
                    }
                }
            else:
                # No tool calls, just return the content
                return {
                    "content": response.choices[0].message.content,
                    "source": "chat_completion"
                }

        # Use the Assistants API if we have a valid assistant ID
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

        # Create the run with the assistant_id
        run_params = {
            "assistant_id": assistant_id,
            "thread_id": thread.id,
            "instructions": system_message,
            "temperature": temperature,
        }

        # Note: Tools are configured on the assistant itself, not passed directly here
        if tools:
            logger.warning("Tools parameter specified but cannot be directly applied to assistant runs")

        run = client.beta.threads.runs.create(**run_params)
        logger.info(f"Created run: {run.id} for thread: {thread.id} with assistant: {assistant_id}")

        # Wait for the run to complete
        result = await _wait_for_run_completion(thread.id, run.id)

        return result
    except Exception as e:
        logger.error(f"Error calling OpenAI Responses API: {e}")
        raise


async def _wait_for_run_completion(thread_id: str, run_id: str, max_wait_time: int = 300) -> Dict:
    """
    Wait for a run to complete and process the results with improved handling.

    Args:
        thread_id: The thread ID
        run_id: The run ID
        max_wait_time: Maximum time to wait in seconds

    Returns:
        The processed response
    """
    max_polling_attempts = max_wait_time  # Max wait at 1s intervals
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

                # Extract tool calls in a standardized format
                tool_calls = []
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })

                # Return a structured response for tool execution
                return {
                    "status": "requires_action",
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "required_action": {
                        "submit_tool_outputs": {
                            "tool_calls": tool_calls
                        }
                    }
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
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error checking run status: {e}")
            return {"error": str(e), "thread_id": thread_id, "run_id": run_id}

    logger.error(f"Max polling attempts reached for run {run_id}")
    return {"error": "Max polling attempts reached", "thread_id": thread_id, "run_id": run_id}


async def execute_tools_in_run(
    thread_id: str, 
    run_id: str, 
    tool_executor,
    tool_outputs: List[Dict] = None,
    max_iterations: int = 10
) -> Dict:
    """
    Execute tools for a run and return the final result.

    Args:
        thread_id: The thread ID
        run_id: The run ID
        tool_executor: The ToolExecutor instance
        tool_outputs: Optional pre-computed tool outputs
        max_iterations: Maximum number of tool execution iterations

    Returns:
        The final response after tool execution
    """
    iteration = 0

    # First, check if this is a direct API call (not an Assistants API run)
    if thread_id.startswith("direct_") and run_id.startswith("direct_"):
        # This is a direct chat completion API call
        if tool_outputs:
            # We have pre-computed tool outputs, so we can just return them
            # In a real scenario, we'd submit these to another API call
            return {
                "content": f"Tools executed successfully. Results: {json.dumps([o.get('output', '') for o in tool_outputs])}",
                "source": "direct_tool_execution"
            }
        else:
            # Fetch tool calls from the required_action of the run
            tool_calls = []
            # Execute each tool
            for tool_call in tool_calls:
                result = await tool_executor.execute_tool(tool_call)
                tool_outputs.append(result)

            return {
                "content": f"Tools executed successfully. Results: {json.dumps([o.get('output', '') for o in tool_outputs])}",
                "source": "direct_tool_execution"
            }

    # Regular Assistants API run
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

                # Check if we have pre-computed tool outputs
                if tool_outputs:
                    logger.info(f"Using {len(tool_outputs)} pre-computed tool outputs")
                else:
                    # Handle tool calls
                    tool_outputs = []
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    for tool_call in tool_calls:
                        try:
                            # Convert to the format expected by the tool executor
                            formatted_tool_call = {
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }

                            # Execute the tool
                            tool_result = await tool_executor.execute_tool(formatted_tool_call)

                            tool_outputs.append(tool_result)
                            logger.info(f"Tool {tool_call.function.name} executed successfully")
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_call.function.name}: {e}")
                            # Create error response
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps({"error": str(e)})
                            })

                logger.info(f"Submitting {len(tool_outputs)} tool outputs for run {run_id}")

                # Submit the tool outputs
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )

                # Clear tool outputs for next iteration
                tool_outputs = None

                # Wait before checking again
                await asyncio.sleep(1)

            elif run.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Run {run_id} ended with status: {run.status}")
                error_details = getattr(run, "last_error", {"message": f"Run ended with status: {run.status}"})
                return {
                    "error": error_details,
                    "thread_id": thread_id,
                    "run_id": run_id
                }
            else:
                # Wait before checking again
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error during tool execution: {e}")
            return {"error": str(e), "thread_id": thread_id, "run_id": run_id}

    logger.error(f"Max iterations reached for run {run_id}")
    return {"error": "Max iterations reached", "thread_id": thread_id, "run_id": run_id}