#!/usr/bin/env python

"""
Benchmark script for comparing standard API vs Responses API performance.
"""

import asyncio
import time
import statistics
import json
from typing import List, Dict, Any

from atia.agent_core import AgentCore
from atia.tool_registry import ToolRegistry
from atia.function_builder import FunctionBuilder, FunctionDefinition
from atia.function_builder.models import ApiType, ParameterType, FunctionParameter
from atia.tool_executor import ToolExecutor


async def register_test_function(tool_registry):
    """Register a test function in the tool registry."""
    # Create a function builder
    function_builder = FunctionBuilder()

    # Create a simple echo function
    function_def = FunctionDefinition(
        name="echo_function",
        description="Echo back the input message",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="message",
                param_type=ParameterType.STRING,
                description="The message to echo back",
                required=True
            )
        ],
        code="""
async def echo_function(message: str):
    \"\"\"
    Echo back the input message.

    Args:
        message: The message to echo back

    Returns:
        Dict with the echoed message
    \"\"\"
    return {"message": message, "status": "success"}
""",
        endpoint="/echo",
        method="GET",
        tags=["test", "echo"]
    )

    # Register the function
    tool = await tool_registry.register_function(function_def)
    return tool


async def run_standard_api_benchmark(iterations: int = 5):
    """Benchmark the standard API."""
    agent = AgentCore()

    query = "What's the weather like in Paris?"
    results = []

    print(f"\n[Standard API Benchmark - {iterations} iterations]")
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")
        start_time = time.time()

        response = await agent.process_query(query)

        end_time = time.time()
        duration = end_time - start_time
        results.append(duration)

        print(f"Duration: {duration:.2f} seconds")

    return {
        "api_type": "Standard API",
        "iterations": iterations,
        "mean": statistics.mean(results),
        "median": statistics.median(results),
        "min": min(results),
        "max": max(results),
        "std_dev": statistics.stdev(results) if len(results) > 1 else 0
    }


async def run_responses_api_benchmark(iterations: int = 5):
    """Benchmark the Responses API."""
    # Create the tool registry
    tool_registry = ToolRegistry()

    # Register a test function
    tool = await register_test_function(tool_registry)

    # Create a tool executor
    tool_executor = ToolExecutor(tool_registry)

    # Create an agent with the tool registry
    agent = AgentCore(tool_registry=tool_registry)
    agent.tool_executor = tool_executor

    # Get the tool schema for Responses API
    function_builder = FunctionBuilder()
    function_def = await tool_registry._get_function_definition(tool.function_id)
    tool_schema = function_builder.generate_responses_api_tool_schema(function_def)

    query = "What's the weather like in Paris?"
    results = []

    print(f"\n[Responses API Benchmark - {iterations} iterations]")
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")
        start_time = time.time()

        response = await agent.process_query_with_responses_api(
            query=query,
            tools=[tool_schema]
        )

        end_time = time.time()
        duration = end_time - start_time
        results.append(duration)

        print(f"Duration: {duration:.2f} seconds")

    return {
        "api_type": "Responses API",
        "iterations": iterations,
        "mean": statistics.mean(results),
        "median": statistics.median(results),
        "min": min(results),
        "max": max(results),
        "std_dev": statistics.stdev(results) if len(results) > 1 else 0
    }


def print_benchmark_results(standard_results: Dict, responses_results: Dict):
    """Print benchmark results in a formatted way."""
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    print("\nStandard API:")
    print(f"- Mean: {standard_results['mean']:.2f} seconds")
    print(f"- Median: {standard_results['median']:.2f} seconds")
    print(f"- Min: {standard_results['min']:.2f} seconds")
    print(f"- Max: {standard_results['max']:.2f} seconds")
    print(f"- Std Dev: {standard_results['std_dev']:.2f} seconds")

    print("\nResponses API:")
    print(f"- Mean: {responses_results['mean']:.2f} seconds")
    print(f"- Median: {responses_results['median']:.2f} seconds")
    print(f"- Min: {responses_results['min']:.2f} seconds")
    print(f"- Max: {responses_results['max']:.2f} seconds")
    print(f"- Std Dev: {responses_results['std_dev']:.2f} seconds")

    # Calculate percentage difference
    mean_diff = ((responses_results['mean'] - standard_results['mean']) / standard_results['mean']) * 100
    median_diff = ((responses_results['median'] - standard_results['median']) / standard_results['median']) * 100

    print("\nComparison:")
    print(f"- Mean difference: {mean_diff:.1f}% {'slower' if mean_diff > 0 else 'faster'}")
    print(f"- Median difference: {median_diff:.1f}% {'slower' if median_diff > 0 else 'faster'}")

    print("\nNOTE: These benchmarks include network latency and API processing time.")
    print("=" * 50)


async def main():
    """Run the benchmarks."""
    print("🔍 ATIA API Performance Benchmark")
    print("================================")

    # Run benchmarks with fewer iterations to save API credits
    iterations = 3

    # Run standard API benchmark
    standard_results = await run_standard_api_benchmark(iterations)

    # Run Responses API benchmark
    responses_results = await run_responses_api_benchmark(iterations)

    # Print results
    print_benchmark_results(standard_results, responses_results)

    # Save results to file
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "standard_api": standard_results,
        "responses_api": responses_results
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())