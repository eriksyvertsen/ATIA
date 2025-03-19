#!/usr/bin/env python

"""
Integration test script for ATIA Phase 4.

This script tests the entire ATIA pipeline with emphasis on Phase 4 features.
"""

import asyncio
import time
import json
import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integration_test.log")
    ]
)

logger = logging.getLogger(__name__)

# Import ATIA components
from atia.agent_core import AgentCore
from atia.need_identifier import NeedIdentifier
from atia.api_discovery import APIDiscovery
from atia.doc_processor import DocumentationProcessor
from atia.account_manager import AccountManager
from atia.tool_registry import ToolRegistry
from atia.function_builder import FunctionBuilder
from atia.tool_executor import ToolExecutor
from atia.utils.analytics import dashboard
from atia.utils.cache import ResponseCache
from atia.config import settings
from atia.utils.error_handling import AsyncErrorContext, ATIAError, LLMError


class IntegrationTest:
    """Integration test for ATIA Phase 4."""

    def __init__(self):
        """Initialize the integration test."""
        self.logger = logging.getLogger(__name__)
        self.session_id = f"test-{int(time.time())}"
        self.results = {
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            },
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None
        }

        # Initialize components
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.agent_core = AgentCore(tool_registry=self.tool_registry)
        self.need_identifier = NeedIdentifier()
        self.api_discovery = APIDiscovery()
        self.doc_processor = DocumentationProcessor()
        self.account_manager = AccountManager()
        self.function_builder = FunctionBuilder()
        self.response_cache = ResponseCache()

    async def run_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.

        Returns:
            Test results dictionary
        """
        start_time = time.time()
        self.logger.info("Starting ATIA Phase 4 integration tests")

        try:
            # Test Agent Core with cache
            await self.test_agent_core_with_cache()

            # Test Need Identifier with error handling
            await self.test_need_identifier_with_error_handling()

            # Test API Discovery with evaluation
            await self.test_api_discovery_with_evaluation()

            # Test Documentation Processor with formats
            await self.test_doc_processor_formats()

            # Test full pipeline
            await self.test_full_pipeline()

            # Test analytics dashboard
            await self.test_analytics_dashboard()

            # Test Responses API integration
            await self.test_responses_api_integration()

        except Exception as e:
            self.logger.error(f"Error running integration tests: {e}")

        finally:
            # Calculate test duration
            end_time = time.time()
            duration_seconds = end_time - start_time

            # Update results
            self.results["end_time"] = datetime.now().isoformat()
            self.results["duration_seconds"] = duration_seconds

            # Print summary
            self.logger.info(f"Integration tests completed in {duration_seconds:.2f} seconds")
            self.logger.info(f"Total tests: {self.results['summary']['total']}")
            self.logger.info(f"Passed: {self.results['summary']['passed']}")
            self.logger.info(f"Failed: {self.results['summary']['failed']}")
            self.logger.info(f"Skipped: {self.results['summary']['skipped']}")

            return self.results

    def record_result(self, test_name: str, passed: bool, details: Optional[Dict[str, Any]] = None, 
                    error: Optional[Exception] = None, skipped: bool = False) -> None:
        """
        Record a test result.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            details: Optional test details
            error: Optional exception if the test failed
            skipped: Whether the test was skipped
        """
        result = {
            "name": test_name,
            "passed": passed,
            "skipped": skipped,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        if error:
            result["error"] = {
                "type": error.__class__.__name__,
                "message": str(error)
            }

            if isinstance(error, ATIAError):
                result["error"]["component"] = error.component
                result["error"]["details"] = error.details

        self.results["tests"].append(result)

        # Update summary
        self.results["summary"]["total"] += 1
        if skipped:
            self.results["summary"]["skipped"] += 1
        elif passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1

        # Log result
        if skipped:
            self.logger.info(f"SKIPPED: {test_name}")
        elif passed:
            self.logger.info(f"PASSED: {test_name}")
        else:
            self.logger.error(f"FAILED: {test_name}")
            if error:
                self.logger.error(f"  Error: {error}")

    async def test_agent_core_with_cache(self) -> None:
        """Test Agent Core with caching mechanism."""
        test_name = "Agent Core with Cache"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Use a query that should be cacheable
            query = "What's the weather like in Paris?"

            # First query should miss cache
            start_time = time.time()
            response1 = await self.agent_core.process_query(query)
            first_query_time = time.time() - start_time

            # Check cache statistics before second query
            cache_stats_before = self.response_cache.stats()

            # Second query should hit cache
            start_time = time.time()
            response2 = await self.agent_core.process_query(query)
            second_query_time = time.time() - start_time

            # Check cache statistics after second query
            cache_stats_after = self.response_cache.stats()

            # Verify cache hit
            cache_hit = cache_stats_after["hits"] > cache_stats_before["hits"]

            # Verify responses match
            responses_match = response1 == response2

            # Verify second query was faster
            speed_improvement = first_query_time > second_query_time

            # Test passes if cache was hit, responses match, and second query was faster
            passed = cache_hit and responses_match and speed_improvement

            self.record_result(
                test_name=test_name,
                passed=passed,
                details={
                    "first_query_time_seconds": first_query_time,
                    "second_query_time_seconds": second_query_time,
                    "cache_hit": cache_hit,
                    "responses_match": responses_match,
                    "speed_improvement": speed_improvement,
                    "cache_stats_before": cache_stats_before,
                    "cache_stats_after": cache_stats_after
                }
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )

    async def test_need_identifier_with_error_handling(self) -> None:
        """Test Need Identifier with error handling."""
        test_name = "Need Identifier with Error Handling"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Simulate a situation where the LLM API might fail
            original_openai_api_key = settings.openai_api_key

            # Test with invalid API key to trigger error
            try:
                settings.openai_api_key = "invalid_key"

                # This should fail but be caught by error handling
                async with AsyncErrorContext("need_identifier", "Failed to identify need", 
                                           LLMError, default_return=None):
                    await self.need_identifier.identify_tool_need("Test query")

                # If we got here, the error was handled correctly
                passed = True
                error_details = "Error was caught and handled correctly"

            except Exception as e:
                # If an exception was raised, the error handling didn't work
                passed = False
                error_details = f"Error handling failed: {e}"

            finally:
                # Restore original API key
                settings.openai_api_key = original_openai_api_key

            self.record_result(
                test_name=test_name,
                passed=passed,
                details={
                    "error_handling": error_details
                }
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )

    async def test_api_discovery_with_evaluation(self) -> None:
        """Test API Discovery with evaluation of candidates."""
        test_name = "API Discovery with Evaluation"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Skip if Responses API is disabled
            if settings.disable_responses_api:
                self.record_result(
                    test_name=test_name,
                    passed=True,
                    skipped=True,
                    details={"reason": "Responses API is disabled"}
                )
                return

            # Test with a clear capability description
            capability = "Translate text from English to French"

            # Search for APIs with evaluation
            api_candidates = await self.api_discovery.search_for_api(
                capability_description=capability,
                evaluate=True
            )

            # Verify we got results
            has_results = len(api_candidates) > 0

            # Check if relevance scores are set
            has_relevance_scores = all(hasattr(c, "relevance_score") and c.relevance_score > 0 
                                     for c in api_candidates)

            # Check if capabilities list is populated
            has_capabilities = any(hasattr(c, "capabilities") and c.capabilities 
                                 for c in api_candidates)

            # Test passes if we have results with relevance scores
            passed = has_results and has_relevance_scores

            self.record_result(
                test_name=test_name,
                passed=passed,
                details={
                    "capability": capability,
                    "num_results": len(api_candidates),
                    "has_relevance_scores": has_relevance_scores,
                    "has_capabilities": has_capabilities,
                    "top_candidate": api_candidates[0].name if has_results else None,
                    "top_relevance": api_candidates[0].relevance_score if has_results else None
                }
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )

    async def test_doc_processor_formats(self) -> None:
        """Test Documentation Processor with different formats."""
        test_name = "Documentation Processor Formats"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Test with different document formats
            formats = {
                "openapi": """
                {
                    "openapi": "3.0.0",
                    "info": {
                        "title": "Test API",
                        "description": "API for testing",
                        "version": "1.0.0"
                    },
                    "servers": [
                        {
                            "url": "https://api.example.com/v1"
                        }
                    ],
                    "paths": {
                        "/test": {
                            "get": {
                                "summary": "Test endpoint",
                                "description": "Endpoint for testing",
                                "parameters": [
                                    {
                                        "name": "param",
                                        "in": "query",
                                        "description": "Test parameter",
                                        "required": true,
                                        "schema": {
                                            "type": "string"
                                        }
                                    }
                                ],
                                "responses": {
                                    "200": {
                                        "description": "Successful response",
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                    "type": "object",
                                                    "properties": {
                                                        "result": {
                                                            "type": "string"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                """,
                "markdown": """
                # Test API

                API for testing.

                ## Base URL

                https://api.example.com/v1

                ## Endpoints

                ### GET /test

                Test endpoint

                #### Parameters

                - `param` (query, required): Test parameter

                #### Response

                ```json
                {
                    "result": "string"
                }
                ```
                """,
                "html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Test API</title>
                </head>
                <body>
                    <h1>Test API</h1>
                    <p>API for testing.</p>

                    <h2>Base URL</h2>
                    <pre>https://api.example.com/v1</pre>

                    <h2>Endpoints</h2>

                    <h3>GET /test</h3>
                    <p>Test endpoint</p>

                    <h4>Parameters</h4>
                    <ul>
                        <li><code>param</code> (query, required): Test parameter</li>
                    </ul>

                    <h4>Response</h4>
                    <pre>
                    {
                        "result": "string"
                    }
                    </pre>
                </body>
                </html>
                """
            }

            results = {}

            for format_name, content in formats.items():
                # Process the document
                doc_type = format_name
                api_info = await self.doc_processor.process_documentation(content, doc_type)

                # Check if processing was successful
                success = (
                    api_info is not None and
                    hasattr(api_info, "base_url") and
                    api_info.base_url == "https://api.example.com/v1" and
                    hasattr(api_info, "endpoints") and
                    len(api_info.endpoints) > 0
                )

                results[format_name] = {
                    "success": success,
                    "base_url": api_info.base_url if api_info else None,
                    "num_endpoints": len(api_info.endpoints) if api_info and hasattr(api_info, "endpoints") else 0
                }

            # Test passes if all formats were processed successfully
            passed = all(result["success"] for result in results.values())

            self.record_result(
                test_name=test_name,
                passed=passed,
                details={
                    "formats_tested": list(formats.keys()),
                    "results": results
                }
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )

    async def test_full_pipeline(self) -> None:
        """Test the full ATIA pipeline."""
        test_name = "Full Pipeline Integration"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Use a query that should trigger tool need
            query = "I need to translate a paragraph from English to Spanish"

            pipeline_successful = True
            details = {}

            # Step 1: Process with agent core
            self.logger.info("Step 1: Process with agent core")
            agent_response = await self.agent_core.process_query(query)
            details["agent_response"] = agent_response is not None

            # Step 2: Identify tool need
            self.logger.info("Step 2: Identify tool need")
            tool_need = await self.need_identifier.identify_tool_need(query)
            details["tool_need"] = tool_need is not None

            if not tool_need:
                pipeline_successful = False
                details["failure_point"] = "Tool need not identified"
            else:
                # Step 3: Discover APIs
                self.logger.info("Step 3: Discover APIs")
                api_candidates = await self.api_discovery.search_for_api(
                    tool_need.description,
                    num_results=3
                )
                details["api_candidates"] = len(api_candidates) > 0

                if not api_candidates:
                    pipeline_successful = False
                    details["failure_point"] = "No API candidates found"
                else:
                    details["top_api"] = api_candidates[0].name

                    # The rest of the pipeline would continue here in a full implementation
                    # For this test, we'll consider the pipeline successful if we get this far

            self.record_result(
                test_name=test_name,
                passed=pipeline_successful,
                details=details
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )

    async def test_analytics_dashboard(self) -> None:
        """Test the analytics dashboard."""
        test_name = "Analytics Dashboard"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Skip if analytics is disabled
            if not settings.enable_analytics:
                self.record_result(
                    test_name=test_name,
                    passed=True,
                    skipped=True,
                    details={"reason": "Analytics is disabled"}
                )
                return

            # Get current metrics
            metrics = dashboard.get_metrics()

            # Generate dashboard HTML
            from atia.utils.analytics import generate_dashboard_html
            html = generate_dashboard_html()

            # Test passes if we got metrics and HTML
            passed = metrics is not None and html and len(html) > 0

            self.record_result(
                test_name=test_name,
                passed=passed,
                details={
                    "metrics_available": metrics is not None,
                    "html_generated": html is not None and len(html) > 0,
                    "html_length": len(html) if html else 0
                }
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )

    async def test_responses_api_integration(self) -> None:
        """Test the Responses API integration."""
        test_name = "Responses API Integration"
        self.logger.info(f"Running test: {test_name}")

        try:
            # Skip if Responses API is disabled
            if settings.disable_responses_api:
                self.record_result(
                    test_name=test_name,
                    passed=True,
                    skipped=True,
                    details={"reason": "Responses API is disabled"}
                )
                return

            # Use a query that should NOT trigger tool execution
            query = "Tell me a short joke about programming"

            # Process with Responses API
            response = await self.agent_core.process_query_with_responses_api(query)

            # Check if we got a response
            has_response = (
                response is not None and
                isinstance(response, dict) and
                "content" in response and
                response["content"]
            )

            # Test passes if we got a response
            passed = has_response

            self.record_result(
                test_name=test_name,
                passed=passed,
                details={
                    "query": query,
                    "has_response": has_response,
                    "response_keys": list(response.keys()) if response else None,
                    "content_length": len(response.get("content", "")) if has_response else 0
                }
            )

        except Exception as e:
            self.record_result(
                test_name=test_name,
                passed=False,
                error=e
            )


async def main():
    """Run the integration tests."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="ATIA Phase 4 Integration Tests")
    parser.add_argument("--output", "-o", help="Output file for test results (JSON)")
    parser.add_argument("--format", "-f", choices=["json", "text"], default="text",
                      help="Output format (default: text)")

    args = parser.parse_args()

    # Run the tests
    test = IntegrationTest()
    results = await test.run_tests()

    # Output results
    if args.format == "json" or args.output:
        # Write JSON results
        json_results = json.dumps(results, indent=2)

        if args.output:
            # Write to file
            with open(args.output, "w") as f:
                f.write(json_results)
            print(f"Test results written to {args.output}")
        else:
            # Print to stdout
            print(json_results)
    else:
        # Print text summary
        print("\n=== ATIA Phase 4 Integration Test Results ===\n")

        print(f"Total tests: {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Skipped: {results['summary']['skipped']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds\n")

        print("Test Details:")
        for i, test in enumerate(results["tests"], 1):
            status = "SKIPPED" if test["skipped"] else "PASSED" if test["passed"] else "FAILED"
            print(f"{i}. {test['name']}: {status}")

            if not test["passed"] and not test["skipped"] and "error" in test:
                print(f"   Error: {test['error']['type']}: {test['error']['message']}")

        print()

    # Return exit code based on test results
    return 0 if results["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)