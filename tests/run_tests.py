#!/usr/bin/env python

"""
Test runner for ATIA project.

This script runs all relevant tests or specific test categories based on command-line arguments.
"""

import argparse
import subprocess
import sys
import time


def run_command(command, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n{'=' * 80}")
        print(f"Running {description}")
        print(f"{'=' * 80}\n")

    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True)
        end_time = time.time()
        print(f"\n✅ Command completed successfully in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}")
        return False


def run_core_component_tests():
    """Run tests for core components (Phase 1-2)."""
    tests = [
        "tests/test_agent_core.py",
        "tests/test_api_discovery.py",
        "tests/test_need_identifier.py",
        "tests/test_doc_processor.py",
        "tests/test_account_manager.py",
        "tests/test_function_builder.py",
        "tests/test_tool_registry.py",
    ]
    return run_command(f"python -m pytest {' '.join(tests)} -v", "Core Component Tests")


def run_responses_api_tests():
    """Run tests for Responses API integration (Phase 3)."""
    tests = [
        "tests/test_responses_api.py",
    ]
    return run_command(f"python -m pytest {' '.join(tests)} -v", "Responses API Tests")


def run_end_to_end_tests():
    """Run end-to-end integration tests."""
    tests = [
        "tests/test_end_to_end_responses_api.py",
        "tests/test_responses_api_error_handling.py",
    ]
    return run_command(f"python -m pytest {' '.join(tests)} -v", "End-to-End Tests")


def run_manual_tests():
    """Run manual test scripts."""
    scripts = [
        "test_responses_api.py",
        "benchmark_atia.py",
    ]
    success = True
    for script in scripts:
        if not run_command(f"python {script}", f"Manual Test: {script}"):
            success = False
    return success


def run_all_tests():
    """Run all tests."""
    success = True

    # Run pytest tests
    if not run_command("python -m pytest", "All pytest Tests"):
        success = False

    # Run manual test scripts
    if not run_manual_tests():
        success = False

    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run ATIA tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--core", action="store_true", help="Run core component tests")
    parser.add_argument("--responses", action="store_true", help="Run Responses API tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--manual", action="store_true", help="Run manual test scripts")

    args = parser.parse_args()

    # If no arguments, run all tests
    if not any([args.all, args.core, args.responses, args.e2e, args.manual]):
        args.all = True

    success = True

    if args.all:
        success = run_all_tests() and success
    else:
        if args.core:
            success = run_core_component_tests() and success

        if args.responses:
            success = run_responses_api_tests() and success

        if args.e2e:
            success = run_end_to_end_tests() and success

        if args.manual:
            success = run_manual_tests() and success

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())