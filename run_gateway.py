#!/usr/bin/env python
"""
Run the ATIA API Gateway.

This script provides a command-line interface for starting the ATIA API Gateway.
"""

import argparse
import logging
import os
import sys
from typing import Optional

from atia.api_gateway.gateway import run_gateway
from atia.config import settings


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.

    Args:
        log_level: Log level
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        numeric_level = logging.INFO

    log_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }

    # Add file handler if log file is specified
    if log_file:
        log_config['filename'] = log_file
        log_config['filemode'] = 'a'

    # Configure logging
    logging.basicConfig(**log_config)

    # Set OpenAI logging level to WARNING to reduce noise
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Set urllib3 logging level to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Set uvicorn access logs to WARNING to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the ATIA API Gateway")

    parser.add_argument(
        "--host", 
        type=str,
        default=settings.api_gateway_host,
        help=f"Host to bind to (default: {settings.api_gateway_host})"
    )

    parser.add_argument(
        "--port", 
        type=int,
        default=settings.api_gateway_port,
        help=f"Port to bind to (default: {settings.api_gateway_port})"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Log level (default: {settings.log_level})"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: None, logs to console)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (default: False)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set environment variables based on arguments
    if args.host:
        os.environ["API_GATEWAY_HOST"] = args.host

    if args.port:
        os.environ["API_GATEWAY_PORT"] = str(args.port)

    if args.debug:
        os.environ["DEBUG"] = "True"

    # Set up logging
    setup_logging(args.log_level, args.log_file)

    # Log startup message
    logging.info(f"Starting ATIA API Gateway on {args.host}:{args.port}")

    try:
        # Run the gateway
        run_gateway()
    except Exception as e:
        logging.error(f"Error running API Gateway: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()