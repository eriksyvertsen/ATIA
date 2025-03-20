#!/usr/bin/env python

"""
Network Diagnostic Tool for ATIA
Tests DNS resolution and connection to RapidAPI
"""

import asyncio
import socket
import aiohttp
import time
import sys

async def test_dns_resolution(hostname="api.rapidapi.com"):
    """Test if we can resolve a hostname to an IP address."""
    print(f"\nTesting DNS resolution for {hostname}...")
    try:
        # Try standard DNS resolution
        ip_address = socket.gethostbyname(hostname)
        print(f"✅ DNS resolution succeeded: {hostname} -> {ip_address}")
        return ip_address
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        return None

async def test_http_connection(url="https://api.rapidapi.com/ping", headers=None):
    """Test if we can connect to an HTTP endpoint."""
    print(f"\nTesting HTTP connection to {url}...")
    if headers is None:
        headers = {}

    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.get(url, headers=headers, timeout=15) as response:
                duration = time.time() - start_time
                print(f"✅ HTTP connection succeeded: Status {response.status} in {duration:.2f}s")
                return True
    except aiohttp.ClientConnectorError as e:
        print(f"❌ HTTP connection failed (connection error): {e}")
    except aiohttp.ClientError as e:
        print(f"❌ HTTP connection failed (client error): {e}")
    except asyncio.TimeoutError:
        print(f"❌ HTTP connection timed out after 15 seconds")
    except Exception as e:
        print(f"❌ HTTP connection failed (unexpected error): {e}")
    return False

async def test_rapidapi_connection(api_key=None):
    """Test connection to RapidAPI specifically."""
    print("\nTesting RapidAPI connection...")

    if not api_key:
        print("⚠️ No RapidAPI key provided, performing basic connectivity test")

    # Try to connect to RapidAPI's documentation site first (doesn't require auth)
    basic_result = await test_http_connection("https://docs.rapidapi.com/")

    if api_key:
        # Try actual API endpoint with authentication
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "api.rapidapi.com"
        }

        api_result = await test_http_connection("https://api.rapidapi.com/v2/ping", headers)

        if api_result:
            print("\n✅ RapidAPI connectivity test successful with authentication")
        else:
            print("\n❌ RapidAPI connectivity test failed with authentication")

        return api_result
    else:
        # Just return the basic test result
        if basic_result:
            print("\n✅ Basic RapidAPI connectivity test successful (documentation site)")
        else:
            print("\n❌ Basic RapidAPI connectivity test failed (documentation site)")

        return basic_result

async def test_alternative_apis():
    """Test connectivity to alternative API directories."""
    print("\nTesting connectivity to alternative API directories...")

    apis = [
        "https://api.apis.guru/v2/list.json",  # APIs.guru directory
        "https://api.publicapis.org/entries",  # Public APIs directory
        "https://github.com/public-apis/public-apis"  # GitHub repo with API listings
    ]

    results = []

    for url in apis:
        result = await test_http_connection(url)
        results.append(result)

    if any(results):
        print("\n✅ At least one alternative API directory is accessible")
    else:
        print("\n❌ All alternative API directories are inaccessible")

    return any(results)

async def main():
    """Run all diagnostic tests."""
    print("=== ATIA Network Diagnostics ===")

    from atia.config import settings

    # Get API key from settings if available
    rapid_api_key = settings.rapid_api_key if hasattr(settings, 'rapid_api_key') else None

    # Run tests
    ip_address = await test_dns_resolution()

    if ip_address:
        # If DNS resolution works, test HTTP connection
        await test_http_connection()

        # Test RapidAPI connection
        await test_rapidapi_connection(rapid_api_key)
    else:
        print("\n⚠️ DNS resolution failed, testing alternative APIs...")
        # If DNS resolution fails, try alternative APIs
        await test_alternative_apis()

        # Suggest direct IP usage
        print("\n💡 Suggestion: Consider using direct IP addresses instead of hostnames if DNS resolution is failing")

    print("\n=== Diagnostic Complete ===")

if __name__ == "__main__":
    asyncio.run(main())