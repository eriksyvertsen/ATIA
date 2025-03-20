#!/usr/bin/env python

"""
Public APIs Diagnostic Tool
Test different endpoints and configurations for the Public APIs directory
"""

import asyncio
import aiohttp
import json
import sys

async def test_endpoint(url, description=None):
    """Test a specific API endpoint."""
    description = description or url
    print(f"\nTesting {description}...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                status = response.status
                print(f"Status: {status}")

                if status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, dict):
                            print(f"✅ Success - received JSON response with {len(data.keys())} top-level keys")
                            if 'entries' in data:
                                print(f"  Found {len(data['entries'])} API entries")
                            print("  Response keys:", ", ".join(data.keys()))
                        else:
                            print(f"✅ Success - received JSON response (not an object)")
                        return True, data
                    except json.JSONDecodeError:
                        text = await response.text()
                        print(f"⚠️ Received non-JSON response: {text[:100]}...")
                        return False, None
                else:
                    text = await response.text()
                    print(f"❌ Failed with status {status}: {text[:100]}...")
                    return False, None
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Connection error: {e}")
        return False, None
    except asyncio.TimeoutError:
        print(f"❌ Request timed out")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None

async def check_public_apis_service():
    """Comprehensive check of the Public APIs service."""
    print("=== Public APIs Service Diagnostic ===")

    # List of potential endpoints to try
    endpoints = [
        ("https://api.publicapis.org/entries", "Standard entries endpoint"),
        ("https://api.publicapis.org/", "Root endpoint"),
        ("https://api.publicapis.org/random", "Random API endpoint"),
        ("https://api.publicapis.org/categories", "Categories endpoint"),
        ("https://api.publicapis.org/health", "Health check endpoint"),
        ("https://www.publicapis.org/", "Website instead of API"),
        ("https://api.publicapis.io/entries", "Alternative domain (.io)"),
        ("https://public-apis-api.herokuapp.com/api/v1/apis/entries", "Deprecated Heroku endpoint"),
        ("https://github.com/public-apis/public-apis", "GitHub repository")
    ]

    # Test all endpoints
    working_endpoints = []

    for url, description in endpoints:
        success, data = await test_endpoint(url, description)
        if success:
            working_endpoints.append((url, description, data))

    # Summary
    print("\n=== SUMMARY ===")
    if working_endpoints:
        print(f"Found {len(working_endpoints)} working endpoints:")
        for url, description, _ in working_endpoints:
            print(f"- {description}: {url}")

        # Check for any entry data to determine which is the real API endpoint
        for url, description, data in working_endpoints:
            if isinstance(data, dict) and 'entries' in data and data['entries']:
                print(f"\n✅ RECOMMENDED ENDPOINT: {url}")
                print(f"  This endpoint has actual API entries data.")
                return url
    else:
        print("❌ No working endpoints found.")
        print("  The service may be down or has changed significantly.")

    return None

async def main():
    """Main entry point."""
    working_endpoint = await check_public_apis_service()

    if working_endpoint:
        print("\n=== SUGGESTED SOLUTION ===")
        print(f"Update your code to use: {working_endpoint}")
    else:
        print("\n=== SUGGESTED SOLUTION ===")
        print("The Public APIs service appears to be down or significantly changed.")
        print("Consider using only these alternatives:")
        print("1. APIs.guru (https://api.apis.guru/v2/list.json)")
        print("2. RapidAPI (with direct IP addressing)")
        print("3. Your fallback hardcoded APIs")

if __name__ == "__main__":
    asyncio.run(main())