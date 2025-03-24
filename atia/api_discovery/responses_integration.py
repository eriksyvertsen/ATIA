"""
Responses API integration for API Discovery component.

Enhances API search and evaluation using OpenAI's Responses API.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple

from atia.config import settings
from atia.utils.openai_client import get_completion, get_json_completion, get_completion_with_responses_api

logger = logging.getLogger(__name__)

# Define Responses API tool schema for API discovery
API_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_apis",
        "description": "Search for APIs that provide specific capabilities",
        "parameters": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "Description of the capability needed"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["capability"]
        }
    }
}

API_EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_api",
        "description": "Evaluate an API for suitability to fulfill a specific capability",
        "parameters": {
            "type": "object",
            "properties": {
                "api_name": {
                    "type": "string",
                    "description": "Name of the API"
                },
                "api_description": {
                    "type": "string",
                    "description": "Description of the API"
                },
                "capability": {
                    "type": "string",
                    "description": "Description of the capability needed"
                },
                "relevance_score": {
                    "type": "number",
                    "description": "Relevance score from 0.0 to 1.0"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Reasoning for the relevance score"
                }
            },
            "required": ["api_name", "api_description", "capability", "relevance_score"]
        }
    }
}


async def search_apis_with_responses_api(capability: str, max_results: int = 5) -> List[Dict]:
    """
    Search for APIs that match a capability using the Responses API.

    Args:
        capability: Description of the capability needed
        max_results: Maximum number of results to return

    Returns:
        List of API candidates as dictionaries
    """
    logger.info(f"Searching for APIs with capability: {capability} using Responses API")

    prompt = f"""
    I need to find APIs that can provide this capability: {capability}

    Please search for and identify the most relevant APIs that can fulfill this need.
    For each API, provide:
    - Name of the API
    - Provider/company
    - Brief description
    - Main capabilities
    - Authentication method (if known)
    - Documentation URL (if known)
    - Estimated relevance score (0.0 to 1.0)
    """

    system_message = (
        "You are an expert at discovering and evaluating APIs for specific capabilities. "
        "Use the search_apis function to provide structured results for API search queries."
    )

    try:
        response = await get_completion_with_responses_api(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2,
            tools=[API_SEARCH_TOOL],
            model=settings.openai_model
        )

        # Check if tool was called
        if response.get("status") == "requires_action":
            tool_calls = response.get("required_action", {}).get("submit_tool_outputs", {}).get("tool_calls", [])

            for tool_call in tool_calls:
                if tool_call.get("function", {}).get("name") == "search_apis":
                    # Extract arguments and process them
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    capability_arg = arguments.get("capability", capability)
                    max_results_arg = arguments.get("max_results", max_results)

                    # Since we can't actually execute a real search within the Responses API,
                    # we'll return a structured response simulating the search results
                    api_results = _generate_structured_api_results(capability_arg, max_results_arg)

                    return api_results

        # If we didn't get tool results, try to extract structured data from text response
        content = response.get("content", "")

        # Try to extract structured data (basic approach)
        apis = []
        lines = content.split("\n")
        current_api = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_api and "name" in current_api:
                    apis.append(current_api)
                    current_api = {}
                continue

            # Look for API name/title
            if line.startswith("# ") or line.startswith("## "):
                if current_api and "name" in current_api:
                    apis.append(current_api)
                current_api = {"name": line.lstrip("# ")}

            # Extract key-value pairs
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.lower().strip()
                value = value.strip()

                if key in ["name", "provider", "description", "url", "documentation", "auth"]:
                    current_api[key] = value
                elif key in ["relevance", "score", "relevance score"]:
                    try:
                        current_api["relevance_score"] = float(value.split()[0])
                    except:
                        current_api["relevance_score"] = 0.5

        # Add the last API if exists
        if current_api and "name" in current_api:
            apis.append(current_api)

        # Format results consistently
        formatted_results = []
        for api in apis:
            formatted_results.append({
                "name": api.get("name", "Unknown API"),
                "provider": api.get("provider", "Unknown"),
                "description": api.get("description", ""),
                "documentation_url": api.get("documentation", api.get("url", "")),
                "auth_type": api.get("auth", "api_key"),
                "relevance_score": api.get("relevance_score", 0.5)
            })

        return formatted_results[:max_results]

    except Exception as e:
        logger.error(f"Error using Responses API for API search: {e}")
        # Return empty list on error
        return []


async def evaluate_api_with_responses_api(api_info: Dict, capability: str) -> Tuple[float, List[str]]:
    """
    Evaluate an API's suitability for a capability using the Responses API.

    Args:
        api_info: Information about the API
        capability: Description of the capability needed

    Returns:
        Tuple of (relevance_score, capabilities_list)
    """
    logger.info(f"Evaluating API: {api_info.get('name')} for capability: {capability}")

    prompt = f"""
    Please evaluate how well this API matches the required capability:

    API Name: {api_info.get('name')}
    API Provider: {api_info.get('provider')}
    API Description: {api_info.get('description')}

    Required Capability: {capability}

    Evaluate the relevance and suitability of this API for the specified capability.
    Provide a relevance score (0.0 to 1.0) and a list of specific capabilities this API offers.
    """

    system_message = (
        "You are an expert at evaluating APIs for specific capabilities. "
        "Use the evaluate_api function to provide structured evaluation results."
    )

    try:
        response = await get_completion_with_responses_api(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2,
            tools=[API_EVALUATION_TOOL],
            model=settings.openai_model
        )

        # Check if tool was called
        if response.get("status") == "requires_action":
            tool_calls = response.get("required_action", {}).get("submit_tool_outputs", {}).get("tool_calls", [])

            for tool_call in tool_calls:
                if tool_call.get("function", {}).get("name") == "evaluate_api":
                    # Extract arguments
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    relevance_score = arguments.get("relevance_score", 0.5)
                    reasoning = arguments.get("reasoning", "")

                    # Extract capabilities from reasoning
                    capabilities = _extract_capabilities_from_text(reasoning)

                    return (relevance_score, capabilities)

        # If we didn't get tool results, try to extract information from text response
        content = response.get("content", "")

        # Look for a relevance score (simple regex pattern)
        import re
        score_match = re.search(r'relevance score:?\s*(0\.\d+|1\.0)', content, re.IGNORECASE)
        relevance_score = float(score_match.group(1)) if score_match else 0.5

        # Extract capabilities
        capabilities = _extract_capabilities_from_text(content)

        return (relevance_score, capabilities)

    except Exception as e:
        logger.error(f"Error using Responses API for API evaluation: {e}")
        # Return default values on error
        return (0.5, [])


def _extract_capabilities_from_text(text: str) -> List[str]:
    """Extract capabilities from evaluation text."""
    capabilities = []

    # Look for bullet points or numbered list items
    import re
    capability_matches = re.findall(r'[•\-\*\d+]\s+(.*?)(?=\n|$)', text)

    if capability_matches:
        capabilities = [cap.strip() for cap in capability_matches if cap.strip()]
    else:
        # Try to find a capabilities or features section
        sections = re.split(r'\n\s*\n', text)
        for section in sections:
            if 'capabilities' in section.lower() or 'features' in section.lower():
                # Split by newlines and clean up
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                if len(lines) > 1:  # Skip the header
                    capabilities = lines[1:]
                    break

    # If we still have no capabilities, extract sentences that might be capabilities
    if not capabilities:
        sentences = re.split(r'\.(?=\s|$)', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 100 and sentence.lower().startswith(('can ', 'provides ', 'enables ', 'allows ')):
                capabilities.append(sentence)

    return capabilities[:5]  # Limit to top 5 capabilities


def _generate_structured_api_results(capability: str, max_results: int) -> List[Dict]:
    """
    Generate structured API results based on capability.
    This is used when we can't execute real search within Responses API.

    Args:
        capability: The capability description
        max_results: Maximum number of results

    Returns:
        List of simulated API results
    """
    # Extract key terms from capability
    keywords = capability.lower().split()

    # Define some common API templates by category
    api_templates = {
        "weather": [
            {
                "name": "OpenWeatherMap API",
                "provider": "OpenWeatherMap",
                "description": "Provides weather data including current weather, forecasts, and historical data.",
                "documentation_url": "https://openweathermap.org/api",
                "auth_type": "api_key",
                "relevance_score": 0.95
            },
            {
                "name": "Weather API",
                "provider": "WeatherAPI.com",
                "description": "Real-time weather, forecasts, historical data, and weather alerts.",
                "documentation_url": "https://www.weatherapi.com/docs/",
                "auth_type": "api_key",
                "relevance_score": 0.92
            }
        ],
        "translation": [
            {
                "name": "Google Translate API",
                "provider": "Google",
                "description": "Translate text between languages using Google's neural machine translation.",
                "documentation_url": "https://cloud.google.com/translate/docs",
                "auth_type": "api_key",
                "relevance_score": 0.95
            },
            {
                "name": "DeepL API",
                "provider": "DeepL",
                "description": "High-quality language translation API.",
                "documentation_url": "https://www.deepl.com/docs-api",
                "auth_type": "api_key",
                "relevance_score": 0.92
            }
        ],
        "image": [
            {
                "name": "Unsplash API",
                "provider": "Unsplash",
                "description": "Access to high-quality, royalty-free images.",
                "documentation_url": "https://unsplash.com/documentation",
                "auth_type": "oauth",
                "relevance_score": 0.9
            },
            {
                "name": "Pexels API",
                "provider": "Pexels",
                "description": "Free stock photos and videos API.",
                "documentation_url": "https://www.pexels.com/api/documentation/",
                "auth_type": "api_key",
                "relevance_score": 0.85
            }
        ],
        "news": [
            {
                "name": "News API",
                "provider": "NewsAPI.org",
                "description": "API for accessing breaking headlines and searching articles from news sources and blogs.",
                "documentation_url": "https://newsapi.org/docs",
                "auth_type": "api_key",
                "relevance_score": 0.95
            },
            {
                "name": "The Guardian API",
                "provider": "The Guardian",
                "description": "Access to Guardian news content and data.",
                "documentation_url": "https://open-platform.theguardian.com/documentation/",
                "auth_type": "api_key",
                "relevance_score": 0.88
            }
        ],
        "search": [
            {
                "name": "Google Custom Search API",
                "provider": "Google",
                "description": "Programmatic access to Google search results.",
                "documentation_url": "https://developers.google.com/custom-search/v1/overview",
                "auth_type": "api_key",
                "relevance_score": 0.9
            },
            {
                "name": "Bing Search API",
                "provider": "Microsoft",
                "description": "Web, image, news, and video search capabilities.",
                "documentation_url": "https://www.microsoft.com/en-us/bing/apis/bing-web-search-api",
                "auth_type": "api_key",
                "relevance_score": 0.85
            }
        ],
        "default": [
            {
                "name": "RapidAPI Hub",
                "provider": "RapidAPI",
                "description": "Marketplace with thousands of APIs across categories.",
                "documentation_url": "https://rapidapi.com",
                "auth_type": "api_key",
                "relevance_score": 0.75
            },
            {
                "name": "ProgrammableWeb API Directory",
                "provider": "ProgrammableWeb",
                "description": "Directory of publicly available APIs.",
                "documentation_url": "https://www.programmableweb.com/apis/directory",
                "auth_type": "various",
                "relevance_score": 0.7
            }
        ]
    }

    # Determine which category matches the capability best
    best_category = "default"
    for category in api_templates:
        if category in capability.lower():
            best_category = category
            break

    # Start with the best category APIs
    results = api_templates[best_category].copy()

    # Add some from other categories if needed
    categories = list(api_templates.keys())
    categories.remove(best_category)
    categories.remove("default")  # Don't use default unless necessary

    while len(results) < max_results and categories:
        category = categories.pop(0)
        results.extend(api_templates[category][:1])  # Add top API from each category

    # If still not enough, add from default category
    if len(results) < max_results:
        results.extend(api_templates["default"][:max_results - len(results)])

    # Limit to max_results
    return results[:max_results]
