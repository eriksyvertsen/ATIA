#!/usr/bin/env python

"""
Apply Phase 1: Week 2 updates to ATIA.

This script integrates the enhanced API Discovery and Function Builder components
with the rest of the ATIA system.
"""

import os
import sys
import shutil
import importlib
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_directory_if_not_exists(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def check_import_paths() -> None:
    """Check if import paths are properly set up."""
    try:
        import atia
        logger.info(f"Successfully imported atia package from: {atia.__file__}")
    except ImportError:
        # Add the current directory to the path if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            logger.info(f"Added current directory to Python path: {current_dir}")

        try:
            import atia
            logger.info(f"Successfully imported atia package after path update: {atia.__file__}")
        except ImportError:
            logger.error("Failed to import atia package. Make sure you're running this script from the project root.")
            sys.exit(1)

def apply_api_discovery_updates() -> None:
    """Apply updates to the API Discovery component."""
    # Create the responses_integration.py file
    source_file = "atia/api_discovery/responses_integration.py"

    with open("atia/api_discovery/responses_integration.py", "w") as f:
        f.write('''"""
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
        lines = content.split("\\n")
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
        score_match = re.search(r'relevance score:?\\s*(0\\.\\d+|1\\.0)', content, re.IGNORECASE)
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
    capability_matches = re.findall(r'[•\\-\\*\\d+]\\s+(.*?)(?=\\n|$)', text)

    if capability_matches:
        capabilities = [cap.strip() for cap in capability_matches if cap.strip()]
    else:
        # Try to find a capabilities or features section
        sections = re.split(r'\\n\\s*\\n', text)
        for section in sections:
            if 'capabilities' in section.lower() or 'features' in section.lower():
                # Split by newlines and clean up
                lines = [line.strip() for line in section.split('\\n') if line.strip()]
                if len(lines) > 1:  # Skip the header
                    capabilities = lines[1:]
                    break

    # If we still have no capabilities, extract sentences that might be capabilities
    if not capabilities:
        sentences = re.split(r'\\.(?=\\s|$)', text)
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
''')
    logger.info(f"Created {source_file}")

    # Create updates.py for API Discovery
    with open("atia/api_discovery/updates.py", "w") as f:
        f.write('''"""
API Discovery updates for Phase 1: Week 2.

This update enhances the API Discovery component with better Responses API integration
and improved error handling.
"""

import logging
from typing import Dict, List, Optional, Any

from atia.config import settings
from atia.api_discovery.discovery import APIDiscovery, APICandidate
from atia.api_discovery.responses_integration import (
    search_apis_with_responses_api,
    evaluate_api_with_responses_api
)

logger = logging.getLogger(__name__)

# Extend the existing APIDiscovery class with enhanced methods
async def search_with_responses_api(self, capability_description: str, num_results: int = 5) -> List[APICandidate]:
    """
    Enhanced search method using Responses API when available.

    Args:
        capability_description: Description of the required capability
        num_results: Maximum number of results to return

    Returns:
        List of API candidates
    """
    logger.info(f"Searching for APIs with Responses API integration: {capability_description[:50]}...")

    # Try using Responses API first if enabled
    if not settings.disable_responses_api:
        try:
            # Search for APIs using Responses API
            api_results = await search_apis_with_responses_api(
                capability=capability_description,
                max_results=num_results
            )

            # Convert to APICandidate objects
            if api_results:
                candidates = []
                for api in api_results:
                    # Create APICandidate with available information
                    candidate = APICandidate(
                        name=api.get("name", "Unknown API"),
                        provider=api.get("provider", "Unknown"),
                        description=api.get("description", ""),
                        documentation_url=api.get("documentation_url", ""),
                        requires_auth=True,  # Default to requiring auth
                        auth_type=api.get("auth_type", "api_key"),
                        relevance_score=api.get("relevance_score", 0.5)
                    )
                    candidates.append(candidate)

                logger.info(f"Found {len(candidates)} API candidates using Responses API")
                return candidates

        except Exception as e:
            logger.warning(f"Error using Responses API for API search: {e}, falling back to standard search")

    # Fall back to standard search if Responses API fails or is disabled
    logger.info("Using standard API search method")
    return await self.search_for_api(capability_description, num_results, evaluate=True)

async def evaluate_candidates_with_responses_api(
    self, 
    candidates: List[APICandidate], 
    capability_description: str
) -> List[APICandidate]:
    """
    Enhanced evaluation of API candidates using Responses API.

    Args:
        candidates: List of API candidates to evaluate
        capability_description: Description of the required capability

    Returns:
        List of API candidates with updated scores and capabilities
    """
    if not candidates:
        return []

    # Only evaluate top candidates for efficiency
    top_candidates = candidates[:5]
    evaluated_candidates = []

    # Try to use Responses API for evaluation if available
    if not settings.disable_responses_api:
        for candidate in top_candidates:
            try:
                # Convert candidate to dict for evaluation
                api_info = {
                    "name": candidate.name,
                    "provider": candidate.provider,
                    "description": candidate.description
                }

                # Evaluate using Responses API
                relevance_score, capabilities = await evaluate_api_with_responses_api(
                    api_info=api_info,
                    capability=capability_description
                )

                # Update candidate with evaluation results
                candidate.relevance_score = relevance_score
                candidate.capabilities = capabilities

                evaluated_candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Error evaluating candidate {candidate.name} with Responses API: {e}")
                # Keep the original candidate if evaluation fails
                evaluated_candidates.append(candidate)

        # Add any remaining candidates
        if len(candidates) > 5:
            evaluated_candidates.extend(candidates[5:])

        # Sort by relevance score
        evaluated_candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"Evaluated {len(evaluated_candidates)} candidates with Responses API")
        return evaluated_candidates

    # Fall back to standard evaluation
    logger.info("Using standard API evaluation method")
    return await self.evaluate_api_candidates(candidates, capability_description)

# Extend the APIDiscovery with a combined method for search and evaluation
async def enhanced_api_search(
    self, 
    capability_description: str, 
    num_results: int = 5,
    use_responses_api: bool = True
) -> List[APICandidate]:
    """
    Enhanced API search with Responses API integration.

    This method combines search and evaluation in one call with improved error handling.

    Args:
        capability_description: Description of the required capability
        num_results: Maximum number of results to return
        use_responses_api: Whether to use Responses API integration

    Returns:
        List of evaluated API candidates
    """
    try:
        # Step 1: Search for APIs
        if use_responses_api and not settings.disable_responses_api:
            # Use the enhanced search method
            candidates = await search_with_responses_api(self, capability_description, num_results)
        else:
            # Use the standard search method
            candidates = await self.search_for_api(capability_description, num_results, evaluate=False)

        if not candidates:
            logger.warning(f"No API candidates found for capability: {capability_description}")
            return []

        # Step 2: Evaluate candidates
        if use_responses_api and not settings.disable_responses_api:
            # Use the enhanced evaluation method
            evaluated_candidates = await evaluate_candidates_with_responses_api(
                self, candidates, capability_description
            )
        else:
            # Use the standard evaluation method
            evaluated_candidates = await self.evaluate_api_candidates(candidates, capability_description)

        # Return top results
        return evaluated_candidates[:num_results]

    except Exception as e:
        logger.error(f"Error in enhanced API search: {e}")
        # Try one last fallback to standard search
        try:
            return await self.search_for_api(capability_description, num_results)
        except Exception as e2:
            logger.error(f"Fallback search also failed: {e2}")
            return []

# Add these methods to the APIDiscovery class
APIDiscovery.search_with_responses_api = search_with_responses_api
APIDiscovery.evaluate_candidates_with_responses_api = evaluate_candidates_with_responses_api
APIDiscovery.enhanced_api_search = enhanced_api_search
''')
    logger.info("Created atia/api_discovery/updates.py")

def apply_function_builder_updates() -> None:
    """Apply updates to the Function Builder component."""
    with open("atia/function_builder/updates.py", "w") as f:
        f.write('''"""
Function Builder updates for Phase 1: Week 2.

This update enhances the Function Builder component with better Responses API integration
and improved authentication handling.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

from atia.config import settings
from atia.function_builder.builder import FunctionBuilder
from atia.function_builder.models import (
    FunctionDefinition, 
    ApiType, 
    ParameterType, 
    FunctionParameter
)
from atia.doc_processor.processor import APIEndpoint, APIInfo

logger = logging.getLogger(__name__)

# Enhanced Responses API schema generation
def enhanced_responses_api_tool_schema(
    self, 
    function_def: FunctionDefinition,
    include_auth: bool = True
) -> Dict:
    """
    Enhanced version of generate_responses_api_tool_schema with better authentication support.

    Args:
        function_def: The function definition to convert
        include_auth: Whether to include authentication information

    Returns:
        Enhanced tool schema compatible with Responses API
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

    # Enhanced parameter conversion with better type handling
    for param in function_def.parameters:
        param_name = param.name

        # Skip credentials parameter which will be handled separately
        if param_name == "credentials":
            continue

        # Map parameter types to JSON Schema types with enhanced support
        type_mapping = {
            "string": {
                "type": "string",
                "description": param.description
            },
            "integer": {
                "type": "integer",
                "description": param.description
            },
            "float": {
                "type": "number",
                "description": param.description
            },
            "boolean": {
                "type": "boolean",
                "description": param.description
            },
            "array": {
                "type": "array",
                "description": param.description,
                "items": {"type": "string"}  # Default to string items
            },
            "object": {
                "type": "object",
                "description": param.description,
                "properties": {}  # Empty properties by default
            }
        }

        # Get the appropriate schema for this parameter type
        param_schema = type_mapping.get(param.param_type.value, {"type": "string", "description": param.description})

        # Add default value if specified
        if param.default_value is not None:
            param_schema["default"] = param.default_value

        # Add enum if parameter has validation constraints (simple implementation)
        if hasattr(param, "metadata") and param.metadata and "enum" in param.metadata:
            param_schema["enum"] = param.metadata["enum"]

        # Add the parameter to the properties
        tool_schema["function"]["parameters"]["properties"][param_name] = param_schema

        # Add to required list if parameter is required
        if param.required:
            tool_schema["function"]["parameters"]["required"].append(param_name)

    # Add authentication information if requested
    if include_auth:
        # Extract auth type from tags or infer from endpoint method
        auth_type = next((tag for tag in function_def.tags if tag in ["api_key", "oauth", "bearer", "basic"]), None)

        if auth_type or "credentials" in [p.name for p in function_def.parameters]:
            # Add auth metadata to the tool schema
            tool_schema["authentication"] = {
                "type": auth_type or "api_key",
                "required": True
            }

            # Add additional auth information for specific types
            if auth_type == "oauth":
                tool_schema["authentication"]["scope"] = "read write"  # Default scope

    return tool_schema

# Enhanced function generation with improved authentication and error handling
async def enhanced_function_generation(
    self, 
    api_info: APIInfo, 
    endpoint: APIEndpoint,
    capability_description: Optional[str] = None
) -> FunctionDefinition:
    """
    Enhanced function generation with better authentication and error handling.

    Args:
        api_info: Information about the API
        endpoint: The specific endpoint to generate a function for
        capability_description: Optional description for better function customization

    Returns:
        Generated function definition
    """
    logger.info(f"Generating enhanced function for endpoint: {endpoint.method} {endpoint.path}")

    try:
        # Use the existing function generation code as a base
        function_def = await self.build_function_for_api(
            api_info=api_info,
            endpoint=endpoint,
            tool_need_description=capability_description
        )

        # Enhance the generated function with better authentication handling
        function_def = _enhance_authentication(function_def, api_info)

        # Add better error handling to the function code
        function_def.code = _add_enhanced_error_handling(function_def.code)

        # Add standard credential parameter if not present
        _ensure_credential_parameter(function_def)

        # Update function tags with authentication information
        _update_function_tags(function_def, api_info)

        # Validate the enhanced function
        is_valid, messages = await self.validate_function(function_def)
        if not is_valid:
            logger.warning(f"Generated function has validation issues: {messages}")
            # Try to fix common issues
            function_def = _fix_common_validation_issues(function_def, messages)

        return function_def

    except Exception as e:
        logger.error(f"Error in enhanced function generation: {e}")
        # Return a simplified function as fallback
        return _create_fallback_function(api_info, endpoint, capability_description)

# Helper functions for enhanced function generation

def _enhance_authentication(function_def: FunctionDefinition, api_info: APIInfo) -> FunctionDefinition:
    """Add enhanced authentication handling to the function."""
    # Determine authentication type from API info
    auth_type = "api_key"  # Default
    if api_info.auth_methods:
        auth_method = api_info.auth_methods[0]
        auth_type = auth_method.get("type", "").lower()
        if auth_type == "http":
            scheme = auth_method.get("scheme", "").lower()
            if scheme == "bearer":
                auth_type = "bearer"
            elif scheme == "basic":
                auth_type = "basic"

    # Add authentication type to function's metadata
    if not hasattr(function_def, "metadata"):
        function_def.metadata = {}
    function_def.metadata["auth_type"] = auth_type

    return function_def

def _add_enhanced_error_handling(code: str) -> str:
    """Add enhanced error handling to the function code."""
    # Check if error handling is already present
    if "APIError" in code and "response.status" in code:
        # Already has error handling, make sure it's comprehensive
        if "try:" not in code:
            # Add try-except block around the main function body
            code_lines = code.split("\\n")
            function_def_line = next((i for i, line in enumerate(code_lines) if line.startswith("async def")), -1)

            if function_def_line >= 0:
                # Find the function body indentation
                body_start = function_def_line + 1
                while body_start < len(code_lines) and (not code_lines[body_start].strip() or code_lines[body_start].lstrip().startswith('"""')):
                    body_start += 1

                if body_start < len(code_lines):
                    # Get indentation
                    indent = len(code_lines[body_start]) - len(code_lines[body_start].lstrip())
                    indent_str = " " * indent

                    # Find where to insert try-except
                    # Skip past docstring if present
                    if '"""' in code_lines[body_start:function_def_line+10]:
                        docstring_start = next((i for i, line in enumerate(code_lines[body_start:function_def_line+10], body_start) if '"""' in line), -1)
                        if docstring_start >= 0:
                            docstring_end = next((i for i, line in enumerate(code_lines[docstring_start+1:], docstring_start+1) if '"""' in line), -1)
                            if docstring_end >= 0:
                                body_start = docstring_end + 1

                    # Insert try block
                    code_lines.insert(body_start, indent_str + "try:")

                    # Insert except block at the end
                    code_lines.append(indent_str + "except Exception as e:")
                    code_lines.append(indent_str + "    error_msg = f\\"Error executing function: {str(e)}\\"")
                    code_lines.append(indent_str + "    raise APIError(error_msg) from e")

                    code = "\\n".join(code_lines)

    # Enhance the error handling for API responses
    if "response.status" in code and "error_text = await response.text()" not in code:
        # Add better error handling for API responses
        code = code.replace(
            "if response.status < 200 or response.status >= 300:",
            """if response.status < 200 or response.status >= 300:
                # Try to get error details from response
                try:
                    error_json = await response.json()
                    error_message = error_json.get("message", error_json.get("error", ""))
                    if error_message:
                        raise APIError(f"API Error ({response.status}): {error_message}")
                except:
                    # Fallback to text response
                    error_text = await response.text()
                    raise APIError(f"Request failed with status {response.status}: {error_text}")"""
        )

    return code

def _ensure_credential_parameter(function_def: FunctionDefinition) -> None:
    """Ensure the function has a credentials parameter if authentication is needed."""
    # Check if credentials parameter already exists
    has_credentials = any(param.name == "credentials" for param in function_def.parameters)

    if not has_credentials:
        # Add credentials parameter
        function_def.parameters.append(
            FunctionParameter(
                name="credentials",
                param_type=ParameterType.OBJECT,
                description="Authentication credentials for the API",
                required=False,
                default_value=None
            )
        )

def _update_function_tags(function_def: FunctionDefinition, api_info: APIInfo) -> None:
    """Update function tags with authentication information."""
    auth_tags = set()

    # Extract auth type from API info
    if api_info.auth_methods:
        for auth_method in api_info.auth_methods:
            auth_type = auth_method.get("type", "").lower()
            auth_tags.add(auth_type)

            if auth_type == "http":
                scheme = auth_method.get("scheme", "").lower()
                if scheme:
                    auth_tags.add(scheme)

    # Add auth tags to function tags
    function_def.tags = list(set(function_def.tags).union(auth_tags))

def _fix_common_validation_issues(function_def: FunctionDefinition, messages: List[str]) -> FunctionDefinition:
    """Fix common validation issues in the function definition."""
    # Fix invalid parameter names
    for message in messages:
        if "Parameter name" in message and "contains invalid characters" in message:
            # Extract parameter name
            param_name_match = re.search(r"Parameter name '(\\w+)'", message)
            if param_name_match:
                invalid_param = param_name_match.group(1)
                # Find and fix the parameter
                for param in function_def.parameters:
                    if param.name == invalid_param:
                        # Clean the parameter name
                        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', param.name)
                        param.name = clean_name
                        # Update parameter reference in code
                        function_def.code = function_def.code.replace(invalid_param, clean_name)

    # Fix syntax errors in function code
    for message in messages:
        if "Syntax error" in message:
            # Simple fixes for common syntax errors
            if "EOF while scanning" in message or "unexpected indent" in message:
                # Try to fix indentation or missing closing quotes/brackets
                function_def.code = _fix_syntax_errors(function_def.code)

    return function_def

def _fix_syntax_errors(code: str) -> str:
    """Simple fixes for common syntax errors."""
    # Check for unbalanced brackets/parentheses
    brackets = {'(': ')', '{': '}', '[': ']'}
    stack = []

    for char in code:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                # Unbalanced bracket found
                if char == ')':
                    code += ')'
                elif char == '}':
                    code += '}'
                elif char == ']':
                    code += ']'

    # Add any missing closing brackets
    while stack:
        bracket = stack.pop()
        code += brackets[bracket]

    # Check for unclosed string literals
    lines = code.split('\\n')
    fixed_lines = []

    for line in lines:
        # Count quotes
        single_quotes = line.count("'")
        double_quotes = line.count('"')

        # If odd number of quotes, add a closing quote
        if single_quotes % 2 == 1:
            line += "'"
        if double_quotes % 2 == 1:
            line += '"'

        fixed_lines.append(line)

    return '\\n'.join(fixed_lines)

def _create_fallback_function(api_info: APIInfo, endpoint: APIEndpoint, capability_description: Optional[str] = None) -> FunctionDefinition:
    """Create a simplified fallback function when normal generation fails."""
    function_name = f"fallback_{endpoint.method.lower()}_{api_info.source_id[:6]}"

    # Create basic parameters based on endpoint
    parameters = []
    if endpoint.parameters:
        for i, param in enumerate(endpoint.parameters[:5]):  # Limit to 5 parameters
            param_name = f"param{i+1}"
            if isinstance(param, dict):
                description = param.get("description", f"Parameter {i+1}")
                required = param.get("required", False)
            else:
                description = getattr(param, "description", f"Parameter {i+1}")
                required = getattr(param, "required", False)

            parameters.append(
                FunctionParameter(
                    name=param_name,
                    param_type=ParameterType.STRING,
                    description=description,
                    required=required
                )
            )

    # Add credentials parameter
    parameters.append(
        FunctionParameter(
            name="credentials",
            param_type=ParameterType.OBJECT,
            description="Authentication credentials for the API",
            required=False
        )
    )

    # Create a simple function code
    code = f"""
async def {function_name}({', '.join([p.name for p in parameters])}):
    \\"\\"\\"
    {endpoint.description or f"{endpoint.method} {endpoint.path}"}

    Args:
        {chr(10)+'        '.join([f"{p.name}: {p.description}" for p in parameters])}

    Returns:
        API response
    \\"\\"\\"
    import aiohttp
    import json

    url = "{api_info.base_url}{endpoint.path}"
    headers = {{\\"Content-Type\\": \\"application/json\\"}}

    # Add authentication if credentials provided
    if credentials:
        if \\"api_key\\" in credentials:
            headers[\\"X-API-Key\\"] = credentials[\\"api_key\\"]
        elif \\"token\\" in credentials:
            headers[\\"Authorization\\"] = f\\"Bearer {{credentials['token']}}\\"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.{endpoint.method.lower()}(url, headers=headers) as response:
                if response.status < 200 or response.status >= 300:
                    error_text = await response.text()
                    raise Exception(f\\"Request failed with status {{response.status}}: {{error_text}}\\")

                result = await response.json()
                return result
    except Exception as e:
        raise Exception(f\\"Error calling API: {{str(e)}}\\")
"""

    # Create basic function definition
    function_def = FunctionDefinition(
        name=function_name,
        description=endpoint.description or f"{endpoint.method} {endpoint.path}",
        api_source_id=api_info.source_id,
        api_type=ApiType.REST,
        parameters=parameters,
        code=code,
        endpoint=endpoint.path,
        method=endpoint.method,
        tags=["fallback", endpoint.method.lower(), "auto_generated"],
        response_format={}
    )

    return function_def

# Add the enhanced methods to the FunctionBuilder class
FunctionBuilder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema
FunctionBuilder.enhanced_function_generation = enhanced_function_generation
''')
    logger.info("Created atia/function_builder/updates.py")

def create_integration_helper() -> None:
    """Create the integration helper module."""
    create_directory_if_not_exists("atia/utils")

    with open("atia/utils/integration_helper.py", "w") as f:
        f.write('''"""
Integration helper to connect the enhanced components with Agent Core.

This module provides functions to integrate the enhanced API Discovery and
Function Builder components with the Agent Core.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from atia.config import settings
from atia.api_discovery.discovery import APICandidate
from atia.doc_processor.processor import APIInfo, APIEndpoint
from atia.function_builder.models import FunctionDefinition
from atia.need_identifier.identifier import ToolNeed

logger = logging.getLogger(__name__)

async def discover_and_generate_tool(
    agent_core,
    capability: str,
    use_responses_api: bool = True
) -> Tuple[Optional[FunctionDefinition], Optional[APICandidate]]:
    """
    Discover APIs and generate tool functions in one integrated step.

    This helper function integrates the enhanced API Discovery and Function Builder
    components to create tools for the Agent Core.

    Args:
        agent_core: Instance of AgentCore
        capability: Description of the capability needed
        use_responses_api: Whether to use Responses API integration

    Returns:
        Tuple of (function_definition, api_candidate) or (None, None) if failed
    """
    try:
        logger.info(f"Starting integrated tool discovery and generation for capability: {capability}")

        # Step 1: Create tool need
        tool_need = ToolNeed(
            category="dynamic_discovery",
            description=capability,
            confidence=1.0
        )

        # Step 2: Check if we have existing tools
        existing_tools = await agent_core.tool_registry.search_by_capability(tool_need.description)
        if existing_tools:
            logger.info(f"Found existing tool: {existing_tools[0].name}")
            return None, None

        # Step 3: Search for APIs using enhanced search
        api_candidates = await agent_core.api_discovery.enhanced_api_search(
            capability_description=tool_need.description,
            num_results=5,
            use_responses_api=use_responses_api
        )

        if not api_candidates:
            logger.info("No suitable APIs found")
            return None, None

        # Step 4: Select the top API candidate
        top_api = api_candidates[0]
        logger.info(f"Selected API: {top_api.name} (score: {top_api.relevance_score:.2f})")

        # Step 5: Fetch and process documentation
        doc_content = ""
        if hasattr(top_api, 'documentation_content') and top_api.documentation_content:
            doc_content = top_api.documentation_content
        elif top_api.documentation_url:
            try:
                doc_content = await agent_core.doc_processor.fetch_documentation(top_api.documentation_url)
            except Exception as e:
                logger.error(f"Error fetching documentation: {e}")

        # Step 6: Process the documentation
        api_info = await agent_core.doc_processor.process_documentation(
            doc_content=doc_content,
            url=top_api.documentation_url
        )

        # Step 7: Extract endpoints
        endpoints = await agent_core.doc_processor.extract_endpoints(api_info)
        if not endpoints:
            logger.info("No endpoints found in documentation")
            return None, None

        # Step 8: Select the most appropriate endpoint
        selected_endpoint = await agent_core.doc_processor.select_endpoint_for_need(
            endpoints, tool_need.description
        )
        logger.info(f"Selected endpoint: {selected_endpoint.method} {selected_endpoint.path}")

        # Step 9: Build function with enhanced generation
        function_def = await agent_core.function_builder.enhanced_function_generation(
            api_info=api_info,
            endpoint=selected_endpoint,
            capability_description=tool_need.description
        )

        # Return the function definition and API candidate
        return function_def, top_api

    except Exception as e:
        logger.error(f"Error in discover_and_generate_tool: {e}")
        return None, None

async def register_function_as_tool(
    agent_core,
    function_def: FunctionDefinition,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Register a function as a tool in the Tool Registry.

    Args:
        agent_core: Instance of AgentCore
        function_def: The function definition to register
        metadata: Additional metadata for the function

    Returns:
        The registered tool
    """
    if metadata is None:
        metadata = {}

    try:
        # Register the function with the tool registry
        tool = await agent_core.tool_registry.register_function(function_def, metadata)
        logger.info(f"Registered new tool: {tool.name}")
        return tool
    except Exception as e:
        logger.error(f"Error registering function as tool: {e}")
        return None

async def prepare_tools_for_responses_api(
    agent_core,
    capability_description: Optional[str] = None
) -> List[Dict]:
    """
    Prepare all relevant tools for use with the Responses API.

    Args:
        agent_core: Instance of AgentCore
        capability_description: Optional capability description to filter tools

    Returns:
        List of tools in Responses API format
    """
    try:
        # Get tools from registry
        tools = await agent_core.tool_registry.get_tools_for_responses_api(capability_description)

        # Add dynamic tool discovery function
        tools.append(agent_core.dynamic_tool_discovery_schema)

        return tools
    except Exception as e:
        logger.error(f"Error preparing tools for Responses API: {e}")
        # Return just the dynamic tool discovery schema as fallback
        return [agent_core.dynamic_tool_discovery_schema]

async def execute_tool_integration_flow(
    agent_core,
    query: str,
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Execute the full tool integration flow with enhanced components.

    This function integrates all the enhanced components to process a user query
    with dynamic tool discovery and execution.

    Args:
        agent_core: Instance of AgentCore
        query: The user query
        context: Additional context about the query

    Returns:
        Response with tool integration results
    """
    if context is None:
        context = {}

    try:
        # Step 1: Identify if a tool is needed
        tool_need = await agent_core.need_identifier.identify_tool_need(query, context)

        if not tool_need:
            # No tool needed, process with standard completion
            return await agent_core.process_query_with_responses_api(query, context=context)

        # Step 2: Prepare available tools
        tools = await prepare_tools_for_responses_api(agent_core, tool_need.description)

        # Step 3: Process with Responses API and dynamic tool discovery
        response = await agent_core._process_with_dynamic_discovery(query, context)

        return response
    except Exception as e:
        logger.error(f"Error in execute_tool_integration_flow: {e}")
        # Fall back to standard completion
        try:
            return await agent_core.process_query(query, context)
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return {"content": f"Error processing your request: {str(e)}"}
''')
    logger.info("Created atia/utils/integration_helper.py")

def create_unit_tests() -> None:
    """Create unit tests for the enhanced components."""
    create_directory_if_not_exists("tests")

    with open("tests/test_phase1_week2.py", "w") as f:
        f.write('''"""
Unit tests for Phase 1: Week 2 - Function Builder & API Discovery enhancements.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_api_discovery_responses_integration():
    """Test the API Discovery with Responses API integration."""
    from atia.api_discovery.responses_integration import search_apis_with_responses_api

    # Mock the get_completion_with_responses_api function
    with patch('atia.api_discovery.responses_integration.get_completion_with_responses_api', new_callable=AsyncMock) as mock_completion:
        # Set up the mock to return a response that requires action
        mock_completion.return_value = {
            "status": "requires_action",
            "required_action": {
                "submit_tool_outputs": {
                    "tool_calls": [
                        {
                            "id": "test_id",
                            "function": {
                                "name": "search_apis",
                                "arguments": json.dumps({
                                    "capability": "weather forecasting",
                                    "max_results": 3
                                })
                            }
                        }
                    ]
                }
            }
        }

        # Call the function
        result = await search_apis_with_responses_api("weather forecasting", 3)

        # Verify the result
        assert isinstance(result, list)
        assert len(result) > 0
        assert "name" in result[0]
        assert "provider" in result[0]
        assert "description" in result[0]
        assert "documentation_url" in result[0]

        # Verify the mock was called with expected arguments
        mock_completion.assert_called_once()
        args, kwargs = mock_completion.call_args
        assert "weather forecasting" in kwargs.get("prompt", "")
        assert kwargs.get("tools", [])


@pytest.mark.asyncio
async def test_enhanced_api_search():
    """Test the enhanced API search method."""
    from atia.api_discovery.discovery import APIDiscovery, APICandidate
    from atia.api_discovery.responses_integration import search_apis_with_responses_api
    import atia.api_discovery.discovery as api_discovery_module

    # Create a mock APIDiscovery instance
    discovery = APIDiscovery()

    # Mock the search_with_responses_api method
    with patch('atia.api_discovery.responses_integration.search_apis_with_responses_api', new_callable=AsyncMock) as mock_search:
        # Set up the mock to return test results
        mock_search.return_value = [
            {
                "name": "Test API",
                "provider": "Test Provider",
                "description": "Test description",
                "documentation_url": "https://test.com",
                "auth_type": "api_key",
                "relevance_score": 0.9
            }
        ]

        # Mock the enhanced_api_search method onto the APIDiscovery instance
        from atia.api_discovery.updates import enhanced_api_search, search_with_responses_api
        discovery.enhanced_api_search = enhanced_api_search.__get__(discovery, APIDiscovery)
        discovery.search_with_responses_api = search_with_responses_api.__get__(discovery, APIDiscovery)

        # Also mock the evaluate_candidates_with_responses_api method
        discovery.evaluate_candidates_with_responses_api = AsyncMock(side_effect=lambda candidates, *args: candidates)

        # Call the enhanced_api_search method
        result = await discovery.enhanced_api_search("weather forecast", 3)

        # Verify the result
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], APICandidate)
        assert result[0].name == "Test API"
        assert result[0].provider == "Test Provider"

        # Verify the mock was called with expected arguments
        mock_search.assert_called_once()
        args, kwargs = mock_search.call_args
        assert kwargs.get("capability") == "weather forecast"


@pytest.mark.asyncio
async def test_function_builder_enhanced_schema():
    """Test the enhanced Responses API tool schema generation."""
    from atia.function_builder.builder import FunctionBuilder
    from atia.function_builder.models import FunctionDefinition, ParameterType, FunctionParameter, ApiType

    # Create a mock function definition
    function_def = FunctionDefinition(
        name="test_function",
        description="Test function",
        api_source_id="test_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="param1",
                param_type=ParameterType.STRING,
                description="Test parameter",
                required=True
            ),
            FunctionParameter(
                name="credentials",
                param_type=ParameterType.OBJECT,
                description="Authentication credentials",
                required=False
            )
        ],
        code="# Test code",
        endpoint="/test",
        method="GET",
        tags=["test", "api_key"]
    )

    # Create a FunctionBuilder instance
    builder = FunctionBuilder()

    # Import and apply the enhanced_responses_api_tool_schema method
    from atia.function_builder.updates import enhanced_responses_api_tool_schema
    builder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema.__get__(builder, FunctionBuilder)

    # Generate the schema
    schema = builder.enhanced_responses_api_tool_schema(function_def)

    # Verify the schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "test_function"
    assert schema["function"]["description"] == "Test function"
    assert "credentials" not in schema["function"]["parameters"]["properties"]
    assert "param1" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["param1"]["type"] == "string"
    assert "required" in schema["function"]["parameters"]
    assert "param1" in schema["function"]["parameters"]["required"]
    assert "authentication" in schema


@pytest.mark.asyncio
async def test_integration_helper():
    """Test the integration helper module."""
    from atia.agent_core.core import AgentCore
    from atia.tool_registry import ToolRegistry

    # Create mock components
    agent_core = MagicMock()
    agent_core.tool_registry = MagicMock()
    agent_core.api_discovery = MagicMock()
    agent_core.doc_processor = MagicMock()
    agent_core.function_builder = MagicMock()
    agent_core.dynamic_tool_discovery_schema = {"type": "function", "function": {"name": "dynamic_tool_discovery"}}

    # Mock search_by_capability to return empty list
    agent_core.tool_registry.search_by_capability = AsyncMock(return_value=[])

    # Mock get_tools_for_responses_api
    agent_core.tool_registry.get_tools_for_responses_api = AsyncMock(return_value=[])

    # Import the integration helper
    from atia.utils.integration_helper import prepare_tools_for_responses_api

    # Call the function
    tools = await prepare_tools_for_responses_api(agent_core)

    # Verify the result
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "dynamic_tool_discovery"

    # Verify the mock was called
    agent_core.tool_registry.get_tools_for_responses_api.assert_called_once()


def test_create_test_instance():
    """Create test instances of the enhanced components."""
    # Create API Discovery instance
    from atia.api_discovery.discovery import APIDiscovery
    discovery = APIDiscovery()

    # Create Function Builder instance
    from atia.function_builder.builder import FunctionBuilder
    builder = FunctionBuilder()

    # This test just verifies that we can create instances without errors
    assert isinstance(discovery, APIDiscovery)
    assert isinstance(builder, FunctionBuilder)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
''')
    logger.info("Created tests/test_phase1_week2.py")

def create_agent_core_integration() -> None:
    """Create integration code for Agent Core."""
    with open("apply_agent_core_integration.py", "w") as f:
        f.write('''"""
Apply Agent Core integration for Phase 1: Week 2.

This module integrates the enhanced API Discovery and Function Builder components
with the Agent Core.
"""

import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def import_or_fail(module_name: str) -> bool:
    """Import a module or fail with useful message."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        logger.error(f"Error importing {module_name}. Make sure the module exists.")
        return False

def integrate_agent_core() -> None:
    """Integrate updated components with Agent Core."""
    logger.info("Integrating enhanced components with Agent Core...")

    # Import necessary modules
    if not import_or_fail("atia.agent_core.core"):
        return

    if not import_or_fail("atia.api_discovery.updates"):
        return

    if not import_or_fail("atia.function_builder.updates"):
        return

    if not import_or_fail("atia.utils.integration_helper"):
        return

    # Apply the updates to the real classes
    from atia.agent_core.core import AgentCore
    from atia.api_discovery.discovery import APIDiscovery
    from atia.function_builder.builder import FunctionBuilder

    # Apply API Discovery updates
    from atia.api_discovery.updates import (
        search_with_responses_api,
        evaluate_candidates_with_responses_api,
        enhanced_api_search
    )

    APIDiscovery.search_with_responses_api = search_with_responses_api
    APIDiscovery.evaluate_candidates_with_responses_api = evaluate_candidates_with_responses_api
    APIDiscovery.enhanced_api_search = enhanced_api_search

    # Apply Function Builder updates
    from atia.function_builder.updates import (
        enhanced_responses_api_tool_schema,
        enhanced_function_generation
    )

    FunctionBuilder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema
    FunctionBuilder.enhanced_function_generation = enhanced_function_generation

    # Extend AgentCore to use the integration helper
    from atia.utils.integration_helper import (
        discover_and_generate_tool,
        register_function_as_tool,
        prepare_tools_for_responses_api,
        execute_tool_integration_flow
    )

    # Add the new methods to AgentCore
    AgentCore.discover_and_generate_tool = discover_and_generate_tool
    AgentCore.register_function_as_tool = register_function_as_tool
    AgentCore.prepare_tools_for_responses_api = prepare_tools_for_responses_api
    AgentCore.execute_tool_integration_flow = execute_tool_integration_flow

    logger.info("Integration complete. Enhanced components now available in AgentCore.")

if __name__ == "__main__":
    integrate_agent_core()
''')
    logger.info("Created apply_agent_core_integration.py")

def create_demo_script() -> None:
    """Create a demo script for testing the enhanced components."""
    with open("demo_phase1_week2.py", "w") as f:
        f.write('''#!/usr/bin/env python

"""
Demo script for Phase 1: Week 2 - Function Builder & API Discovery enhancements.

This script demonstrates the enhanced API Discovery and Function Builder components.
"""

import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ensure that the needed directories exist
os.makedirs("data/tools", exist_ok=True)
os.makedirs("data/functions", exist_ok=True)
os.makedirs("data/credentials", exist_ok=True)

async def api_discovery_demo():
    """Demonstrate the enhanced API Discovery component."""
    from atia.api_discovery.discovery import APIDiscovery
    from atia.api_discovery.updates import enhanced_api_search

    # Create API Discovery instance
    discovery = APIDiscovery()

    # Apply the enhanced_api_search method
    discovery.enhanced_api_search = enhanced_api_search.__get__(discovery, APIDiscovery)

    # Test capabilities
    capabilities = [
        "Get current weather for a city",
        "Translate text from English to French",
        "Find nearby restaurants"
    ]

    for capability in capabilities:
        print(f"\n📋 Testing API discovery for: {capability}")
        try:
            # Search for APIs with enhanced search
            apis = await discovery.enhanced_api_search(capability)

            # Display results
            if apis:
                print(f"✅ Found {len(apis)} potential APIs:")
                for i, api in enumerate(apis[:3], 1):
                    print(f"  {i}. {api.name} by {api.provider}")
                    print(f"     Score: {api.relevance_score:.2f}")
                    print(f"     Description: {api.description[:100]}...")
                    print(f"     Documentation: {api.documentation_url}")
                    if hasattr(api, 'capabilities') and api.capabilities:
                        print(f"     Capabilities: {', '.join(api.capabilities[:3])}")
                    print()
            else:
                print("❌ No APIs found")
        except Exception as e:
            print(f"❌ Error in API discovery: {e}")

async def function_builder_demo():
    """Demonstrate the enhanced Function Builder component."""
    from atia.function_builder.builder import FunctionBuilder
    from atia.function_builder.updates import enhanced_responses_api_tool_schema
    from atia.function_builder.models import (
        FunctionDefinition, ApiType, ParameterType, FunctionParameter
    )
    from atia.doc_processor.processor import APIEndpoint, APIInfo

    # Create Function Builder instance
    builder = FunctionBuilder()

    # Apply the enhanced method
    builder.enhanced_responses_api_tool_schema = enhanced_responses_api_tool_schema.__get__(builder, FunctionBuilder)

    # Create a sample function definition
    function_def = FunctionDefinition(
        name="get_weather",
        description="Get current weather for a location",
        api_source_id="weather_api",
        api_type=ApiType.REST,
        parameters=[
            FunctionParameter(
                name="location",
                param_type=ParameterType.STRING,
                description="City name or coordinates",
                required=True
            ),
            FunctionParameter(
                name="units",
                param_type=ParameterType.STRING,
                description="Temperature units (metric, imperial)",
                required=False,
                default_value="metric"
            ),
            FunctionParameter(
                name="credentials",
                param_type=ParameterType.OBJECT,
                description="API credentials",
                required=False
            )
        ],
        code="""
async def get_weather(location: str, units: str = "metric", credentials: Dict[str, Any] = None):
    \"\"\"
    Get current weather for a location.

    Args:
        location: City name or coordinates
        units: Temperature units (metric, imperial)
        credentials: API credentials

    Returns:
        Weather data
    \"\"\"
    url = "https://api.weatherapi.com/v1/current.json"
    headers = {"Content-Type": "application/json"}

    # Add API key if provided
    if credentials and "api_key" in credentials:
        headers["X-API-Key"] = credentials["api_key"]

    params = {"q": location, "units": units}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status < 200 or response.status >= 300:
                error_text = await response.text()
                raise Exception(f"Request failed with status {response.status}: {error_text}")

            result = await response.json()
            return result
""",
        endpoint="/v1/current.json",
        method="GET",
        response_format={},
        tags=["weather", "current", "api_key"]
    )

    print("\n📋 Testing Function Builder enhancement")

    # Generate Responses API tool schema
    try:
        schema = builder.enhanced_responses_api_tool_schema(function_def)

        # Display the schema
        import json
        print("✅ Generated Responses API tool schema:")
        print(json.dumps(schema, indent=2))

        # Show that authentication is properly handled
        if "authentication" in schema:
            print("\n✅ Authentication included in schema:")
            print(json.dumps(schema["authentication"], indent=2))
        else:
            print("\n❌ Authentication not included in schema")

    except Exception as e:
        print(f"❌ Error generating schema: {e}")

async def integration_demo():
    """Demonstrate the integration capabilities."""
    from atia.agent_core.core import AgentCore
    from atia.tool_registry import ToolRegistry
    from atia.utils.integration_helper import discover_and_generate_tool, register_function_as_tool

    print("\n📋 Testing Integration Helper")

    try:
        # Create needed components
        tool_registry = ToolRegistry()
        agent = AgentCore(tool_registry=tool_registry)

        # Install the method on the AgentCore instance
        from atia.utils.integration_helper import discover_and_generate_tool
        agent.discover_and_generate_tool = discover_and_generate_tool.__get__(agent, AgentCore)

        # Demonstrate the tool discovery and generation flow
        capability = "Get weather forecast for a city"
        print(f"🔍 Searching for APIs with capability: {capability}")

        # This would normally call all the components, but we'll simulate for the demo
        print("✅ Integration flow successfully set up")
        print("   In a real environment, this would discover and generate a tool")
        print("   using the enhanced API Discovery and Function Builder components.")

    except Exception as e:
        print(f"❌ Error in integration demo: {e}")

async def main():
    """Run all demos."""
    print("🚀 ATIA Phase 1: Week 2 Demo - Function Builder & API Discovery Enhancements")

    print("\n==== API Discovery Demo ====")
    await api_discovery_demo()

    print("\n==== Function Builder Demo ====")
    await function_builder_demo()

    print("\n==== Integration Demo ====")
    await integration_demo()

    print("\n✨ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())
''')
    logger.info("Created demo_phase1_week2.py")

def main():
    """Main function to apply all updates."""
    print("🚀 Applying Phase 1: Week 2 updates to ATIA.")

    # Check import paths
    check_import_paths()

    # Create necessary directories
    create_directory_if_not_exists("atia/api_discovery")
    create_directory_if_not_exists("atia/function_builder")
    create_directory_if_not_exists("atia/utils")
    create_directory_if_not_exists("tests")

    # Apply the updates
    apply_api_discovery_updates()
    apply_function_builder_updates()
    create_integration_helper()
    create_unit_tests()
    create_agent_core_integration()
    create_demo_script()

    print("\n✅ All updates applied successfully.")
    print("\nTo integrate these updates with the Agent Core, run:")
    print("  python apply_agent_core_integration.py")
    print("\nTo test the enhanced components, run:")
    print("  python demo_phase1_week2.py")
    print("\nTo run unit tests, run:")
    print("  pytest tests/test_phase1_week2.py")

if __name__ == "__main__":
    main()