"""
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
