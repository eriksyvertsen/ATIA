import json
from typing import Dict, List, Optional

import aiohttp
import numpy as np
from numpy.linalg import norm
from pydantic import BaseModel

from atia.config import settings
from atia.utils.openai_client import get_completion, get_embedding


class APICandidate(BaseModel):
    """
    Represents an API candidate discovered for a tool need.
    """
    name: str
    provider: str
    description: str
    documentation_url: str
    requires_auth: bool = True
    auth_type: Optional[str] = None  # "api_key", "oauth", "bearer", etc.
    pricing_tier: Optional[str] = None  # "free", "freemium", "paid"
    relevance_score: float = 0.0
    documentation_content: Optional[str] = None


class APIDiscovery:
    """
    Search for relevant APIs that can fulfill the identified tool need.
    """

    def __init__(self):
        """
        Initialize the API Discovery component.
        """
        self.rapid_api_key = settings.rapid_api_key
        self.serp_api_key = settings.serp_api_key
        self.github_token = settings.github_token

    def formulate_search_query(self, capability_description: str) -> str:
        """
        Generate an effective search query based on capability description.

        Args:
            capability_description: Description of the required capability

        Returns:
            Formulated search query
        """
        prompt = f"Generate a search query to find APIs that provide this capability: {capability_description}"

        system_message = (
            "You are an expert at formulating search queries for finding APIs. "
            "Generate a concise, effective search query that would find relevant APIs."
        )

        return get_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,
            max_tokens=100,
            model="gpt-4o-mini"  # Using a smaller model for efficiency
        )

    async def execute_web_search(self, search_query: str) -> List[Dict]:
        """
        Execute a web search for APIs using the formulated query.

        Args:
            search_query: The formulated search query

        Returns:
            List of search results
        """
        # For Phase 1, we'll use RapidAPI as our primary source
        # In later phases, this will include SERP API and GitHub API

        if not self.rapid_api_key:
            # Fallback to dummy results if no API key is provided
            return self._get_dummy_search_results(search_query)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-rapidapi-host": "rapidapi-marketplace.p.rapidapi.com",
                    "x-rapidapi-key": self.rapid_api_key
                }

                params = {
                    "query": search_query,
                    "page": "1",
                    "per_page": "10"
                }

                url = "https://rapidapi-marketplace.p.rapidapi.com/search"

                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Error searching RapidAPI: {response.status}")
                        return self._get_dummy_search_results(search_query)
        except Exception as e:
            print(f"Error executing web search: {e}")
            return self._get_dummy_search_results(search_query)

    def _get_dummy_search_results(self, search_query: str) -> List[Dict]:
        """
        Generate dummy search results for testing or when API keys are not available.

        Args:
            search_query: The search query

        Returns:
            List of dummy search results
        """
        # This is a simplified version for Phase 1
        # These would be replaced with actual search results in production
        return [
            {
                "name": "Translation API",
                "provider": "Google Cloud",
                "description": "Translate text between languages using Google's machine learning technology.",
                "documentation_url": "https://cloud.google.com/translate/docs",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            },
            {
                "name": "DeepL API",
                "provider": "DeepL",
                "description": "Neural machine translation API for high-quality translations.",
                "documentation_url": "https://www.deepl.com/docs-api",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            },
            {
                "name": "Azure Translator",
                "provider": "Microsoft",
                "description": "A cloud-based machine translation service supporting multiple languages.",
                "documentation_url": "https://docs.microsoft.com/en-us/azure/cognitive-services/translator/",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            }
        ]

    def filter_api_documentation(self, search_results: List[Dict]) -> List[Dict]:
        """
        Filter search results to include only those with accessible documentation.

        Args:
            search_results: Raw search results

        Returns:
            Filtered search results
        """
        # For Phase 1, this is a pass-through filter
        # In later phases, this would check documentation accessibility
        return search_results

    async def rank_by_relevance(
        self, 
        filtered_results: List[Dict], 
        capability_description: str
    ) -> List[APICandidate]:
        """
        Rank filtered results by relevance to the capability description.

        Args:
            filtered_results: Filtered search results
            capability_description: Description of the required capability

        Returns:
            List of ranked API candidates
        """
        # Generate embedding for capability description
        capability_embedding = await get_embedding(capability_description)

        result_candidates = []

        for result in filtered_results:
            # Generate embedding for the API description
            api_description = f"{result.get('name', '')} {result.get('description', '')}"
            api_embedding = await get_embedding(api_description)

            # Calculate cosine similarity
            similarity = self._calculate_cosine_similarity(capability_embedding, api_embedding)

            # Create API candidate
            candidate = APICandidate(
                name=result.get("name", "Unknown API"),
                provider=result.get("provider", "Unknown Provider"),
                description=result.get("description", ""),
                documentation_url=result.get("documentation_url", ""),
                requires_auth=result.get("requires_auth", True),
                auth_type=result.get("auth_type", None),
                pricing_tier=result.get("pricing_tier", None),
                relevance_score=similarity
            )

            result_candidates.append(candidate)

        # Sort by relevance score (descending)
        return sorted(result_candidates, key=lambda x: x.relevance_score, reverse=True)

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (between 0 and 1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    async def search_for_api(
        self, 
        capability_description: str, 
        num_results: int = 5
    ) -> List[APICandidate]:
        """
        Search for APIs that can fulfill the capability description.

        Args:
            capability_description: Description of the required capability
            num_results: Maximum number of results to return

        Returns:
            List of API candidates
        """
        search_query = self.formulate_search_query(capability_description)
        search_results = await self.execute_web_search(search_query)
        filtered_results = self.filter_api_documentation(search_results)
        ranked_results = await self.rank_by_relevance(filtered_results, capability_description)

        return ranked_results[:num_results]