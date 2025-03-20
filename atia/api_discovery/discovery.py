"""
Enhanced API Discovery component for Phase 4.

This component searches for and evaluates APIs that can fulfill an identified tool need,
with improved search capabilities and Responses API integration.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import hashlib
import socket

import aiohttp
import numpy as np
from numpy.linalg import norm
from pydantic import BaseModel

from atia.config import settings
from atia.utils.openai_client import get_completion, get_embedding, get_completion_with_responses_api
from atia.utils.cache import ResponseCache, VectorCache


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
    capabilities: List[str] = []
    popularity: Optional[float] = None
    last_updated: Optional[str] = None


class APIDiscovery:
    """
    Search for relevant APIs that can fulfill the identified tool need.

    Enhanced with Responses API for improved search capabilities and evaluation.
    """

    def __init__(self):
        """
        Initialize the API Discovery component with enhanced reliability.
        """
        self.rapid_api_key = settings.rapid_api_key
        self.serp_api_key = settings.serp_api_key
        self.github_token = settings.github_token
        self.api_cache = ResponseCache(ttl_seconds=86400)  # Cache API search results for 24 hours
        self.vector_cache = VectorCache()  # Cache for embeddings
        self.logger = logging.getLogger(__name__)

        # Check network availability
        self.network_available = self._check_network_availability()

        # APIs to search (in order of preference)
        if self.network_available:
            self.search_providers = [
                self._search_rapidapi,
                self._search_apis_guru,     # Alternative search provider
                self._search_github_repos,
                self._search_programmatic_web,
                self._fallback_search       # Fallback still included at the end
            ]
            self.logger.info("Network is available, using all API search providers")
        else:
            # If network isn't available, only use fallback
            self.search_providers = [self._fallback_search]
            self.logger.warning("Network appears unavailable, using only fallback search provider")

    def _check_network_availability(self) -> bool:
        """
        Check if the network is available by trying to resolve a well-known hostname.

        Returns:
            True if network is available, False otherwise
        """
        try:
            # Try to resolve google.com as a basic network check
            socket.gethostbyname("google.com")
            return True
        except:
            # Any error means network might not be available
            return False

    async def formulate_search_query(self, capability_description: str) -> str:
        """
        Generate an effective search query based on capability description.

        Args:
            capability_description: Description of the required capability

        Returns:
            Formulated search query
        """
        # Check cache first
        cache_key = f"search_query:{hashlib.md5(capability_description.encode()).hexdigest()}"
        cached_query = self.api_cache.get(cache_key)
        if cached_query:
            self.logger.info(f"Using cached search query for: {capability_description[:50]}...")
            return cached_query

        prompt = f"Generate a search query to find APIs that provide this capability: {capability_description}"

        system_message = (
            "You are an expert at formulating search queries for finding APIs. "
            "Generate a concise, effective search query that would find relevant APIs."
        )

        # Try to use Responses API if available
        if not settings.disable_responses_api:
            try:
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    model="gpt-4o-mini"  # Using a smaller model for efficiency
                )
                search_query = response.get("content", "").strip()
            except Exception as e:
                self.logger.warning(f"Error using Responses API for search query: {e}")
                # Fall back to standard completion
                search_query = get_completion(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,
                    max_tokens=100,
                    model="gpt-4o-mini"
                )
        else:
            # Use standard completion
            search_query = get_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,
                max_tokens=100,
                model="gpt-4o-mini"
            )

        # Cache the query
        self.api_cache.set(cache_key, search_query)

        return search_query

    async def execute_web_search(self, search_query: str) -> List[Dict]:
        """
        Execute a web search for APIs using the formulated query with enhanced reliability.

        Args:
            search_query: The formulated search query

        Returns:
            List of search results
        """
        # Check cache first
        cache_key = f"web_search:{hashlib.md5(search_query.encode()).hexdigest()}"
        cached_results = self.api_cache.get(cache_key)
        if cached_results:
            self.logger.info(f"Using cached web search results for: {search_query}")
            return cached_results

        # Run searches in parallel with multiple providers
        all_results = []
        search_tasks = []

        for search_provider in self.search_providers:
            search_tasks.append(search_provider(search_query))

        # Wait for all searches to complete
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results, skipping any that failed
        for result in search_results:
            if isinstance(result, Exception):
                self.logger.warning(f"Search provider failed: {result}")
                continue

            all_results.extend(result)

        # If we have no results, use the fallback
        if not all_results:
            self.logger.warning("All search providers failed, using hardcoded fallback")
            all_results = self._get_hardcoded_fallback(search_query)

        # Ensure all results have a valid description
        for result in all_results:
            if result.get("description") is None or not isinstance(result.get("description"), str):
                result["description"] = f"API for {search_query}"

        # Log summary of results
        self.logger.info(f"API search found {len(all_results)} total results")

        # Cache the results
        self.api_cache.set(cache_key, all_results)

        return all_results

    async def _search_rapidapi(self, search_query: str) -> List[Dict]:
        """Search RapidAPI Hub for APIs using direct IP addressing when needed."""
        if not self.rapid_api_key:
            self.logger.info("No RapidAPI key provided, skipping RapidAPI search")
            return []

        try:
            # Create ClientSession with DNS cache enabled and increased timeout
            connector = aiohttp.TCPConnector(
                ssl=False,  # Disable SSL verification for troubleshooting
                family=socket.AF_INET,  # Use IPv4 only
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {
                    "X-RapidAPI-Key": self.rapid_api_key,
                    "X-RapidAPI-Host": "api.rapidapi.com"
                }

                params = {
                    "q": search_query,
                    "sort": "score",
                    "limit": "10"
                }

                # Try with hostname first
                url = "https://api.rapidapi.com/search"

                try:
                    self.logger.info(f"Attempting RapidAPI search with hostname: {url}")
                    async with session.get(
                        url, 
                        headers=headers,
                        params=params,
                        timeout=15  # Increased timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Transform to our format
                            results = []
                            for item in data.get("results", []):
                                results.append({
                                    "name": item.get("name", "Unknown API"),
                                    "provider": item.get("provider", {}).get("name", "RapidAPI"),
                                    "description": item.get("description") or f"API on RapidAPI related to {search_query}",
                                    "documentation_url": item.get("documentationUrl", ""),
                                    "requires_auth": True,
                                    "auth_type": "api_key",
                                    "pricing_tier": item.get("pricing", {}).get("model", "unknown"),
                                    "popularity": item.get("score", 0)
                                })

                            self.logger.info(f"RapidAPI search returned {len(results)} results")
                            return results
                        else:
                            self.logger.error(f"RapidAPI search failed: HTTP {response.status}")
                            return []
                except aiohttp.ClientConnectorError as e:
                    # If hostname resolution fails, try with hardcoded IP address
                    self.logger.warning(f"Could not connect to RapidAPI with hostname: {e}, trying with IP address")

                    # Try with hardcoded IP (this IP may change, should be updated periodically)
                    # This is a fallback approach when DNS resolution fails
                    rapidapi_ip = "104.18.23.30"  # Example IP - might need updating
                    url_with_ip = f"https://{rapidapi_ip}/search"

                    try:
                        # Add Host header when using direct IP
                        ip_headers = headers.copy()
                        ip_headers["Host"] = "api.rapidapi.com"

                        async with session.get(
                            url_with_ip, 
                            headers=ip_headers,
                            params=params,
                            timeout=15
                        ) as response:
                            if response.status == 200:
                                data = await response.json()

                                # Transform to our format (same as above)
                                results = []
                                for item in data.get("results", []):
                                    results.append({
                                        "name": item.get("name", "Unknown API"),
                                        "provider": item.get("provider", {}).get("name", "RapidAPI"),
                                        "description": item.get("description") or f"API on RapidAPI related to {search_query}",
                                        "documentation_url": item.get("documentationUrl", ""),
                                        "requires_auth": True,
                                        "auth_type": "api_key",
                                        "pricing_tier": item.get("pricing", {}).get("model", "unknown"),
                                        "popularity": item.get("score", 0)
                                    })

                                self.logger.info(f"RapidAPI search with IP address returned {len(results)} results")
                                return results
                            else:
                                self.logger.error(f"RapidAPI search with IP failed: HTTP {response.status}")
                                return []
                    except Exception as ip_e:
                        self.logger.error(f"RapidAPI search with IP address failed: {ip_e}")
                        return []

        except asyncio.TimeoutError:
            self.logger.error("RapidAPI connection timeout")
            return []
        except Exception as e:
            self.logger.error(f"RapidAPI search error: {str(e)}")
            return []

    async def _search_apis_guru(self, search_query: str) -> List[Dict]:
        """
        Search APIs.guru directory for APIs matching the query.

        Args:
            search_query: The search query string

        Returns:
            List of API results
        """
        self.logger.info(f"Searching APIs.guru for: {search_query}")

        try:
            async with aiohttp.ClientSession() as session:
                # APIs.guru provides a directory of OpenAPI specifications
                url = "https://api.apis.guru/v2/list.json"

                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        apis_list = await response.json()

                        # Search through the APIs
                        search_terms = search_query.lower().split()
                        results = []

                        for api_name, api_versions in apis_list.items():
                            # Get the latest version's info
                            api_info = api_versions.get("versions", {}).get(
                                api_versions.get("preferred", ""), {})

                            # Extract relevant information
                            info = api_info.get("info", {})
                            title = info.get("title", "")
                            description = info.get("description", "")

                            # Check if search terms match
                            text_to_search = f"{title} {description} {api_name}".lower()
                            if any(term in text_to_search for term in search_terms):
                                results.append({
                                    "name": title or api_name,
                                    "provider": api_name.split(":")[0] if ":" in api_name else "Unknown",
                                    "description": description or f"API related to {search_query}",
                                    "documentation_url": info.get("x-origin", [{}])[0].get("url", ""),
                                    "requires_auth": "security" in api_info or "securityDefinitions" in api_info,
                                    "auth_type": self._detect_auth_type(api_info),
                                })

                        self.logger.info(f"APIs.guru search returned {len(results)} results")
                        return results[:10]  # Limit to top 10 results
                    else:
                        self.logger.error(f"APIs.guru search failed: HTTP {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"APIs.guru search error: {str(e)}")
            return []

    async def _search_programmatic_web(self, search_query: str) -> List[Dict]:
        """Search ProgrammableWeb for APIs."""
        try:
            async with aiohttp.ClientSession() as session:
                # ProgrammableWeb doesn't have a public API, so we'd need to scrape their site
                # This is a placeholder for a real implementation
                await asyncio.sleep(0.1)  # Simulate a search
                return []
        except Exception as e:
            self.logger.error(f"Error searching ProgrammableWeb: {e}")
            return []

    async def _search_github_repos(self, search_query: str) -> List[Dict]:
        """Search GitHub for API repositories."""
        if not self.github_token:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"token {self.github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }

                params = {
                    "q": f"{search_query} api in:name,description,readme",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 10
                }

                url = "https://api.github.com/search/repositories"

                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Transform to our format
                        results = []
                        for item in data.get("items", []):
                            results.append({
                                "name": item.get("name", "Unknown Repository"),
                                "provider": "GitHub",
                                "description": item.get("description") or f"GitHub repository related to {search_query}",
                                "documentation_url": item.get("html_url", ""),
                                "requires_auth": False,
                                "popularity": item.get("stargazers_count", 0),
                                "last_updated": item.get("updated_at", "")
                            })

                        return results
                    else:
                        self.logger.error(f"Error searching GitHub: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error searching GitHub: {e}")
            return []

    async def _fallback_search(self, search_query: str) -> List[Dict]:
        """
        Fallback search provider that returns hardcoded results when other providers fail.

        Args:
            search_query: The search query string

        Returns:
            List of hardcoded API results
        """
        self.logger.info("Using fallback search method with hardcoded results")
        return self._get_hardcoded_fallback(search_query)

    def _get_hardcoded_fallback(self, search_query: str) -> List[Dict]:
        """
        Provide hardcoded fallback API results for testing.

        Args:
            search_query: Search query (used to make results relevant)

        Returns:
            List of hardcoded API results
        """
        # Generic fallback APIs that could be useful for many use cases
        fallback_apis = [
            {
                "name": "OpenAI API",
                "provider": "OpenAI",
                "description": "API for language processing, text generation, embeddings, and more.",
                "documentation_url": "https://platform.openai.com/docs/",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "paid"
            },
            {
                "name": "Google Translate API",
                "provider": "Google",
                "description": "API for translating text between languages.",
                "documentation_url": "https://cloud.google.com/translate/docs",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            },
            {
                "name": "Weather API",
                "provider": "OpenWeatherMap",
                "description": "Provides weather data including current weather, forecasts, and historical data.",
                "documentation_url": "https://openweathermap.org/api",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            },
            {
                "name": "Cloudinary API",
                "provider": "Cloudinary",
                "description": "Cloud-based image and video management services.",
                "documentation_url": "https://cloudinary.com/documentation/",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            },
            {
                "name": "News API",
                "provider": "NewsAPI.org",
                "description": "API for accessing breaking headlines and searching articles from news sources and blogs.",
                "documentation_url": "https://newsapi.org/docs",
                "requires_auth": True,
                "auth_type": "api_key",
                "pricing_tier": "freemium"
            }
        ]

        return fallback_apis

    def filter_api_documentation(self, api_results: List[Dict]) -> List[Dict]:
        """
        Filter API results to ensure valid documentation URLs.

        Args:
            api_results: List of API results from search

        Returns:
            Filtered list of API results
        """
        filtered_results = []
        for api in api_results:
            # Check if documentation URL exists and is valid
            doc_url = api.get("documentation_url")
            if doc_url and isinstance(doc_url, str) and doc_url.startswith(("http://", "https://")):
                filtered_results.append(api)

        return filtered_results

    async def rank_by_relevance(self, api_results: List[Dict], 
                              capability_description: str) -> List[APICandidate]:
        """
        Rank API results by relevance to the capability description.

        Args:
            api_results: List of API results
            capability_description: Description of the required capability

        Returns:
            List of API candidates ranked by relevance
        """
        if not api_results:
            return []

        # Generate embedding for capability description
        try:
            capability_embedding = await get_embedding(capability_description)
        except Exception as e:
            self.logger.warning(f"Could not generate embedding for capability: {e}")
            # Fall back to simple keyword matching
            return self._keyword_rank_fallback(api_results, capability_description)

        # Calculate relevance scores using embeddings
        candidates = []
        for api in api_results:
            # Generate embedding for API description
            description = api.get("description", "")

            # Use cached embedding if available
            api_embedding = self.vector_cache.get(description)
            if api_embedding is None:
                try:
                    api_embedding = await get_embedding(description)
                    self.vector_cache.set(description, api_embedding)
                except Exception as e:
                    self.logger.warning(f"Error generating embedding for API {api.get('name')}: {e}")
                    # Skip this API if we can't generate an embedding
                    continue

            # Calculate cosine similarity
            similarity = self._calculate_cosine_similarity(capability_embedding, api_embedding)

            # Create APICandidate
            candidate = APICandidate(
                name=api.get("name", "Unknown API"),
                provider=api.get("provider", "Unknown"),
                description=description,
                documentation_url=api.get("documentation_url", ""),
                requires_auth=api.get("requires_auth", True),
                auth_type=api.get("auth_type"),
                pricing_tier=api.get("pricing_tier"),
                relevance_score=float(similarity),
                popularity=api.get("popularity"),
                last_updated=api.get("last_updated")
            )

            candidates.append(candidate)

        # Sort by relevance score (highest first)
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        return candidates

    def _keyword_rank_fallback(self, api_results: List[Dict], 
                             capability_description: str) -> List[APICandidate]:
        """
        Fallback ranking method using keyword matching.

        Args:
            api_results: List of API results
            capability_description: Description of the required capability

        Returns:
            List of API candidates ranked by relevance
        """
        # Extract keywords from capability description
        keywords = set(capability_description.lower().split())

        candidates = []
        for api in api_results:
            description = api.get("description", "").lower()
            name = api.get("name", "").lower()

            # Count keyword matches
            match_count = sum(1 for keyword in keywords if keyword in description or keyword in name)
            relevance_score = min(match_count / max(len(keywords), 1), 1.0)

            candidate = APICandidate(
                name=api.get("name", "Unknown API"),
                provider=api.get("provider", "Unknown"),
                description=api.get("description", ""),
                documentation_url=api.get("documentation_url", ""),
                requires_auth=api.get("requires_auth", True),
                auth_type=api.get("auth_type"),
                pricing_tier=api.get("pricing_tier"),
                relevance_score=relevance_score,
                popularity=api.get("popularity"),
                last_updated=api.get("last_updated")
            )

            candidates.append(candidate)

        # Sort by relevance score (highest first)
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        return candidates

    async def evaluate_api_candidates(self, candidates: List[APICandidate], 
                                   capability_description: str) -> List[APICandidate]:
        """
        Evaluate API candidates using LLM to determine their suitability.

        Args:
            candidates: List of API candidates
            capability_description: Description of the required capability

        Returns:
            List of API candidates with updated relevance scores and capabilities
        """
        if not candidates or len(candidates) == 0:
            return []

        # Only evaluate top 5 candidates for efficiency
        top_candidates = candidates[:5]

        # Check if Responses API is disabled
        if settings.disable_responses_api:
            self.logger.info("Responses API disabled, skipping candidate evaluation")
            return candidates

        # Evaluate each candidate
        evaluated_candidates = []
        for candidate in top_candidates:
            try:
                # Construct evaluation prompt
                prompt = f"""
                Evaluate how well this API matches the required capability:

                Capability: {capability_description}

                API: {candidate.name}
                Provider: {candidate.provider}
                Description: {candidate.description}
                Documentation: {candidate.documentation_url}

                Please evaluate:
                1. Relevance score (0.0 to 1.0) for how well this API matches the capability
                2. List of specific capabilities this API provides related to the requirement
                """

                system_message = (
                    "You are an expert at evaluating APIs for their suitability to fulfill specific capabilities. "
                    "Provide a fair and accurate assessment."
                )

                # Use Responses API for evaluation
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.2,
                    model=settings.openai_model
                )

                content = response.get("content", "")

                # Extract relevance score and capabilities from the response
                relevance_score = self._extract_relevance_score(content)
                capabilities = self._extract_capabilities(content)

                # Update candidate with evaluation results
                candidate.relevance_score = relevance_score or candidate.relevance_score
                candidate.capabilities = capabilities

                evaluated_candidates.append(candidate)

            except Exception as e:
                self.logger.error(f"Error evaluating candidate {candidate.name}: {e}")
                # Keep the original candidate if evaluation fails
                evaluated_candidates.append(candidate)

        # Sort by relevance score
        evaluated_candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        # Add any remaining candidates
        if len(candidates) > 5:
            evaluated_candidates.extend(candidates[5:])

        return evaluated_candidates

    def _extract_relevance_score(self, content: str) -> Optional[float]:
        """Extract relevance score from evaluation content."""
        try:
            # Look for patterns like "Relevance score: 0.85" or similar
            import re
            score_match = re.search(r'relevance\s+score:?\s*(0\.\d+|1\.0|1)', content, re.IGNORECASE)
            if score_match:
                return float(score_match.group(1))

            # Try to find any number between 0 and 1
            number_match = re.search(r'(0\.\d+|1\.0|1)', content)
            if number_match:
                return float(number_match.group(1))

            return None
        except Exception:
            return None

    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract capabilities list from evaluation content."""
        capabilities = []

        # Look for capabilities in a list format (numbered or bulleted)
        import re
        capability_matches = re.findall(r'[•\-\*\d+]\s+(.*?)(?=\n|$)', content)

        # If we found capabilities in list format, use them
        if capability_matches:
            capabilities = [cap.strip() for cap in capability_matches if cap.strip()]
        else:
            # Otherwise, try to extract entire capabilities section
            capabilities_section_match = re.search(r'capabilities:?(.*?)(?=\n\n|$)', 
                                                 content, 
                                                 re.IGNORECASE | re.DOTALL)
            if capabilities_section_match:
                capabilities_text = capabilities_section_match.group(1).strip()
                # Split by newlines or commas
                for line in re.split(r'[,\n]', capabilities_text):
                    cap = line.strip()
                    if cap and not cap.startswith(('0.', '1.')):  # Avoid relevance scores
                        capabilities.append(cap)

        return capabilities

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
        num_results: int = 5,
        evaluate: bool = True
    ) -> List[APICandidate]:
        """
        Search for APIs that can fulfill the capability description.

        Args:
            capability_description: Description of the required capability
            num_results: Maximum number of results to return
            evaluate: Whether to evaluate candidates using Responses API

        Returns:
            List of API candidates
        """
        # Check cache first
        cache_key = f"api_search:{hashlib.md5(capability_description.encode()).hexdigest()}"
        cached_results = self.api_cache.get(cache_key)
        if cached_results:
            # Deserialize cached results
            self.logger.info(f"Using cached API search results for: {capability_description[:50]}...")
            return [APICandidate.parse_raw(candidate) for candidate in cached_results]

        # Start timing
        import time
        start_time = time.time()

        search_query = await self.formulate_search_query(capability_description)
        self.logger.info(f"Search query: {search_query}")

        search_results = await self.execute_web_search(search_query)
        self.logger.info(f"Found {len(search_results)} search results")

        filtered_results = self.filter_api_documentation(search_results)
        self.logger.info(f"Filtered to {len(filtered_results)} results with documentation")

        # For testing: if we have no results, add mock results
        if not filtered_results:
            filtered_results = self._get_hardcoded_fallback(search_query)
            self.logger.info(f"Added {len(filtered_results)} mock results for testing")

        ranked_results = await self.rank_by_relevance(filtered_results, capability_description)
        self.logger.info(f"Ranked {len(ranked_results)} results by relevance")

        # Evaluate candidates if requested
        if evaluate and len(ranked_results) > 0:
            ranked_results = await self.evaluate_api_candidates(ranked_results, capability_description)
            self.logger.info("Evaluated API candidates")

        # Log timing
        end_time = time.time()
        self.logger.info(f"API search completed in {end_time - start_time:.2f} seconds")

        # Cache results
        self.api_cache.set(
            cache_key, 
            [candidate.json() for candidate in ranked_results[:num_results]]
        )

        return ranked_results[:num_results]

    def _detect_auth_type(self, api_info: Dict) -> str:
        """
        Detect authentication type from OpenAPI spec information.

        Args:
            api_info: OpenAPI specification information

        Returns:
            Detected authentication type
        """
        # Check security definitions
        security_defs = api_info.get("securityDefinitions", {})

        if not security_defs:
            return "unknown"

        for _, def_info in security_defs.items():
            auth_type = def_info.get("type", "").lower()

            if auth_type == "apikey":
                return "api_key"
            elif auth_type == "oauth2":
                return "oauth"
            elif auth_type == "http" and def_info.get("scheme", "").lower() == "bearer":
                return "bearer"
            elif auth_type == "http" and def_info.get("scheme", "").lower() == "basic":
                return "basic"

        return "api_key"  # Default assumption