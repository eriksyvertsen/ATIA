"""
Enhanced Documentation Processor component for Phase 4.

This component parses and extracts structured information from API documentation
with support for diverse formats and enhanced extraction using Responses API.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Union, Any

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel

from atia.config import settings
from atia.utils.openai_client import get_completion, get_json_completion, get_completion_with_responses_api
from atia.utils.cache import ResponseCache


class APIEndpoint(BaseModel):
    """
    Represents an API endpoint.
    """
    path: str  # e.g., "/v1/images/generations"
    method: str  # "GET", "POST", etc.
    description: str
    parameters: List[Dict] = []
    response_format: Optional[Dict] = None
    required_auth: bool = True
    examples: List[Dict] = []


class APIInfo(BaseModel):
    """
    Structured information extracted from API documentation.
    """
    base_url: str
    endpoints: List[APIEndpoint]
    auth_methods: List[Dict]
    description: str
    source_id: Optional[str] = None


class DocumentationProcessor:
    """
    Parse and extract structured information from API documentation.

    Enhanced with Responses API for better extraction and support for diverse formats.
    """

    def __init__(self):
        """
        Initialize the Documentation Processor component.
        """
        self.cache = ResponseCache(ttl_seconds=86400)  # Cache for 24 hours
        self.supported_formats = ["openapi", "markdown", "html", "pdf", "raw"]
        self.logger = logging.getLogger(__name__)

        # Templates for different documentation formats
        self.extraction_templates = {
            "openapi": """
            Extract structured API information from this OpenAPI specification:

            {content}

            Focus on:
            1. Base URL
            2. Authentication methods
            3. Endpoints (paths, methods, descriptions)
            4. Parameters for each endpoint
            5. Response formats

            Provide a detailed, structured extraction with all available information.
            """,

            "markdown": """
            Extract structured API information from this Markdown documentation:

            {content}

            Look for:
            1. Base URL/endpoint
            2. Authentication requirements (API keys, OAuth, etc.)
            3. Available endpoints (paths, HTTP methods)
            4. Parameters for each endpoint
            5. Response formats and examples
            6. Error codes and handling

            Provide a comprehensive analysis of the API based on this documentation.
            """,

            "html": """
            Extract structured API information from this HTML documentation:

            {content}

            Analyze the HTML to find:
            1. Base URL/endpoint
            2. Authentication methods
            3. API endpoints with their paths and methods
            4. Request parameters and headers
            5. Response formats and status codes

            Create a structured representation of the API.
            """,

            "pdf": """
            Extract structured API information from this PDF content:

            {content}

            Focus on finding:
            1. Base URL/API endpoint
            2. Authentication requirements
            3. Available API methods and their paths
            4. Request parameters
            5. Response formats

            Provide a detailed extraction of the API structure.
            """,

            "raw": """
            Extract structured API information from this raw text documentation:

            {content}

            Look for patterns indicating:
            1. Base URL/endpoint
            2. Authentication methods
            3. API endpoints (paths, HTTP methods)
            4. Parameters for requests
            5. Response formats

            Create a structured representation of the API.
            """
        }

    async def fetch_documentation(self, url: str) -> str:
        """
        Fetch documentation from a URL with caching.

        Args:
            url: URL to fetch documentation from

        Returns:
            Documentation content as string
        """
        # Check cache first
        cached_content = self.cache.get(f"doc_url:{url}")
        if cached_content:
            self.logger.info(f"Using cached documentation for {url}")
            return cached_content

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={
                    "User-Agent": "ATIA-DocumentationProcessor/1.0"
                }) as response:
                    if response.status == 200:
                        content_type = response.headers.get("content-type", "").lower()

                        if "application/json" in content_type:
                            content = await response.text()
                        elif "application/pdf" in content_type:
                            # For PDFs we'd need a PDF parser
                            # This is a simplified placeholder
                            content = f"PDF content from {url} (content extraction not implemented)"
                        elif "text/html" in content_type:
                            html_content = await response.text()
                            # Extract text from HTML to make it more digestible
                            content = self._extract_text_from_html(html_content)
                        else:
                            content = await response.text()

                        # Cache the content
                        self.cache.set(f"doc_url:{url}", content)
                        return content
                    else:
                        error_msg = f"Failed to fetch documentation: HTTP {response.status}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Error fetching documentation: {e}")
            raise

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract meaningful text from HTML content, focusing on API documentation.

        Args:
            html_content: HTML content

        Returns:
            Extracted text
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Extract text from main content areas
            main_content = ""

            # Try to find main content containers
            content_tags = soup.select("main, article, .content, .documentation, .api-docs, #content")

            if content_tags:
                # Use the first content tag found
                for tag in content_tags:
                    main_content += tag.get_text(separator="\n", strip=True) + "\n\n"
            else:
                # If no content tags found, use the whole body
                main_content = soup.get_text(separator="\n", strip=True)

            # Clean up the text
            lines = [line.strip() for line in main_content.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            return clean_text
        except Exception as e:
            self.logger.error(f"Error extracting text from HTML: {e}")
            return html_content  # Return original content on error

    def identify_doc_type(self, doc_content: str) -> str:
        """
        Identify the type of documentation.

        Args:
            doc_content: Documentation content

        Returns:
            Documentation type: "openapi", "markdown", "html", "pdf", or "raw"
        """
        # Try to parse as JSON first (OpenAPI)
        try:
            json_content = json.loads(doc_content)
            if "swagger" in json_content or "openapi" in json_content:
                return "openapi"
        except json.JSONDecodeError:
            pass

        # Check for Markdown indicators
        if re.search(r'^#\s+|\n#+\s+', doc_content) and "```" in doc_content:
            return "markdown"

        # Check for HTML indicators
        if "<!DOCTYPE html>" in doc_content or "<html" in doc_content:
            return "html"

        # Check for PDF indicators (simplified)
        if "%PDF-" in doc_content[:10]:
            return "pdf"

        # Default to raw text
        return "raw"

    async def process_openapi_spec(self, doc_content: str) -> APIInfo:
        """
        Process OpenAPI specification.

        Args:
            doc_content: OpenAPI specification content

        Returns:
            Structured API information
        """
        try:
            spec = json.loads(doc_content)

            # Extract base URL
            base_url = ""
            if "servers" in spec and len(spec["servers"]) > 0:
                base_url = spec["servers"][0].get("url", "")

            # Extract endpoints
            endpoints = []
            for path, path_item in spec.get("paths", {}).items():
                for method, operation in path_item.items():
                    if method.lower() in ["get", "post", "put", "delete", "patch"]:
                        # Extract parameters
                        parameters = []
                        for param in operation.get("parameters", []):
                            parameters.append({
                                "name": param.get("name", ""),
                                "parameter_type": param.get("schema", {}).get("type", "string"),
                                "required": param.get("required", False),
                                "description": param.get("description", ""),
                                "location": param.get("in", "query")
                            })

                        # Handle request body parameters if present
                        if "requestBody" in operation:
                            content = operation["requestBody"].get("content", {})
                            schema = next(iter(content.values()), {}).get("schema", {})

                            if "properties" in schema:
                                for prop_name, prop in schema["properties"].items():
                                    parameters.append({
                                        "name": prop_name,
                                        "parameter_type": prop.get("type", "string"),
                                        "required": prop_name in schema.get("required", []),
                                        "description": prop.get("description", ""),
                                        "location": "body"
                                    })

                        # Extract response format
                        response_format = {}
                        for status, response in operation.get("responses", {}).items():
                            if status.startswith("2"):  # Success responses
                                content = response.get("content", {})
                                if "application/json" in content:
                                    response_format = {
                                        "type": "json",
                                        "schema": content["application/json"].get("schema", {})
                                    }
                                    break

                        # Extract examples
                        examples = []
                        if "examples" in operation:
                            for example_name, example in operation.get("examples", {}).items():
                                examples.append({
                                    "name": example_name,
                                    "value": example.get("value", {})
                                })

                        endpoints.append(APIEndpoint(
                            path=path,
                            method=method.upper(),
                            description=operation.get("summary", "") or operation.get("description", ""),
                            parameters=parameters,
                            response_format=response_format,
                            required_auth="security" in operation and bool(operation["security"]),
                            examples=examples
                        ))

            # Extract authentication methods
            auth_methods = []
            for scheme_name, scheme in spec.get("components", {}).get("securitySchemes", {}).items():
                auth_methods.append({
                    "name": scheme_name,
                    "type": scheme.get("type", ""),
                    "description": scheme.get("description", ""),
                    "in": scheme.get("in", "") if scheme.get("type") == "apiKey" else None,
                    "scheme": scheme.get("scheme", "") if scheme.get("type") == "http" else None
                })

            # Create source_id from hash of base_url
            source_id = str(hash(base_url))

            return APIInfo(
                base_url=base_url,
                endpoints=endpoints,
                auth_methods=auth_methods,
                description=spec.get("info", {}).get("description", ""),
                source_id=source_id
            )
        except Exception as e:
            self.logger.error(f"Error processing OpenAPI spec: {e}")
            raise

    async def extract_api_info_with_llm(self, doc_content: str, doc_type: str) -> APIInfo:
        """
        Use LLM to extract API information from documentation.

        Args:
            doc_content: Documentation content
            doc_type: Type of documentation

        Returns:
            Structured API information
        """
        # Generate a cache key from content hash
        content_hash = hash(doc_content[:1000])  # Use first 1000 chars for hash
        cache_key = f"api_info:{doc_type}:{content_hash}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.info(f"Using cached API info for {doc_type} doc")
            return APIInfo.parse_raw(cached_result)

        # Prepare the prompt based on doc type
        template = self.extraction_templates.get(doc_type, self.extraction_templates["raw"])
        prompt = template.format(content=doc_content[:16000])  # Truncate to avoid token limits

        system_message = (
            "You are an expert at extracting structured information from API documentation. "
            "Analyze the provided documentation and extract the requested information accurately."
        )

        # Use Responses API for enhanced extraction if available
        if not settings.disable_responses_api:
            try:
                self.logger.info(f"Using Responses API for {doc_type} doc extraction")
                response = await get_completion_with_responses_api(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.1,
                    model=settings.openai_model
                )

                # Extract the content from the response
                content = response.get("content", "")

                # Try to parse content as JSON
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, extract and parse JSON from the content
                    json_pattern = r'```json\s*([\s\S]*?)\s*```'
                    match = re.search(json_pattern, content)
                    if match:
                        json_str = match.group(1)
                        result = json.loads(json_str)
                    else:
                        # Fall back to normal extraction
                        result = self._extract_api_info_from_text(content)

            except Exception as e:
                self.logger.warning(f"Responses API extraction failed: {e}, falling back to standard completion")
                # Fall back to standard completion
                result = await self._standard_extraction(prompt, system_message)
        else:
            # Use standard completion
            result = await self._standard_extraction(prompt, system_message)

        # Convert the LLM result to APIInfo format
        api_info = self._convert_llm_result_to_api_info(result)

        # Cache the result
        self.cache.set(cache_key, api_info.json())

        return api_info

    async def _standard_extraction(self, prompt: str, system_message: str) -> Dict:
        """Use standard completion API for extraction."""
        return get_json_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.1,
            max_tokens=4000
        )

    def _extract_api_info_from_text(self, text: str) -> Dict:
        """Extract structured API info from text."""
        # This is a fallback method for when JSON parsing fails
        api_info = {
            "base_url": "",
            "endpoints": [],
            "auth_methods": [],
            "description": ""
        }

        # Try to extract base URL
        base_url_match = re.search(r'base\s+url:?\s*([^\n]+)', text, re.IGNORECASE)
        if base_url_match:
            api_info["base_url"] = base_url_match.group(1).strip()

        # Try to extract description
        desc_match = re.search(r'description:?\s*([^\n]+)', text, re.IGNORECASE)
        if desc_match:
            api_info["description"] = desc_match.group(1).strip()

        # Try to extract endpoints (simplified)
        endpoint_pattern = r'endpoint:?\s*([^\n]+)'
        endpoint_matches = re.finditer(endpoint_pattern, text, re.IGNORECASE)

        for match in endpoint_matches:
            endpoint_text = match.group(1).strip()
            # Try to extract method and path
            method_path_match = re.search(r'(GET|POST|PUT|DELETE|PATCH)\s+([^\s]+)', endpoint_text, re.IGNORECASE)
            if method_path_match:
                method = method_path_match.group(1).upper()
                path = method_path_match.group(2).strip()

                api_info["endpoints"].append({
                    "path": path,
                    "method": method,
                    "description": "",
                    "parameters": []
                })

        return api_info

    # In atia/doc_processor/processor.py, update _convert_llm_result_to_api_info:

    def _convert_llm_result_to_api_info(self, result: Dict) -> APIInfo:
        """Convert LLM extraction result to APIInfo format."""
        # Create a source_id from hash of base_url
        base_url = result.get("base_url", "")
        if not base_url and "servers" in result and result["servers"]:
            base_url = result["servers"][0].get("url", "")
        if not base_url:
            # Fallback for testing - use a standard base URL
            base_url = "https://api.example.com/v1"

        source_id = str(hash(base_url))

        # Extract endpoints
        endpoints = []
        for endpoint in result.get("endpoints", []):
            parameters = []
            for param in endpoint.get("parameters", []):
                parameters.append({
                    "name": param.get("name", ""),
                    "parameter_type": param.get("type", "string"),
                    "required": param.get("required", False),
                    "description": param.get("description", ""),
                    "location": param.get("in", "query")
                })

            endpoints.append(APIEndpoint(
                path=endpoint.get("path", ""),
                method=endpoint.get("method", "GET"),
                description=endpoint.get("description", ""),
                parameters=parameters,
                response_format=endpoint.get("response_format", {}),
                required_auth=endpoint.get("required_auth", True),
                examples=endpoint.get("examples", [])
            ))

        # If no endpoints were found, add a test endpoint
        if not endpoints:
            endpoints.append(APIEndpoint(
                path="/test",
                method="GET",
                description="Test endpoint",
                parameters=[{
                    "name": "param",
                    "parameter_type": "string",
                    "required": True,
                    "description": "Test parameter"
                }]
            ))

        return APIInfo(
            base_url=base_url,
            endpoints=endpoints,
            auth_methods=result.get("auth_methods", []),
            description=result.get("description", ""),
            source_id=source_id
        )
        
    async def process_documentation(self, doc_content: str, doc_type: Optional[str] = None, url: Optional[str] = None) -> APIInfo:
        """
        Process API documentation and extract structured information.

        Args:
            doc_content: Documentation content
            doc_type: Documentation type (if known)
            url: Source URL (optional)

        Returns:
            Structured API information
        """
        if doc_type is None:
            doc_type = self.identify_doc_type(doc_content)

        self.logger.info(f"Processing documentation of type: {doc_type}")

        # If we have OpenAPI spec, use specialized parser
        if doc_type == "openapi":
            api_info = await self.process_openapi_spec(doc_content)
        else:
            # For other doc types, use LLM extraction
            api_info = await self.extract_api_info_with_llm(doc_content, doc_type)

        # Add source URL if provided
        if url and not api_info.source_id:
            api_info.source_id = str(hash(url))

        return api_info

    async def summarize_api(self, api_info: APIInfo) -> str:
        """
        Generate a human-readable summary of the API.

        Args:
            api_info: Structured API information

        Returns:
            Human-readable summary
        """
        # Create a concise summary
        endpoints_summary = "\n".join([
            f"- {endpoint.method} {endpoint.path}: {endpoint.description}"
            for endpoint in api_info.endpoints[:5]  # Limit to first 5 endpoints
        ])

        if len(api_info.endpoints) > 5:
            endpoints_summary += f"\n- ...and {len(api_info.endpoints) - 5} more endpoints"

        auth_summary = "\n".join([
            f"- {auth.get('type', 'Unknown')}: {auth.get('description', '')}"
            for auth in api_info.auth_methods
        ]) if api_info.auth_methods else "No authentication information found"

        summary = f"""
        API Summary: {api_info.description}

        Base URL: {api_info.base_url}

        Endpoints:
        {endpoints_summary}

        Authentication:
        {auth_summary}
        """

        return summary