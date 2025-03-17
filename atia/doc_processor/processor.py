import json
from typing import Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel

from atia.utils.openai_client import get_completion, get_json_completion


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


class DocumentationProcessor:
    """
    Parse and extract structured information from API documentation.

    This is a simplified version for Phase 1 that focuses on well-structured OpenAPI specs.
    """

    async def fetch_documentation(self, url: str) -> str:
        """
        Fetch documentation from a URL.

        Args:
            url: URL to fetch documentation from

        Returns:
            Documentation content as string
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        raise Exception(f"Failed to fetch documentation: {response.status}")
        except Exception as e:
            print(f"Error fetching documentation: {e}")
            raise

    def identify_doc_type(self, doc_content: str) -> str:
        """
        Identify the type of documentation.

        Args:
            doc_content: Documentation content

        Returns:
            Documentation type: "openapi", "markdown", "html", or "unknown"
        """
        # Try to parse as JSON first (OpenAPI)
        try:
            json_content = json.loads(doc_content)
            if "swagger" in json_content or "openapi" in json_content:
                return "openapi"
        except json.JSONDecodeError:
            pass

        # Check for Markdown indicators
        if doc_content.startswith("# ") or "## " in doc_content:
            return "markdown"

        # Check for HTML indicators
        if "<!DOCTYPE html>" in doc_content or "<html" in doc_content:
            return "html"

        # Default to unknown
        return "unknown"

    def process_openapi_spec(self, doc_content: str) -> APIInfo:
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

                        endpoints.append(APIEndpoint(
                            path=path,
                            method=method.upper(),
                            description=operation.get("summary", "") or operation.get("description", ""),
                            parameters=parameters,
                            response_format=response_format,
                            required_auth="security" in operation and bool(operation["security"])
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

            return APIInfo(
                base_url=base_url,
                endpoints=endpoints,
                auth_methods=auth_methods,
                description=spec.get("info", {}).get("description", "")
            )
        except Exception as e:
            print(f"Error processing OpenAPI spec: {e}")
            raise

    def extract_api_info_with_llm(self, doc_content: str) -> APIInfo:
        """
        Use LLM to extract API information from documentation.

        Args:
            doc_content: Documentation content

        Returns:
            Structured API information
        """
        prompt = f"""
        Extract the following information from this API documentation:
        1. Base URL
        2. Available endpoints (with paths, methods, and descriptions)
        3. Authentication methods
        4. Required headers
        5. Request parameters for each endpoint
        6. Response formats

        Documentation:
        {doc_content[:8000]}  # Truncate to avoid token limits

        Return the information in JSON format.
        """

        system_message = (
            "You are an expert at extracting structured information from API documentation. "
            "Analyze the provided documentation and extract the requested information accurately."
        )

        try:
            result = get_json_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,  # Lower temperature for more deterministic results
                max_tokens=4000  # Allow for longer responses
            )

            # Convert the LLM result to APIInfo format
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
                    required_auth=endpoint.get("required_auth", True)
                ))

            return APIInfo(
                base_url=result.get("base_url", ""),
                endpoints=endpoints,
                auth_methods=result.get("authentication_methods", []),
                description=result.get("description", "")
            )
        except Exception as e:
            print(f"Error extracting API info with LLM: {e}")
            raise

    async def process_documentation(self, doc_content: str, doc_type: Optional[str] = None) -> APIInfo:
        """
        Process API documentation and extract structured information.

        Args:
            doc_content: Documentation content
            doc_type: Documentation type (if known)

        Returns:
            Structured API information
        """
        if doc_type is None:
            doc_type = self.identify_doc_type(doc_content)

        if doc_type == "openapi":
            return self.process_openapi_spec(doc_content)
        else:
            # For Phase 1, fall back to LLM extraction for non-OpenAPI formats
            return self.extract_api_info_with_llm(doc_content)