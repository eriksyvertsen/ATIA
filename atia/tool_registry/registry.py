"""
Tool Registry implementation for maintaining a registry of integrated tools.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from atia.tool_registry.models import ToolRegistration
from atia.function_builder.models import FunctionDefinition
from atia.utils.openai_client import get_embedding
from atia.config import settings


logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Maintains a registry of integrated tools for reuse across sessions.
    """

    def __init__(self):
        """Initialize the Tool Registry."""
        self._tools = {}  # In-memory storage for Phase 2
        self._pinecone_initialized = False
        self._pinecone_index = None

        # Initialize Pinecone if API key is available
        if hasattr(settings, 'pinecone_api_key') and settings.pinecone_api_key:
            self._init_pinecone()

    def _init_pinecone(self):
        """Initialize Pinecone for vector search."""
        try:
            import pinecone
            from pinecone import Pinecone, ServerlessSpec

            # Initialize Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)

            # Check if our index exists, if not create it
            index_name = "atia-tools"
            existing_indexes = [idx["name"] for idx in pc.list_indexes()]

            if index_name not in existing_indexes:
                # Create a new serverless index
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI's ada embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )

            # Connect to the index
            self._pinecone_index = pc.Index(index_name)
            self._pinecone_initialized = True
            logger.info("Pinecone initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._pinecone_initialized = False

    async def register_function(self, 
                              function_def: FunctionDefinition, 
                              metadata: Dict[str, Any] = None) -> ToolRegistration:
        """
        Register a function in the tool registry.

        Args:
            function_def: Function definition to register
            metadata: Additional metadata for the function

        Returns:
            Tool registration
        """
        if metadata is None:
            metadata = {}

        # Generate embedding for the function
        embedding = await self._generate_embedding(function_def)

        # Create the tool registration
        tool = ToolRegistration(
            name=function_def.name,
            description=function_def.description,
            function_id=function_def.id,
            api_source_id=function_def.api_source_id,
            capability_tags=function_def.tags,
            metadata={
                **metadata,
                "endpoint": function_def.endpoint,
                "method": function_def.method,
                "api_type": function_def.api_type.value
            },
            embedding=embedding,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Store in memory
        self._tools[tool.id] = tool

        # Store in Pinecone if available
        if self._pinecone_initialized and embedding:
            try:
                self._pinecone_index.upsert(
                    vectors=[{
                        "id": tool.id,
                        "values": embedding,
                        "metadata": {
                            "name": tool.name,
                            "description": tool.description,
                            "tags": ",".join(tool.capability_tags),
                            "api_source_id": tool.api_source_id
                        }
                    }]
                )
            except Exception as e:
                logger.error(f"Failed to store embedding in Pinecone: {e}")

        # Store in file (simple implementation for Phase 2)
        os.makedirs("data/tools", exist_ok=True)
        with open(f"data/tools/{tool.id}.json", "w") as f:
            # Convert to dict manually to handle datetime serialization
            tool_dict = {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "function_id": tool.function_id,
                "api_source_id": tool.api_source_id,
                "capability_tags": tool.capability_tags,
                "metadata": tool.metadata,
                "usage_count": tool.usage_count,
                "created_at": tool.created_at.isoformat(),
                "updated_at": tool.updated_at.isoformat(),
                "last_used": tool.last_used.isoformat() if tool.last_used else None
            }
            # Exclude embedding from file storage as it's large
            json.dump(tool_dict, f, indent=2)

        # Also store the function definition
        self._save_function_definition(function_def)

        return tool

    async def search_by_capability(self, capability_description: str, 
                                top_k: int = 5) -> List[ToolRegistration]:
        """
        Search for tools by capability description.

        Args:
            capability_description: Description of the required capability
            top_k: Number of top results to return

        Returns:
            List of matching tools
        """
        # Generate embedding for the capability description
        embedding = await get_embedding(capability_description)

        # Search using Pinecone if available
        if self._pinecone_initialized and embedding:
            try:
                results = self._pinecone_index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True
                )

                tool_ids = [match['id'] for match in results['matches']]
                return [self._tools[tid] for tid in tool_ids if tid in self._tools]
            except Exception as e:
                logger.error(f"Pinecone search failed: {e}")

        # Fallback to simple tag-based search
        matches = []
        for tool in self._tools.values():
            # Calculate relevance score based on tag matches
            score = 0
            for tag in tool.capability_tags:
                if tag.lower() in capability_description.lower():
                    score += 1

            if score > 0:
                matches.append((tool, score))

        # Sort by relevance score and then by usage count
        matches.sort(key=lambda x: (x[1], x[0].usage_count), reverse=True)

        return [tool for tool, _ in matches[:top_k]]

    async def get_tool(self, tool_id: str) -> Optional[ToolRegistration]:
        """
        Retrieve a tool by ID.

        Args:
            tool_id: ID of the tool to retrieve

        Returns:
            Tool registration or None if not found
        """
        return self._tools.get(tool_id)

    async def increment_usage(self, tool_id: str) -> bool:
        """
        Increment the usage count for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            True if successful, False otherwise
        """
        tool = self._tools.get(tool_id)
        if not tool:
            return False

        # Update the tool
        tool.usage_count += 1
        tool.last_used = datetime.now()
        tool.updated_at = datetime.now()

        # Store the updated tool
        self._tools[tool_id] = tool

        # Update the file storage
        os.makedirs("data/tools", exist_ok=True)
        with open(f"data/tools/{tool.id}.json", "w") as f:
            # Convert to dict manually to handle datetime serialization
            tool_dict = {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "function_id": tool.function_id,
                "api_source_id": tool.api_source_id,
                "capability_tags": tool.capability_tags,
                "metadata": tool.metadata,
                "usage_count": tool.usage_count,
                "created_at": tool.created_at.isoformat(),
                "updated_at": tool.updated_at.isoformat(),
                "last_used": tool.last_used.isoformat() if tool.last_used else None
            }
            json.dump(tool_dict, f, indent=2)

        return True

    async def _generate_embedding(self, function_def: FunctionDefinition) -> Optional[List[float]]:
        """Generate embedding for a function definition."""
        try:
            # Combine relevant fields for embedding
            text = f"{function_def.name} {function_def.description} "
            text += f"{' '.join(function_def.tags)} "
            text += f"{function_def.endpoint} {function_def.method}"

            # Get embedding from OpenAI
            embedding = await get_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _save_function_definition(self, function_def: FunctionDefinition) -> None:
        """
        Save a function definition to file.

        Args:
            function_def: Function definition to save
        """
        try:
            # Create functions directory if it doesn't exist
            os.makedirs("data/functions", exist_ok=True)

            # Prepare the data for serialization
            func_data = {
                "id": function_def.id,
                "name": function_def.name,
                "description": function_def.description,
                "api_source_id": function_def.api_source_id,
                "api_type": function_def.api_type.value,
                "parameters": [
                    {
                        "name": param.name,
                        "param_type": param.param_type.value,
                        "description": param.description,
                        "required": param.required,
                        "default_value": param.default_value
                    }
                    for param in function_def.parameters
                ],
                "code": function_def.code,
                "endpoint": function_def.endpoint,
                "method": function_def.method,
                "response_format": function_def.response_format,
                "tags": function_def.tags,
                "created_at": str(function_def.created_at),
                "updated_at": str(function_def.updated_at)
            }

            # Save to file
            with open(f"data/functions/{function_def.id}.json", "w") as f:
                json.dump(func_data, f, indent=2)

            logger.info(f"Saved function definition {function_def.id} to file")
        except Exception as e:
            logger.error(f"Failed to save function definition: {e}")

    async def get_tool_by_name(self, name: str) -> Optional[ToolRegistration]:
        """
        Get a tool by name.

        Args:
            name: Name of the tool

        Returns:
            Tool registration or None if not found
        """
        for tool in self._tools.values():
            if tool.name == name:
                return tool

        return None

    async def get_tools_for_responses_api(self, capability_description: str = None) -> List[Dict]:
        """
        Get tools in Responses API format.

        Args:
            capability_description: Optional capability description to filter tools

        Returns:
            List of tools in Responses API format
        """
        # Import here to avoid circular imports
        from atia.function_builder.builder import FunctionBuilder

        if capability_description:
            # Search for tools matching the capability
            tools = await self.search_by_capability(capability_description)
        else:
            # Get all tools
            tools = list(self._tools.values())

        # Convert to Responses API format
        responses_api_tools = []
        function_builder = FunctionBuilder()

        for tool in tools:
            # Get the function definition
            function_def = await self._get_function_definition(tool.function_id)

            # Convert to Responses API format
            if function_def:
                tool_schema = function_builder.generate_responses_api_tool_schema(function_def)
                responses_api_tools.append(tool_schema)

        return responses_api_tools

    async def _get_function_definition(self, function_id: str) -> Optional[FunctionDefinition]:
        """
        Get a function definition by ID.

        Args:
            function_id: ID of the function

        Returns:
            Function definition or None if not found
        """
        # In a real implementation, this would retrieve from a database
        # For Phase 3, we'll implement a basic file-based lookup

        # Import here to avoid circular imports
        from atia.function_builder.models import FunctionDefinition, ApiType, ParameterType, FunctionParameter
        import os
        import json

        # Check if a file with this function_id exists in the functions directory
        tools_dir = "data/functions"
        os.makedirs(tools_dir, exist_ok=True)
        function_file = f"{tools_dir}/{function_id}.json"

        if os.path.exists(function_file):
            try:
                with open(function_file, "r") as f:
                    func_data = json.load(f)

                # Convert parameters from dict to FunctionParameter objects
                parameters = []
                for param_data in func_data.get("parameters", []):
                    parameters.append(FunctionParameter(
                        name=param_data.get("name", ""),
                        param_type=ParameterType(param_data.get("param_type", "string")),
                        description=param_data.get("description", ""),
                        required=param_data.get("required", False),
                        default_value=param_data.get("default_value")
                    ))

                return FunctionDefinition(
                    id=func_data.get("id", function_id),
                    name=func_data.get("name", ""),
                    description=func_data.get("description", ""),
                    api_source_id=func_data.get("api_source_id", ""),
                    api_type=ApiType(func_data.get("api_type", "rest")),
                    parameters=parameters,
                    code=func_data.get("code", ""),
                    endpoint=func_data.get("endpoint", ""),
                    method=func_data.get("method", "GET"),
                    tags=func_data.get("tags", [])
                )
            except Exception as e:
                logger.error(f"Error loading function definition: {e}")

        # If no file exists or there was an error, return a mock function definition for development
        logger.warning(f"No function definition found for {function_id}, using mock implementation")
        return FunctionDefinition(
            id=function_id,
            name=f"function_{function_id[:8]}",
            description=f"Mock function for {function_id}",
            api_source_id="mock_api",
            api_type=ApiType.REST,
            parameters=[
                FunctionParameter(
                    name="param1",
                    param_type=ParameterType.STRING,
                    description="A test parameter",
                    required=True
                )
            ],
            code=f"""
async def function_{function_id[:8]}(param1: str):
    \"\"\"
    Mock function for {function_id}

    Args:
        param1: A test parameter

    Returns:
        A mock response
    \"\"\"
    return {{"result": param1, "function_id": "{function_id}"}}
""",
            endpoint="/test",
            method="GET"
        )