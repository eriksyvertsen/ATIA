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