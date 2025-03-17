"""
Data models for the Tool Registry component.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from pydantic import BaseModel, Field


class ToolRegistration(BaseModel):
    """Represents a registered tool in the registry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    function_id: str
    api_source_id: str
    capability_tags: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None