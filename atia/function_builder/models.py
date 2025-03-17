"""
Data models for the Function Builder component.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid

from pydantic import BaseModel, Field


class ApiType(Enum):
    """Types of APIs supported."""
    REST = "rest"
    GRAPHQL = "graphql"
    SOAP = "soap"
    GENERIC = "generic"


class ParameterType(Enum):
    """Types of function parameters."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class FunctionParameter(BaseModel):
    """Represents a parameter for a function."""
    name: str
    param_type: ParameterType
    description: str
    required: bool = False
    default_value: Optional[Any] = None


class FunctionDefinition(BaseModel):
    """Represents a generated function definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    api_source_id: str
    api_type: ApiType = ApiType.REST
    parameters: List[FunctionParameter] = []
    code: str
    endpoint: str
    method: str
    response_format: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: str(datetime.now().isoformat()))
    updated_at: str = Field(default_factory=lambda: str(datetime.now().isoformat()))
    tags: List[str] = []
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)