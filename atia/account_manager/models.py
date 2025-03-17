"""
Data models for the Account Manager component.
"""

from datetime import datetime
from enum import Enum, auto
from typing import Dict, Optional, Any
import uuid

from pydantic import BaseModel, Field


class AuthType(Enum):
    """Enumeration of supported authentication types."""
    API_KEY = auto()
    OAUTH = auto()
    BASIC = auto()
    JWT = auto()
    BEARER = auto()
    NONE = auto()


class APICredential(BaseModel):
    """
    Represents an API credential stored securely by the Account Manager.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    api_source_id: str
    api_name: str
    auth_type: AuthType = AuthType.API_KEY
    credential_data: Dict[str, Any] = Field(default_factory=dict)
    expiration_date: Optional[datetime] = None
    last_used: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    rotation_policy: Optional[Dict[str, Any]] = None