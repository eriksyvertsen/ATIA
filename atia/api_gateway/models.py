"""
Data models for the API Gateway component.

This module defines models for API requests, responses, authentication, and validation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, EmailStr, validator, HttpUrl


class UserRole(str, Enum):
    """User roles for API Gateway access control."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    API = "api"  # For service-to-service API access


class APIKey(BaseModel):
    """API key for service authentication."""
    id: str = Field(...)
    name: str
    key: str
    user_id: Optional[str] = None
    roles: List[UserRole] = [UserRole.API]
    scopes: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked: bool = False


class UserAuth(BaseModel):
    """User authentication details."""
    id: str = Field(...)
    username: str
    email: EmailStr
    password_hash: str
    roles: List[UserRole] = [UserRole.USER]
    scopes: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True


class Token(BaseModel):
    """JWT token model."""
    access_token: str
    token_type: str
    expires_at: datetime
    user_id: str
    username: str
    roles: List[UserRole]
    scopes: List[str]


class TokenData(BaseModel):
    """JWT token payload data."""
    sub: str  # user ID
    username: str
    roles: List[UserRole]
    scopes: List[str]
    exp: datetime


class UserCreate(BaseModel):
    """Request model for creating a new user."""
    username: str
    email: EmailStr
    password: str
    roles: Optional[List[UserRole]] = None

    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserResponse(BaseModel):
    """Response model for user data."""
    id: str
    username: str
    email: str
    roles: List[UserRole]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool


class APIKeyCreate(BaseModel):
    """Request model for creating a new API key."""
    name: str
    expires_in_days: Optional[int] = None
    scopes: Optional[List[str]] = None
    roles: Optional[List[UserRole]] = None


class APIKeyResponse(BaseModel):
    """Response model for API key data."""
    id: str
    name: str
    key: str  # Only returned when the key is first created
    created_at: datetime
    expires_at: Optional[datetime]
    scopes: List[str]
    roles: List[UserRole]


class QueryRequest(BaseModel):
    """Query request model for the ATIA system."""
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    use_responses_api: bool = True
    trace: bool = False


class ToolExecuteRequest(BaseModel):
    """Tool execution request model."""
    tool_id: str
    parameters: Dict[str, Any]
    trace: bool = False


class APIDiscoveryRequest(BaseModel):
    """API discovery request model."""
    capability: str
    evaluate: bool = True
    max_results: int = 5


class ToolFeedbackRequest(BaseModel):
    """Tool feedback request model."""
    tool_id: str
    rating: str  # excellent, good, neutral, poor, very_poor
    categories: Optional[List[str]] = None
    comment: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error_occurred: bool = False
    error_detail: Optional[str] = None
    session_id: Optional[str] = None


class APIFeedbackRequest(BaseModel):
    """API feedback request model."""
    api_source_id: str
    rating: str  # excellent, good, neutral, poor, very_poor
    categories: Optional[List[str]] = None
    comment: Optional[str] = None
    response_time_ms: Optional[float] = None
    http_status: Optional[int] = None
    error_occurred: bool = False
    error_detail: Optional[str] = None
    session_id: Optional[str] = None


class SystemHealth(BaseModel):
    """System health status model."""
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]
    resource_utilization: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)


class LogLevel(str, Enum):
    """Log levels for the logging endpoint."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEntry(BaseModel):
    """Log entry model."""
    level: LogLevel
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    component: Optional[str] = None
    context: Optional[Dict[str, Any]] = None