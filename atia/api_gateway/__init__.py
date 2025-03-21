"""
API Gateway component.

Provides a robust external interface for accessing ATIA functionality through
a RESTful API with authentication, rate limiting, and request validation.
"""

from atia.api_gateway.models import (
    UserRole,
    APIKey,
    Token,
    UserAuth,
    SystemHealth
)

from atia.api_gateway.auth import (
    auth_handler,
    get_current_user,
    get_current_api_key,
    get_user_or_api_key,
    require_admin,
    require_user,
    require_api
)

from atia.api_gateway.gateway import app, run_gateway

__all__ = [
    # Models
    "UserRole",
    "APIKey",
    "Token",
    "UserAuth",
    "SystemHealth",

    # Auth
    "auth_handler",
    "get_current_user",
    "get_current_api_key",
    "get_user_or_api_key",
    "require_admin",
    "require_user",
    "require_api",

    # Gateway
    "app",
    "run_gateway"
]