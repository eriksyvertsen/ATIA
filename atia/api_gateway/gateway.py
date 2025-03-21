"""
API Gateway for the ATIA system.

This module provides the main FastAPI application that serves as the API Gateway
for the ATIA system, handling authentication, routing, and request validation.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import psutil
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from atia.config import settings
from atia.api_gateway.models import (
    UserAuth, 
    APIKey, 
    UserRole, 
    Token, 
    UserCreate, 
    UserResponse,
    APIKeyCreate, 
    APIKeyResponse,
    SystemHealth,
    LogLevel,
    LogEntry
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
from atia.utils.error_handling import catch_and_log


# Set up logging
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ATIA API Gateway",
    description="API Gateway for the Autonomous Tool Integration Agent",
    version="1.0.0",
    docs_url=None,  # Disable default docs (we'll add custom auth)
    redoc_url=None  # Disable default redoc (we'll add custom auth)
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gateway start time (for uptime calculation)
START_TIME = datetime.now()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.

    This implements a simple token bucket algorithm for rate limiting.
    In a real implementation, this would integrate with the RateLimiterRegistry
    from the performance module.
    """

    def __init__(self, app, 
                requests_per_second: float = 10.0,
                burst_size: int = 50):
        """
        Initialize the rate limit middleware.

        Args:
            app: FastAPI application
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size
        """
        super().__init__(app)
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.client_limits: Dict[str, Dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next):
        """
        Handle a request, applying rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware/route handler

        Returns:
            The response
        """
        # Extract client ID (IP address or API key)
        client_id = self._get_client_id(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )

        # Process the request
        return await call_next(request)

    def _get_client_id(self, request: Request) -> str:
        """Get client ID from request."""
        # Try to get from API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"

        # Fall back to client IP
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()

        # Initialize client state if needed
        if client_id not in self.client_limits:
            self.client_limits[client_id] = {
                "tokens": self.burst_size,
                "last_update": now
            }

        # Get client state
        client_state = self.client_limits[client_id]

        # Calculate time elapsed since last update
        elapsed = now - client_state["last_update"]

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.requests_per_second
        client_state["tokens"] = min(self.burst_size, client_state["tokens"] + new_tokens)
        client_state["last_update"] = now

        # Check if client has enough tokens
        if client_state["tokens"] >= 1.0:
            client_state["tokens"] -= 1.0
            return True
        else:
            return False


# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    requests_per_second=10.0,  # Default: 10 req/s
    burst_size=50  # Allow bursts of up to 50 requests
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(self, request: Request, call_next):
        """
        Handle a request, logging details.

        Args:
            request: The incoming request
            call_next: The next middleware/route handler

        Returns:
            The response
        """
        # Start timer
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process the request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(f"Response: {response.status_code} - {duration_ms:.2f}ms")

            # Add timing header
            response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

            return response
        except Exception as e:
            # Log error
            logger.error(f"Error processing request: {e}")

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Return error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(e)}
            )


# Add logging middleware
app.add_middleware(LoggingMiddleware)


# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(username: str, password: str):
    """
    Authenticate and get an access token.

    Args:
        username: Username
        password: Password

    Returns:
        JWT token
    """
    user = auth_handler.authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Create token
    token = auth_handler.create_jwt_token(
        user_id=user.id,
        username=user.username,
        roles=user.roles,
        scopes=user.scopes
    )

    return token


# User management endpoints
@app.post("/users", response_model=UserResponse, dependencies=[Depends(require_admin)])
async def create_user(user_data: UserCreate):
    """
    Create a new user.

    Args:
        user_data: User creation data

    Returns:
        Created user
    """
    try:
        # Create user
        user = auth_handler.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            roles=user_data.roles
        )

        # Convert to response model
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            roles=user.roles,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/users/me", response_model=UserResponse)
async def get_current_user_info(user: UserAuth = Depends(get_current_user)):
    """
    Get current user info.

    Returns:
        User info
    """
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        roles=user.roles,
        created_at=user.created_at,
        last_login=user.last_login,
        is_active=user.is_active
    )


@app.get("/users", response_model=List[UserResponse], dependencies=[Depends(require_admin)])
async def get_all_users():
    """
    Get all users.

    Returns:
        List of users
    """
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            roles=user.roles,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active
        )
        for user in auth_handler.users.values()
    ]


# API key management endpoints
@app.post("/api-keys", response_model=APIKeyResponse, dependencies=[Depends(require_user)])
async def create_api_key(key_data: APIKeyCreate, user: UserAuth = Depends(get_current_user)):
    """
    Create a new API key.

    Args:
        key_data: API key creation data

    Returns:
        Created API key
    """
    # Create API key
    api_key = auth_handler.create_api_key(
        name=key_data.name,
        user_id=user.id,
        roles=key_data.roles,
        scopes=key_data.scopes,
        expires_in_days=key_data.expires_in_days
    )

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=api_key.key,  # Only returned when the key is first created
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        scopes=api_key.scopes,
        roles=api_key.roles
    )


@app.get("/api-keys", response_model=List[APIKeyResponse], dependencies=[Depends(require_user)])
async def get_user_api_keys(user: UserAuth = Depends(get_current_user)):
    """
    Get API keys for the current user.

    Returns:
        List of API keys
    """
    # Get all keys for this user
    user_keys = [
        APIKeyResponse(
            id=key.id,
            name=key.name,
            key="********",  # Mask the actual key
            created_at=key.created_at,
            expires_at=key.expires_at,
            scopes=key.scopes,
            roles=key.roles
        )
        for key in auth_handler.api_keys.values()
        if key.user_id == user.id
    ]

    return user_keys


@app.delete("/api-keys/{key_id}", response_model=Dict[str, str], dependencies=[Depends(require_user)])
async def revoke_api_key(key_id: str, user: UserAuth = Depends(get_current_user)):
    """
    Revoke an API key.

    Args:
        key_id: API key ID

    Returns:
        Status message
    """
    # Check if key exists and belongs to user
    if key_id not in auth_handler.api_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    key = auth_handler.api_keys[key_id]

    # Only admin or key owner can revoke
    if key.user_id != user.id and UserRole.ADMIN not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to revoke this API key"
        )

    # Revoke the key
    auth_handler.revoke_api_key(key_id)

    return {"message": "API key revoked successfully"}


# System health endpoints
@app.get("/health", response_model=SystemHealth)
@catch_and_log(component="api_gateway")
async def get_system_health():
    """
    Get system health status.

    Returns:
        System health status
    """
    # Check component statuses
    components = {
        "api_gateway": {"status": "healthy", "uptime_seconds": (datetime.now() - START_TIME).total_seconds()},
        "database": {"status": "healthy", "type": "file-based"}
    }

    # Get resource utilization
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    resource_utilization = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_mb": memory.used / (1024 * 1024),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024 * 1024 * 1024)
    }

    # Determine overall status
    status = "healthy"
    if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
        status = "degraded"

    # Create health status
    health = SystemHealth(
        status=status,
        uptime_seconds=(datetime.now() - START_TIME).total_seconds(),
        components=components,
        resource_utilization=resource_utilization
    )

    return health


# Logs endpoint
@app.post("/logs", response_model=Dict[str, str], dependencies=[Depends(require_user)])
async def add_log_entry(log_entry: LogEntry):
    """
    Add a log entry.

    Args:
        log_entry: Log entry to add

    Returns:
        Status message
    """
    # Map log level
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL
    }

    # Create log message
    log_msg = log_entry.message
    if log_entry.component:
        log_msg = f"[{log_entry.component}] {log_msg}"

    # Add log entry
    logger.log(level_map[log_entry.level], log_msg, extra=log_entry.context)

    return {"message": "Log entry added successfully"}


# OpenAPI/Swagger UI endpoints
@app.get("/docs", include_in_schema=False)
async def get_documentation(user: Union[UserAuth, APIKey] = Depends(get_user_or_api_key)):
    """
    Get Swagger UI documentation.

    Returns:
        Swagger UI HTML
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="ATIA API Gateway",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
    )


@app.get("/redoc", include_in_schema=False)
async def get_redoc(user: Union[UserAuth, APIKey] = Depends(get_user_or_api_key)):
    """
    Get ReDoc documentation.

    Returns:
        ReDoc HTML
    """
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="ATIA API Gateway"
    )


# Include additional API routers
# Note: These would be implemented in separate modules
# We're just defining the structure here

# ATIA Core API endpoints
from fastapi import APIRouter
core_router = APIRouter(prefix="/api/core", tags=["core"])

@core_router.post("/query")
async def process_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    user: Union[UserAuth, APIKey] = Depends(get_user_or_api_key)
):
    """
    Process a query.

    Args:
        query: The query to process
        context: Optional context

    Returns:
        Query result
    """
    # This would call the AgentCore.process_query method
    return {"result": f"Processed query: {query}", "user": user.id}


@core_router.post("/tools/{tool_id}/execute")
async def execute_tool(
    tool_id: str,
    parameters: Dict[str, Any],
    user: Union[UserAuth, APIKey] = Depends(get_user_or_api_key)
):
    """
    Execute a tool.

    Args:
        tool_id: Tool ID
        parameters: Tool parameters

    Returns:
        Tool execution result
    """
    # This would call the ToolExecutor.execute_tool method
    return {"result": f"Executed tool {tool_id} with parameters {parameters}"}


# Add core router
app.include_router(core_router)


# Feedback API endpoints
try:
    from atia.feedback.api import router as feedback_router
    app.include_router(feedback_router)
    logger.info("Feedback API endpoints registered")
except ImportError:
    logger.warning("Feedback module not available, feedback endpoints not registered")


# Performance API endpoints
try:
    from atia.performance.api import router as performance_router
    app.include_router(performance_router)
    logger.info("Performance API endpoints registered")
except ImportError:
    logger.warning("Performance module not available, performance endpoints not registered")


@app.on_event("startup")
async def startup_event():
    """Run when the API gateway starts."""
    logger.info("API Gateway starting up")

    # Create data directories if they don't exist
    os.makedirs("data/auth", exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Run when the API gateway shuts down."""
    logger.info("API Gateway shutting down")


def run_gateway():
    """Run the API gateway with Uvicorn."""
    import uvicorn
    uvicorn.run(
        "atia.api_gateway.gateway:app",
        host=settings.api_gateway_host,
        port=settings.api_gateway_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    run_gateway()