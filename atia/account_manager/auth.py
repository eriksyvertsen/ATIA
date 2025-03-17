"""
Authentication handler for various API authentication methods.

This module provides support for API key, OAuth, basic auth, and JWT authentication methods.
"""

import uuid
import enum
import json
import logging
import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

import httpx
from pydantic import BaseModel, Field, validator, HttpUrl

from atia.config import settings
from atia.utils.openai_client import get_completion
from atia.account_manager.security import SecurityManager


class AuthType(str, enum.Enum):
    """Types of authentication supported."""
    API_KEY = "api_key"
    OAUTH = "oauth"
    BASIC = "basic"
    JWT = "jwt"
    NONE = "none"  # For APIs that don't require authentication


class ApiKeyLocation(str, enum.Enum):
    """Possible locations for API key in requests."""
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"
    BODY = "body"


class OAuthGrantType(str, enum.Enum):
    """OAuth 2.0 grant types."""
    CLIENT_CREDENTIALS = "client_credentials"
    AUTHORIZATION_CODE = "authorization_code"
    PASSWORD = "password"
    REFRESH_TOKEN = "refresh_token"


class APIKeyConfig(BaseModel):
    """Configuration for API key authentication."""
    name: str  # The name of the parameter (e.g., "X-API-Key", "Authorization", etc.)
    location: ApiKeyLocation
    prefix: Optional[str] = None  # Optional prefix (e.g., "Bearer" for "Bearer YOUR_API_KEY")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "prefix": self.prefix
        }


class OAuthConfig(BaseModel):
    """Configuration for OAuth authentication."""
    token_url: HttpUrl
    client_id: str
    client_secret: str
    grant_type: OAuthGrantType
    scope: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_url": str(self.token_url),
            "client_id": self.client_id,
            "client_secret": "********",  # Don't expose the actual secret
            "grant_type": self.grant_type,
            "scope": self.scope
        }


class BasicAuthConfig(BaseModel):
    """Configuration for basic auth."""
    username: str
    password: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "password": "********"  # Don't expose the actual password
        }


class JWTConfig(BaseModel):
    """Configuration for JWT authentication."""
    token_url: HttpUrl
    client_id: str
    client_secret: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_url": str(self.token_url),
            "client_id": self.client_id,
            "client_secret": "********"  # Don't expose the actual secret
        }


class AuthConfig(BaseModel):
    """Complete authentication configuration."""
    auth_type: AuthType
    api_key_config: Optional[APIKeyConfig] = None
    oauth_config: Optional[OAuthConfig] = None
    basic_auth_config: Optional[BasicAuthConfig] = None
    jwt_config: Optional[JWTConfig] = None


class AuthCredentials(BaseModel):
    """Model for storing authentication credentials."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    api_source_id: str
    auth_type: AuthType
    encrypted_value: str
    metadata: Dict[str, Any] = {}
    expiration_date: Optional[datetime.datetime] = None
    last_used: datetime.datetime = Field(default_factory=datetime.datetime.now)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class HumanInterventionRequest(BaseModel):
    """Request for human intervention in authentication process."""
    flow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    auth_flow: Dict[str, Any]
    reason: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    status: str = "pending"


class AuthenticationHandler:
    """
    Handles authentication for various API types.

    This class can process different authentication methods and securely store credentials.
    """

    def __init__(self, security_manager: SecurityManager):
        """
        Initialize the authentication handler.

        Args:
            security_manager: The security manager for handling encryption/decryption
        """
        self.security_manager = security_manager
        self.logger = logging.getLogger(__name__)
        self.auth_methods = {
            AuthType.API_KEY: self.handle_api_key_auth,
            AuthType.OAUTH: self.handle_oauth,
            AuthType.BASIC: self.handle_basic_auth,
            AuthType.JWT: self.handle_jwt_auth,
            AuthType.NONE: self.handle_no_auth
        }

    async def authenticate(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process authentication for an API.

        Args:
            api_info: Information about the API including auth requirements

        Returns:
            Authentication result containing credentials and metadata
        """
        auth_type = api_info.get("auth_type", "unknown")
        try:
            auth_type_enum = AuthType(auth_type)
            if auth_type_enum in self.auth_methods:
                return await self.auth_methods[auth_type_enum](api_info)
            else:
                return await self.handle_unknown_auth(api_info)
        except ValueError:
            self.logger.warning(f"Unknown auth type: {auth_type}")
            return await self.handle_unknown_auth(api_info)

    async def handle_api_key_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle API key authentication.

        Args:
            api_info: API information including auth details

        Returns:
            Authentication result with API key
        """
        self.logger.info(f"Processing API key authentication for {api_info.get('name', 'unknown API')}")

        api_key = self._get_api_key_from_env(api_info)

        if not api_key:
            # Request API key from user
            self.logger.info("API key not found in environment, requesting human intervention")
            flow_id = await self.request_human_intervention(
                auth_flow={"type": "api_key", "api_name": api_info.get("name")},
                reason="API key needed for authentication"
            )
            # This would be where we'd wait for the human to provide the key
            # For Phase 2, we'll simulate this with a placeholder
            api_key = "simulate_user_provided_api_key"  # In production, we'd get this from user input

        # Determine the API key configuration
        api_key_config = self._determine_api_key_config(api_info)

        # Create the encrypted credential
        encrypted_key = self.security_manager.encrypt(api_key)

        return {
            "auth_type": AuthType.API_KEY,
            "api_source_id": api_info.get("id", str(uuid.uuid4())),
            "encrypted_value": encrypted_key,
            "metadata": {
                "config": api_key_config.to_dict()
            }
        }

    async def handle_oauth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle OAuth authentication.

        Args:
            api_info: API information including OAuth details

        Returns:
            Authentication result with OAuth tokens
        """
        self.logger.info(f"Processing OAuth authentication for {api_info.get('name', 'unknown API')}")

        # Determine the OAuth configuration
        oauth_config = self._determine_oauth_config(api_info)

        # For client credentials flow
        if oauth_config.grant_type == OAuthGrantType.CLIENT_CREDENTIALS:
            try:
                # Request token
                data = {
                    "grant_type": oauth_config.grant_type,
                    "client_id": oauth_config.client_id,
                    "client_secret": oauth_config.client_secret
                }

                if oauth_config.scope:
                    data["scope"] = oauth_config.scope

                async with httpx.AsyncClient() as client:
                    response = await client.post(str(oauth_config.token_url), data=data)
                    response.raise_for_status()
                    token_data = response.json()

                # Encrypt the token info
                encrypted_token = self.security_manager.encrypt(json.dumps(token_data))

                # Calculate expiration
                expiration = None
                if "expires_in" in token_data:
                    expires_in = token_data.get("expires_in")
                    if isinstance(expires_in, (int, float)):
                        expiration = datetime.datetime.now() + datetime.timedelta(seconds=expires_in)

                return {
                    "auth_type": AuthType.OAUTH,
                    "api_source_id": api_info.get("id", str(uuid.uuid4())),
                    "encrypted_value": encrypted_token,
                    "expiration_date": expiration,
                    "metadata": {
                        "config": oauth_config.to_dict(),
                        "token_type": token_data.get("token_type", "Bearer")
                    }
                }
            except Exception as e:
                self.logger.error(f"OAuth token request failed: {str(e)}")
                # Request human intervention
                flow_id = await self.request_human_intervention(
                    auth_flow={"type": "oauth", "api_name": api_info.get("name")},
                    reason=f"OAuth authentication failed: {str(e)}"
                )
                raise Exception(f"OAuth authentication failed: {str(e)}")
        else:
            # For more complex flows, request human intervention
            flow_id = await self.request_human_intervention(
                auth_flow={"type": "oauth", "api_name": api_info.get("name"), "grant_type": oauth_config.grant_type},
                reason=f"OAuth flow {oauth_config.grant_type} requires human interaction"
            )
            # Simulate user providing tokens for phase 2
            token_data = {
                "access_token": "simulated_access_token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "simulated_refresh_token"
            }
            encrypted_token = self.security_manager.encrypt(json.dumps(token_data))

            return {
                "auth_type": AuthType.OAUTH,
                "api_source_id": api_info.get("id", str(uuid.uuid4())),
                "encrypted_value": encrypted_token,
                "expiration_date": datetime.datetime.now() + datetime.timedelta(seconds=3600),
                "metadata": {
                    "config": oauth_config.to_dict(),
                    "token_type": "Bearer"
                }
            }

    async def handle_basic_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle basic authentication.

        Args:
            api_info: API information including basic auth details

        Returns:
            Authentication result with basic auth credentials
        """
        self.logger.info(f"Processing basic authentication for {api_info.get('name', 'unknown API')}")

        # For Phase 2, request human intervention for credentials
        flow_id = await self.request_human_intervention(
            auth_flow={"type": "basic", "api_name": api_info.get("name")},
            reason="Basic authentication requires username and password"
        )

        # Simulate user providing credentials for phase 2
        basic_auth_config = BasicAuthConfig(
            username="simulated_username",
            password="simulated_password"
        )

        # Encrypt the credentials
        credentials = {
            "username": basic_auth_config.username,
            "password": basic_auth_config.password
        }
        encrypted_creds = self.security_manager.encrypt(json.dumps(credentials))

        return {
            "auth_type": AuthType.BASIC,
            "api_source_id": api_info.get("id", str(uuid.uuid4())),
            "encrypted_value": encrypted_creds,
            "metadata": {
                "config": basic_auth_config.to_dict()
            }
        }

    async def handle_jwt_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle JWT authentication.

        Args:
            api_info: API information including JWT details

        Returns:
            Authentication result with JWT token
        """
        self.logger.info(f"Processing JWT authentication for {api_info.get('name', 'unknown API')}")

        # JWT is similar to OAuth in many ways
        jwt_config = self._determine_jwt_config(api_info)

        # For Phase 2, request human intervention
        flow_id = await self.request_human_intervention(
            auth_flow={"type": "jwt", "api_name": api_info.get("name")},
            reason="JWT authentication requires manual setup"
        )

        # Simulate token for phase 2
        token_data = {
            "token": "simulated_jwt_token",
            "expires_in": 3600
        }
        encrypted_token = self.security_manager.encrypt(json.dumps(token_data))

        return {
            "auth_type": AuthType.JWT,
            "api_source_id": api_info.get("id", str(uuid.uuid4())),
            "encrypted_value": encrypted_token,
            "expiration_date": datetime.datetime.now() + datetime.timedelta(seconds=3600),
            "metadata": {
                "config": jwt_config.to_dict()
            }
        }

    async def handle_no_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle APIs that don't require authentication.

        Args:
            api_info: API information

        Returns:
            Empty authentication result
        """
        self.logger.info(f"No authentication required for {api_info.get('name', 'unknown API')}")

        return {
            "auth_type": AuthType.NONE,
            "api_source_id": api_info.get("id", str(uuid.uuid4())),
            "encrypted_value": "",
            "metadata": {}
        }

    async def handle_unknown_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle unknown authentication types.

        Args:
            api_info: API information

        Returns:
            Authentication result based on determined auth type
        """
        self.logger.warning(f"Unknown auth type for {api_info.get('name', 'unknown API')}")

        # Try to determine the auth type using AI assistance
        auth_type = await self._determine_auth_type(api_info)

        # Update the API info and retry authentication
        api_info["auth_type"] = auth_type.value
        return await self.authenticate(api_info)

    async def request_human_intervention(self, auth_flow: Dict[str, Any], reason: str) -> str:
        """
        Request human intervention for authentication flows.

        Args:
            auth_flow: Authentication flow details
            reason: Reason for requiring human intervention

        Returns:
            Intervention request ID
        """
        intervention_request = HumanInterventionRequest(
            auth_flow=auth_flow,
            reason=reason
        )

        self.logger.info(f"Requesting human intervention: {intervention_request.flow_id} - {reason}")

        # In a full implementation, this would store the request in a database
        # and notify human operators

        # For Phase 2, we'll log it
        self.logger.warning(
            f"Human intervention required: {intervention_request.flow_id}\n"
            f"Reason: {reason}\n"
            f"Auth Flow: {json.dumps(auth_flow, indent=2)}"
        )

        return intervention_request.flow_id

    def _get_api_key_from_env(self, api_info: Dict[str, Any]) -> Optional[str]:
        """
        Try to find API key in environment variables.

        Args:
            api_info: API information

        Returns:
            API key if found, None otherwise
        """
        # Check common environment variable patterns for the API
        api_name = api_info.get("name", "")
        possible_env_vars = [
            f"{api_name.upper()}_API_KEY",
            f"{api_name.upper().replace(' ', '_')}_KEY",
            f"{api_name.upper().replace(' ', '_').replace('-', '_')}_API_KEY"
        ]

        for env_var in possible_env_vars:
            if hasattr(settings, env_var.lower()):
                value = getattr(settings, env_var.lower())
                if value:
                    return value

        return None

    def _determine_api_key_config(self, api_info: Dict[str, Any]) -> APIKeyConfig:
        """
        Determine API key configuration from API info.

        Args:
            api_info: API information

        Returns:
            APIKeyConfig object
        """
        # Extract configuration from API info, or use defaults
        key_name = api_info.get("key_name", "X-API-Key")
        key_location = api_info.get("key_location", "header")
        key_prefix = api_info.get("key_prefix")

        return APIKeyConfig(
            name=key_name,
            location=ApiKeyLocation(key_location),
            prefix=key_prefix
        )

    def _determine_oauth_config(self, api_info: Dict[str, Any]) -> OAuthConfig:
        """
        Determine OAuth configuration from API info.

        Args:
            api_info: API information

        Returns:
            OAuthConfig object
        """
        # In a real implementation, we would extract this from the API documentation
        # For Phase 2, we'll use placeholders
        oauth_info = api_info.get("oauth", {})

        return OAuthConfig(
            token_url=oauth_info.get("token_url", "https://example.com/oauth/token"),
            client_id=oauth_info.get("client_id", "client_id_placeholder"),
            client_secret=oauth_info.get("client_secret", "client_secret_placeholder"),
            grant_type=OAuthGrantType(oauth_info.get("grant_type", "client_credentials")),
            scope=oauth_info.get("scope")
        )

    def _determine_jwt_config(self, api_info: Dict[str, Any]) -> JWTConfig:
        """
        Determine JWT configuration from API info.

        Args:
            api_info: API information

        Returns:
            JWTConfig object
        """
        # In a real implementation, we would extract this from the API documentation
        # For Phase 2, we'll use placeholders
        jwt_info = api_info.get("jwt", {})

        return JWTConfig(
            token_url=jwt_info.get("token_url", "https://example.com/jwt/token"),
            client_id=jwt_info.get("client_id", "client_id_placeholder"),
            client_secret=jwt_info.get("client_secret", "client_secret_placeholder")
        )

    async def _determine_auth_type(self, api_info: Dict[str, Any]) -> AuthType:
        """
        Use AI to determine the authentication type from API info.

        Args:
            api_info: API information

        Returns:
            Determined AuthType
        """
        prompt = f"""
        Analyze the following API information and determine the most likely authentication method.
        Choose from: api_key, oauth, basic, jwt, or none (if no authentication is required).

        API Name: {api_info.get('name', 'Unknown')}
        API Description: {api_info.get('description', 'No description available')}
        API Documentation URL: {api_info.get('documentation_url', 'No URL available')}

        Return only a single word with the authentication method.
        """

        system_message = (
            "You are an expert at analyzing API documentation and determining authentication methods. "
            "Respond with a single authentication type from the available options."
        )

        try:
            auth_type_str = get_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,
                max_tokens=10,
                model=settings.openai_model
            ).strip().lower()

            # Try to match to an AuthType
            try:
                return AuthType(auth_type_str)
            except ValueError:
                # Default to API key if not recognized
                self.logger.warning(f"Unrecognized auth type '{auth_type_str}', defaulting to API key")
                return AuthType.API_KEY
        except Exception as e:
            self.logger.error(f"Error determining auth type: {str(e)}")
            return AuthType.API_KEY  # Default to API key