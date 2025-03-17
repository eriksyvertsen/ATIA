"""
Account Manager implementation for handling API credentials.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from atia.account_manager.models import APICredential, AuthType
from atia.config import settings


logger = logging.getLogger(__name__)


class AccountManager:
    """
    Handles the account registration process and securely manages API credentials.
    """

    def __init__(self):
        """
        Initialize the Account Manager.
        """
        self._credentials = {}  # In-memory storage for credentials (for Phase 1)
        self._auth_handlers = {
            AuthType.API_KEY: self._handle_api_key_auth,
            AuthType.OAUTH: self._handle_oauth_auth,
            AuthType.BASIC: self._handle_basic_auth,
            AuthType.JWT: self._handle_jwt_auth,
            AuthType.BEARER: self._handle_bearer_auth,
        }

    async def register_api(self, api_info: Dict[str, Any]) -> APICredential:
        """
        Register a new API and create initial credentials.

        Args:
            api_info: Information about the API to register

        Returns:
            The created API credential
        """
        # Determine auth type
        auth_type_str = api_info.get("auth_type", "api_key").lower()
        auth_type_map = {
            "api_key": AuthType.API_KEY,
            "oauth": AuthType.OAUTH,
            "basic": AuthType.BASIC,
            "jwt": AuthType.JWT,
            "bearer": AuthType.BEARER,
            "none": AuthType.NONE
        }
        auth_type = auth_type_map.get(auth_type_str, AuthType.API_KEY)

        # Create credential
        credential = APICredential(
            api_source_id=api_info.get("id", str(uuid.uuid4())),
            api_name=api_info.get("name", "Unknown API"),
            auth_type=auth_type,
            credential_data={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_used=datetime.now()
        )

        # Store credential
        await self.store_credential(credential)

        return credential

    async def store_credential(self, credential: APICredential) -> None:
        """
        Store a credential securely.

        Args:
            credential: The credential to store
        """
        # Update the updated_at timestamp
        credential.updated_at = datetime.now()

        # In Phase 1, we'll just store in memory
        # In later phases, this would use AWS Secrets Manager or similar
        self._credentials[credential.id] = credential

    async def get_credential(self, credential_id: str) -> Optional[APICredential]:
        """
        Retrieve a credential by ID.

        Args:
            credential_id: The ID of the credential to retrieve

        Returns:
            The credential, or None if not found
        """
        return self._credentials.get(credential_id)

    async def refresh_credential(self, credential_id: str) -> Optional[APICredential]:
        """
        Refresh a credential, e.g., get a new OAuth token using a refresh token.

        Args:
            credential_id: The ID of the credential to refresh

        Returns:
            The refreshed credential, or None if not found
        """
        credential = await self.get_credential(credential_id)
        if not credential:
            return None

        if credential.auth_type == AuthType.OAUTH:
            # In a real implementation, this would call the OAuth refresh endpoint
            # For Phase 1, we'll just update the last_used timestamp
            logger.info(f"Refreshing OAuth credential {credential_id}")

            # Create a new instance with updated last_used timestamp to ensure it's different
            refreshed = APICredential(
                id=credential.id,
                api_source_id=credential.api_source_id,
                api_name=credential.api_name,
                auth_type=credential.auth_type,
                credential_data=credential.credential_data,
                expiration_date=credential.expiration_date,
                created_at=credential.created_at,
                rotation_policy=credential.rotation_policy,
                # Set new timestamps for last_used and updated_at with a small delay to ensure they're different
                last_used=datetime.now(),
                updated_at=datetime.now()
            )

            # Store the refreshed credential
            await self.store_credential(refreshed)
            return refreshed

        # For other auth types, just return the existing credential
        return credential

    async def _handle_api_key_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle API key authentication.

        Args:
            api_info: Information about the API

        Returns:
            Credential data for API key auth
        """
        # For Phase 1, this is a placeholder
        return {"api_key": "dummy_api_key"}

    async def _handle_oauth_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle OAuth authentication.

        Args:
            api_info: Information about the API

        Returns:
            Credential data for OAuth auth
        """
        # For Phase 1, this is a placeholder
        return {
            "access_token": "dummy_access_token",
            "refresh_token": "dummy_refresh_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }

    async def _handle_basic_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Basic authentication.

        Args:
            api_info: Information about the API

        Returns:
            Credential data for Basic auth
        """
        # For Phase 1, this is a placeholder
        return {
            "username": "dummy_username",
            "password": "dummy_password"
        }

    async def _handle_jwt_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle JWT authentication.

        Args:
            api_info: Information about the API

        Returns:
            Credential data for JWT auth
        """
        # For Phase 1, this is a placeholder
        return {"token": "dummy_jwt_token"}

    async def _handle_bearer_auth(self, api_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Bearer authentication.

        Args:
            api_info: Information about the API

        Returns:
            Credential data for Bearer auth
        """
        # For Phase 1, this is a placeholder
        return {"token": "dummy_bearer_token"}