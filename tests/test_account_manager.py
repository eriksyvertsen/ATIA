"""
Tests for the Account Manager component.
"""

import pytest
from datetime import datetime

from atia.account_manager import AccountManager, AuthType
from atia.account_manager.models import APICredential


@pytest.fixture
def account_manager():
    """
    Create an AccountManager instance for testing.
    """
    return AccountManager()


@pytest.mark.asyncio
async def test_register_api(account_manager):
    """Test registering a new API."""
    api_info = {
        "id": "test_api",
        "name": "Test API",
        "auth_type": "api_key"
    }

    credential = await account_manager.register_api(api_info)

    assert credential is not None
    assert credential.api_source_id == "test_api"
    assert credential.api_name == "Test API"
    assert credential.auth_type == AuthType.API_KEY


@pytest.mark.asyncio
async def test_store_and_get_credential(account_manager):
    """Test storing and retrieving a credential."""
    credential = APICredential(
        api_source_id="test_api",
        api_name="Test API",
        auth_type=AuthType.API_KEY,
        credential_data={"api_key": "test_key"}
    )

    await account_manager.store_credential(credential)
    retrieved = await account_manager.get_credential(credential.id)

    assert retrieved is not None
    assert retrieved.id == credential.id
    assert retrieved.api_source_id == "test_api"
    assert retrieved.api_name == "Test API"
    assert retrieved.auth_type == AuthType.API_KEY
    assert retrieved.credential_data["api_key"] == "test_key"


@pytest.mark.asyncio
async def test_get_nonexistent_credential(account_manager):
    """Test retrieving a nonexistent credential."""
    credential = await account_manager.get_credential("nonexistent")
    assert credential is None


@pytest.mark.asyncio
async def test_account_manager_refresh_credential(account_manager):
    """Test refreshing an OAuth credential."""
    # Create and store a test OAuth credential
    api_info = {
        "id": "test_oauth_api",
        "name": "Test OAuth API",
        "auth_type": "oauth"
    }

    # Register the API to create a credential
    credential = await account_manager.register_api(api_info)
    credential_id = credential.id

    # Manually update the auth_type to OAuth for this test
    credential.auth_type = AuthType.OAUTH
    await account_manager.store_credential(credential)

    # Refresh the credential
    refreshed = await account_manager.refresh_credential(credential_id)

    # Check the refreshed credential
    assert refreshed is not None
    assert refreshed.id == credential_id
    assert refreshed.auth_type == AuthType.OAUTH

    # The last_used timestamp should be updated
    assert refreshed.last_used > credential.last_used