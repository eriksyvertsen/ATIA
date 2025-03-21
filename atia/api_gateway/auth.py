"""
Authentication and authorization for the API Gateway.

This module handles JWT token generation, API key authentication, and permission checks.
"""

import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from passlib.context import CryptContext

from atia.config import settings
from atia.api_gateway.models import UserAuth, UserRole, APIKey, Token, TokenData


# Set up logging
logger = logging.getLogger(__name__)

# Authentication and authorization dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Storage paths
AUTH_DIR = "data/auth"
USERS_FILE = os.path.join(AUTH_DIR, "users.json")
API_KEYS_FILE = os.path.join(AUTH_DIR, "api_keys.json")


class AuthHandler:
    """
    Handles authentication and authorization for the API Gateway.
    """

    def __init__(self):
        """Initialize the auth handler."""
        # Create storage directory if it doesn't exist
        os.makedirs(AUTH_DIR, exist_ok=True)

        # Load users and API keys
        self.users: Dict[str, UserAuth] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self._load_users()
        self._load_api_keys()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches hash, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """
        Generate a hash for a password.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    def create_jwt_token(self, 
                        user_id: str, 
                        username: str,
                        roles: List[UserRole],
                        scopes: List[str],
                        expires_delta: Optional[timedelta] = None) -> Token:
        """
        Create a JWT token for a user.

        Args:
            user_id: User ID
            username: Username
            roles: User roles
            scopes: Token scopes
            expires_delta: Optional expiration delta (default: 1 hour)

        Returns:
            JWT token
        """
        # Default expiration time: 1 hour
        if expires_delta is None:
            expires_delta = timedelta(seconds=settings.jwt_expiration_seconds)

        # Token expiration timestamp
        expires_at = datetime.now() + expires_delta

        # Token payload
        payload = {
            "sub": user_id,
            "username": username,
            "roles": [role.value for role in roles],
            "scopes": scopes,
            "exp": expires_at.timestamp()
        }

        # Encode token
        access_token = jwt.encode(
            payload,
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_at=expires_at,
            user_id=user_id,
            username=username,
            roles=roles,
            scopes=scopes
        )

    def verify_jwt_token(self, token: str) -> Optional[TokenData]:
        """
        Verify a JWT token.

        Args:
            token: JWT token

        Returns:
            TokenData if valid, None if invalid
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )

            # Extract token data
            user_id = payload.get("sub")
            username = payload.get("username")
            roles_str = payload.get("roles", [])
            scopes = payload.get("scopes", [])
            exp = payload.get("exp")

            # Validate required fields
            if not user_id or not username or not exp:
                logger.warning(f"JWT token missing required fields: {payload}")
                return None

            # Convert roles to UserRole enum
            roles = []
            for role_str in roles_str:
                try:
                    roles.append(UserRole(role_str))
                except ValueError:
                    logger.warning(f"Invalid role in JWT token: {role_str}")

            # Create token data
            token_data = TokenData(
                sub=user_id,
                username=username,
                roles=roles,
                scopes=scopes,
                exp=datetime.fromtimestamp(exp)
            )

            return token_data
        except jwt.PyJWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None

    def generate_api_key(self) -> str:
        """
        Generate a new API key.

        Returns:
            New API key
        """
        return secrets.token_urlsafe(32)

    def create_user(self, 
                  username: str, 
                  email: str, 
                  password: str,
                  roles: Optional[List[UserRole]] = None) -> UserAuth:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            password: Plain text password
            roles: User roles (default: [UserRole.USER])

        Returns:
            Created user
        """
        # Check if username already exists
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"Username {username} already exists")

        # Hash password
        password_hash = self.get_password_hash(password)

        # Default roles
        if roles is None:
            roles = [UserRole.USER]

        # Create user
        user = UserAuth(
            id=f"usr_{int(time.time())}_{len(self.users) + 1}",
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            created_at=datetime.now()
        )

        # Store user
        self.users[user.id] = user
        self._save_users()

        return user

    def get_user(self, user_id: str) -> Optional[UserAuth]:
        """
        Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User or None if not found
        """
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[UserAuth]:
        """
        Get a user by username.

        Args:
            username: Username

        Returns:
            User or None if not found
        """
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def authenticate_user(self, 
                        username: str, 
                        password: str) -> Optional[UserAuth]:
        """
        Authenticate a user with username and password.

        Args:
            username: Username
            password: Plain text password

        Returns:
            User if authentication succeeds, None otherwise
        """
        user = self.get_user_by_username(username)
        if not user:
            return None

        # Check password
        if not self.verify_password(password, user.password_hash):
            return None

        # Check if user is active
        if not user.is_active:
            return None

        # Update last login
        user.last_login = datetime.now()
        self._save_users()

        return user

    def create_api_key(self, 
                     name: str,
                     user_id: Optional[str] = None,
                     roles: Optional[List[UserRole]] = None,
                     scopes: Optional[List[str]] = None,
                     expires_in_days: Optional[int] = None) -> APIKey:
        """
        Create a new API key.

        Args:
            name: API key name
            user_id: Optional user ID
            roles: API key roles (default: [UserRole.API])
            scopes: API key scopes (default: [])
            expires_in_days: Optional expiration in days

        Returns:
            Created API key
        """
        # Generate key
        key = self.generate_api_key()

        # Default roles
        if roles is None:
            roles = [UserRole.API]

        # Default scopes
        if scopes is None:
            scopes = []

        # Calculate expiration
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Create API key
        api_key = APIKey(
            id=f"key_{int(time.time())}_{len(self.api_keys) + 1}",
            name=name,
            key=key,
            user_id=user_id,
            roles=roles,
            scopes=scopes,
            created_at=datetime.now(),
            expires_at=expires_at
        )

        # Store API key
        self.api_keys[api_key.id] = api_key
        self._save_api_keys()

        return api_key

    def get_api_key(self, key: str) -> Optional[APIKey]:
        """
        Get an API key by its key value.

        Args:
            key: API key value

        Returns:
            API key or None if not found
        """
        for api_key in self.api_keys.values():
            if api_key.key == key:
                return api_key
        return None

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: API key ID

        Returns:
            True if successful, False if key not found
        """
        if key_id not in self.api_keys:
            return False

        # Revoke key
        self.api_keys[key_id].revoked = True
        self.api_keys[key_id].expires_at = datetime.now()
        self._save_api_keys()

        return True

    def has_permission(self, 
                     roles: List[UserRole], 
                     required_roles: List[UserRole]) -> bool:
        """
        Check if user has the required roles.

        Args:
            roles: User roles
            required_roles: Required roles for the operation

        Returns:
            True if user has required roles, False otherwise
        """
        # Admin role has all permissions
        if UserRole.ADMIN in roles:
            return True

        # Check if user has any of the required roles
        return any(role in required_roles for role in roles)

    def _load_users(self) -> None:
        """Load users from file."""
        if not os.path.exists(USERS_FILE):
            # Create a default admin user if no users exist
            self._create_default_admin()
            return

        try:
            with open(USERS_FILE, 'r') as f:
                users_data = json.load(f)

            for user_data in users_data:
                # Convert timestamp strings to datetime
                for key in ['created_at', 'last_login']:
                    if key in user_data and user_data[key] is not None:
                        user_data[key] = datetime.fromisoformat(user_data[key])

                # Convert role strings to UserRole enum
                if 'roles' in user_data:
                    roles = []
                    for role_str in user_data['roles']:
                        try:
                            roles.append(UserRole(role_str))
                        except ValueError:
                            logger.warning(f"Invalid role: {role_str}")
                    user_data['roles'] = roles

                # Create UserAuth object
                user = UserAuth(**user_data)
                self.users[user.id] = user

            logger.info(f"Loaded {len(self.users)} users")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            # Create a default admin user if loading fails
            self._create_default_admin()

    def _save_users(self) -> None:
        """Save users to file."""
        try:
            users_data = []

            for user in self.users.values():
                # Convert to dict
                user_dict = user.model_dump()

                # Convert datetime to strings
                for key in ['created_at', 'last_login']:
                    if key in user_dict and user_dict[key] is not None:
                        user_dict[key] = user_dict[key].isoformat()

                # Convert UserRole enum to strings
                user_dict['roles'] = [role.value for role in user.roles]

                users_data.append(user_dict)

            with open(USERS_FILE, 'w') as f:
                json.dump(users_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    def _load_api_keys(self) -> None:
        """Load API keys from file."""
        if not os.path.exists(API_KEYS_FILE):
            return

        try:
            with open(API_KEYS_FILE, 'r') as f:
                keys_data = json.load(f)

            for key_data in keys_data:
                # Convert timestamp strings to datetime
                for key in ['created_at', 'expires_at', 'last_used_at']:
                    if key in key_data and key_data[key] is not None:
                        key_data[key] = datetime.fromisoformat(key_data[key])

                # Convert role strings to UserRole enum
                if 'roles' in key_data:
                    roles = []
                    for role_str in key_data['roles']:
                        try:
                            roles.append(UserRole(role_str))
                        except ValueError:
                            logger.warning(f"Invalid role: {role_str}")
                    key_data['roles'] = roles

                # Create APIKey object
                api_key = APIKey(**key_data)
                self.api_keys[api_key.id] = api_key

            logger.info(f"Loaded {len(self.api_keys)} API keys")
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")

    def _save_api_keys(self) -> None:
        """Save API keys to file."""
        try:
            keys_data = []

            for api_key in self.api_keys.values():
                # Convert to dict
                key_dict = api_key.model_dump()

                # Convert datetime to strings
                for key in ['created_at', 'expires_at', 'last_used_at']:
                    if key in key_dict and key_dict[key] is not None:
                        key_dict[key] = key_dict[key].isoformat()

                # Convert UserRole enum to strings
                key_dict['roles'] = [role.value for role in api_key.roles]

                keys_data.append(key_dict)

            with open(API_KEYS_FILE, 'w') as f:
                json.dump(keys_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")

    def _create_default_admin(self) -> None:
        """Create a default admin user."""
        # Check if admin already exists
        admin_user = self.get_user_by_username("admin")
        if admin_user:
            return

        # Default admin password (should be changed after first login)
        default_password = os.environ.get("ATIA_ADMIN_PASSWORD", "atia_admin")

        # Create admin user
        admin_user = UserAuth(
            id="usr_admin",
            username="admin",
            email="admin@example.com",
            password_hash=self.get_password_hash(default_password),
            roles=[UserRole.ADMIN],
            created_at=datetime.now()
        )

        # Store admin user
        self.users[admin_user.id] = admin_user
        self._save_users()

        logger.info("Created default admin user")


# Create auth handler instance
auth_handler = AuthHandler()


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserAuth:
    """
    Get the current user from a JWT token.

    Args:
        token: JWT token

    Returns:
        User associated with the token

    Raises:
        HTTPException: If token is invalid or user not found
    """
    # Verify token
    token_data = auth_handler.verify_jwt_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Check if token is expired
    if datetime.now() > token_data.exp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Get user
    user = auth_handler.get_user(token_data.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


async def get_current_api_key(api_key: str = Security(api_key_header)) -> APIKey:
    """
    Get the current API key.

    Args:
        api_key: API key header value

    Returns:
        API key

    Raises:
        HTTPException: If API key is invalid or expired
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "APIKey"}
        )

    # Get API key
    key = auth_handler.get_api_key(api_key)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"}
        )

    # Check if key is revoked
    if key.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
            headers={"WWW-Authenticate": "APIKey"}
        )

    # Check if key is expired
    if key.expires_at and datetime.now() > key.expires_at:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
            headers={"WWW-Authenticate": "APIKey"}
        )

    # Update last used timestamp
    key.last_used_at = datetime.now()
    auth_handler._save_api_keys()

    return key


async def get_user_or_api_key(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> Union[UserAuth, APIKey]:
    """
    Get the current user from either JWT token or API key.

    Args:
        token: Optional JWT token
        api_key: Optional API key

    Returns:
        User or API key

    Raises:
        HTTPException: If no valid authentication provided
    """
    if token:
        return await get_current_user(token)
    elif api_key:
        return await get_current_api_key(api_key)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )


def check_permissions(required_roles: List[UserRole]):
    """
    Dependency for checking if user has required roles.

    Args:
        required_roles: List of roles required for access

    Returns:
        Dependency function
    """
    async def is_authorized(user: Union[UserAuth, APIKey] = Depends(get_user_or_api_key)) -> bool:
        if not auth_handler.has_permission(user.roles, required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized. Required roles: {[r.value for r in required_roles]}"
            )
        return True

    return is_authorized


# Common permissions
require_admin = check_permissions([UserRole.ADMIN])
require_user = check_permissions([UserRole.USER, UserRole.ADMIN])
require_api = check_permissions([UserRole.API, UserRole.ADMIN])