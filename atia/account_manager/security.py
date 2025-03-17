"""
Security manager for handling encryption, key storage, and other security features.

This module provides functionality for securely handling API credentials.
"""

import os
import base64
import json
import logging
import datetime
from typing import Dict, Any, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from atia.config import settings


class SecurityManager:
    """
    Handles security operations for the Account Manager.

    This class provides encryption, decryption, and secure storage capabilities.
    """

    def __init__(self, 
                 encryption_key: Optional[str] = None,
                 salt: Optional[bytes] = None):
        """
        Initialize the security manager.

        Args:
            encryption_key: Optional encryption key, will use environment variable if not provided
            salt: Optional salt for key derivation, will generate if not provided
        """
        self.logger = logging.getLogger(__name__)

        # Initialize encryption key
        self._encryption_key = encryption_key or self._get_or_create_encryption_key()

        # Initialize salt for key derivation
        self._salt = salt or os.urandom(16)

        # Initialize Fernet cipher
        self.cipher = self._setup_encryption()

        # Track audit log events
        self.audit_log = []

    def _get_or_create_encryption_key(self) -> str:
        """
        Get the encryption key from environment or create a new one.

        Returns:
            Encryption key as string
        """
        env_key = os.environ.get("ATIA_ENCRYPTION_KEY", "")

        if not env_key:
            # Generate a new key
            self.logger.warning("No encryption key found in environment. Generating a new one.")
            key = Fernet.generate_key().decode('utf-8')
            self.logger.info("New encryption key generated. Set ATIA_ENCRYPTION_KEY in your environment.")
            return key

        return env_key

    def _setup_encryption(self) -> Fernet:
        """
        Set up Fernet encryption with key derivation.

        Returns:
            Fernet encryption object
        """
        # Use PBKDF2 to derive a key from the master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=100000,
        )

        # Convert the encryption key string to bytes
        key_bytes = self._encryption_key.encode('utf-8')

        # Derive the key
        key = base64.urlsafe_b64encode(kdf.derive(key_bytes))

        # Create Fernet cipher
        return Fernet(key)

    def encrypt(self, data: str) -> str:
        """
        Encrypt sensitive data.

        Args:
            data: Data to encrypt as string

        Returns:
            Encrypted data as string
        """
        try:
            encrypted = self.cipher.encrypt(data.encode('utf-8'))
            self._log_audit_event("encrypt", success=True)
            return encrypted.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            self._log_audit_event("encrypt", success=False, error=str(e))
            raise

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt encrypted data.

        Args:
            encrypted_data: Encrypted data as string

        Returns:
            Decrypted data as string
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode('utf-8'))
            self._log_audit_event("decrypt", success=True)
            return decrypted.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            self._log_audit_event("decrypt", success=False, error=str(e))
            raise

    def secure_store(self, key_id: str, data: Dict[str, Any]) -> None:
        """
        Securely store credentials.

        In a production environment, this would use AWS Secrets Manager or equivalent.
        For Phase 2, this is a simple implementation storing to an encrypted file.

        Args:
            key_id: Unique identifier for the credential
            data: Credential data to store
        """
        try:
            # For Phase 2, we'll just log the operation and implement a basic file storage
            self.logger.info(f"Storing credential with ID: {key_id}")

            # Create a storage directory if it doesn't exist
            os.makedirs('data/credentials', exist_ok=True)

            # Encrypt the data
            encrypted_data = self.encrypt(json.dumps(data))

            # Store in file (simple implementation for Phase 2)
            with open(f"data/credentials/{key_id}.enc", "w") as f:
                f.write(encrypted_data)

            self._log_audit_event("store", key_id=key_id, success=True)
        except Exception as e:
            self.logger.error(f"Error storing credential {key_id}: {str(e)}")
            self._log_audit_event("store", key_id=key_id, success=False, error=str(e))
            raise

    def secure_retrieve(self, key_id: str) -> Dict[str, Any]:
        """
        Securely retrieve credentials.

        Args:
            key_id: Unique identifier for the credential

        Returns:
            Decrypted credential data
        """
        try:
            self.logger.info(f"Retrieving credential with ID: {key_id}")

            # Check if the file exists
            file_path = f"data/credentials/{key_id}.enc"
            if not os.path.exists(file_path):
                self.logger.error(f"Credential {key_id} not found")
                self._log_audit_event("retrieve", key_id=key_id, success=False, error="Not found")
                raise FileNotFoundError(f"Credential {key_id} not found")

            # Read and decrypt the data
            with open(file_path, "r") as f:
                encrypted_data = f.read()

            decrypted_data = self.decrypt(encrypted_data)

            self._log_audit_event("retrieve", key_id=key_id, success=True)
            return json.loads(decrypted_data)
        except Exception as e:
            self.logger.error(f"Error retrieving credential {key_id}: {str(e)}")
            self._log_audit_event("retrieve", key_id=key_id, success=False, error=str(e))
            raise

    def secure_delete(self, key_id: str) -> None:
        """
        Securely delete credentials.

        Args:
            key_id: Unique identifier for the credential
        """
        try:
            self.logger.info(f"Deleting credential with ID: {key_id}")

            # Check if the file exists
            file_path = f"data/credentials/{key_id}.enc"
            if not os.path.exists(file_path):
                self.logger.warning(f"Credential {key_id} not found for deletion")
                return

            # Delete the file
            os.remove(file_path)

            self._log_audit_event("delete", key_id=key_id, success=True)
        except Exception as e:
            self.logger.error(f"Error deleting credential {key_id}: {str(e)}")
            self._log_audit_event("delete", key_id=key_id, success=False, error=str(e))
            raise

    def rotate_key(self, key_id: str) -> None:
        """
        Rotate an API key or token.

        For Phase 2, this is a placeholder method.

        Args:
            key_id: Unique identifier for the credential
        """
        self.logger.info(f"Key rotation for {key_id} requested (not implemented in Phase 2)")
        self._log_audit_event("rotate", key_id=key_id, success=True, note="Not implemented in Phase 2")

    def check_expiry(self, key_id: str) -> bool:
        """
        Check if a credential is expired.

        Args:
            key_id: Unique identifier for the credential

        Returns:
            True if expired, False otherwise
        """
        try:
            credential = self.secure_retrieve(key_id)

            if "expiration_date" in credential:
                expiry_str = credential["expiration_date"]
                if expiry_str:
                    expiry = datetime.datetime.fromisoformat(expiry_str)
                    now = datetime.datetime.now()
                    return now > expiry

            return False
        except Exception as e:
            self.logger.error(f"Error checking expiry for {key_id}: {str(e)}")
            return True  # Assume expired if there's an error

    def _log_audit_event(self, action: str, key_id: Optional[str] = None, 
                        success: bool = True, error: Optional[str] = None,
                        note: Optional[str] = None) -> None:
        """
        Log an audit event.

        Args:
            action: The action performed (encrypt, decrypt, store, retrieve, delete, rotate)
            key_id: Optional key ID the action was performed on
            success: Whether the action was successful
            error: Optional error message if the action failed
            note: Optional additional information
        """
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "success": success
        }

        if key_id:
            event["key_id"] = key_id

        if error:
            event["error"] = error

        if note:
            event["note"] = note

        # Add to in-memory audit log (for Phase 2)
        self.audit_log.append(event)

        # In a production environment, this would be persisted to a database or log file