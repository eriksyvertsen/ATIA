"""
AWS Secrets Manager integration for secure credential storage.

This module provides secure storage of API credentials using AWS Secrets Manager.
"""

import json
import logging
import base64
from typing import Dict, Any, Optional

from atia.config import settings


logger = logging.getLogger(__name__)


class AWSSecurityManager:
    """
    Handles security operations using AWS Secrets Manager.
    """

    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize the AWS security manager.

        Args:
            region_name: AWS region name, defaults to settings or 'us-east-1'
        """
        self.region_name = region_name or getattr(settings, 'aws_region', 'us-east-1')
        self.logger = logging.getLogger(__name__)
        self.client = None

        # Initialize the AWS client
        try:
            import boto3
            from botocore.exceptions import ClientError

            # Store the import for later use
            self.ClientError = ClientError

            self.client = boto3.client(
                service_name='secretsmanager',
                region_name=self.region_name
            )
            self.logger.info(f"AWS Secrets Manager client initialized for region {self.region_name}")
        except ImportError:
            self.logger.warning("boto3 not installed. AWS Secrets Manager integration disabled.")
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Secrets Manager client: {e}")

    def store_secret(self, secret_id: str, secret_value: Dict[str, Any]) -> bool:
        """
        Store a secret in AWS Secrets Manager.

        Args:
            secret_id: Identifier for the secret
            secret_value: Secret value to store

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            self.logger.error("AWS Secrets Manager client not initialized")
            return False

        try:
            # Check if the secret already exists
            try:
                self.client.describe_secret(SecretId=secret_id)
                # Secret exists, update it
                response = self.client.update_secret(
                    SecretId=secret_id,
                    SecretString=json.dumps(secret_value)
                )
            except self.ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Secret doesn't exist, create it
                    response = self.client.create_secret(
                        Name=secret_id,
                        SecretString=json.dumps(secret_value),
                        Tags=[
                            {
                                'Key': 'Application',
                                'Value': 'ATIA'
                            },
                        ]
                    )
                else:
                    raise

            self.logger.info(f"Secret {secret_id} stored successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error storing secret {secret_id}: {e}")
            return False

    def get_secret(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret from AWS Secrets Manager.

        Args:
            secret_id: Identifier for the secret

        Returns:
            Secret value as dictionary, or None if not found
        """
        if not self.client:
            self.logger.error("AWS Secrets Manager client not initialized")
            return None

        try:
            response = self.client.get_secret_value(SecretId=secret_id)
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                self.logger.error(f"Secret {secret_id} does not contain a string value")
                return None
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                self.logger.warning(f"Secret {secret_id} not found")
                return None
            else:
                self.logger.error(f"Error retrieving secret {secret_id}: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving secret {secret_id}: {e}")
            return None

    def delete_secret(self, secret_id: str, force_delete: bool = False) -> bool:
        """
        Delete a secret from AWS Secrets Manager.

        Args:
            secret_id: Identifier for the secret
            force_delete: If True, permanently delete without recovery window

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            self.logger.error("AWS Secrets Manager client not initialized")
            return False

        try:
            if force_delete:
                response = self.client.delete_secret(
                    SecretId=secret_id,
                    ForceDeleteWithoutRecovery=True
                )
            else:
                response = self.client.delete_secret(
                    SecretId=secret_id,
                    RecoveryWindowInDays=30
                )

            self.logger.info(f"Secret {secret_id} deleted successfully")
            return True
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                self.logger.warning(f"Secret {secret_id} not found for deletion")
                return False
            else:
                self.logger.error(f"Error deleting secret {secret_id}: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Unexpected error deleting secret {secret_id}: {e}")
            return False

    def rotate_secret(self, secret_id: str, new_secret_value: Dict[str, Any]) -> bool:
        """
        Rotate a secret in AWS Secrets Manager.

        Args:
            secret_id: Identifier for the secret
            new_secret_value: New secret value

        Returns:
            True if successful, False otherwise
        """
        return self.store_secret(secret_id, new_secret_value)

    def is_available(self) -> bool:
        """
        Check if AWS Secrets Manager is available.

        Returns:
            True if AWS Secrets Manager is available, False otherwise
        """
        return self.client is not None