"""
Response caching mechanism for improved performance.

This module provides LRU (Least Recently Used) caching for API responses
to reduce latency and API usage.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import threading

logger = logging.getLogger(__name__)

class ResponseCache:
    """
    LRU Cache for API responses with time-based expiration.

    This implementation uses a simple in-memory dictionary with an ordered
    list to track LRU items. More sophisticated implementations could use
    Redis or another distributed cache.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the response cache.

        Args:
            max_size: Maximum number of items to keep in cache
            ttl_seconds: Time to live in seconds for cache items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.lru_keys: List[str] = []  # Ordered list of keys from least to most recently used
        self.lock = threading.RLock()  # Thread-safe access
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Start background cleanup thread for expired items
        self._start_cleanup_thread()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, timestamp = self.cache[key]

            # Check if item has expired
            if time.time() - timestamp > self.ttl_seconds:
                # Remove expired item
                self._remove_item(key)
                self.misses += 1
                return None

            # Update LRU order (move to most recently used)
            self._update_lru(key)

            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """
        Add or update an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # If key exists, update it
            if key in self.cache:
                self._remove_item(key)

            # Check if cache is full
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self._evict_lru()

            # Add new item
            self.cache[key] = (value, time.time())
            self.lru_keys.append(key)

    def _remove_item(self, key: str) -> None:
        """Remove an item from the cache."""
        if key in self.cache:
            del self.cache[key]
            self.lru_keys.remove(key)

    def _update_lru(self, key: str) -> None:
        """Update the LRU order for a key."""
        if key in self.lru_keys:
            self.lru_keys.remove(key)
            self.lru_keys.append(key)

    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if self.lru_keys:
            lru_key = self.lru_keys[0]
            self._remove_item(lru_key)
            self.evictions += 1

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.lru_keys.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": hit_ratio,
                "evictions": self.evictions
            }

    def _start_cleanup_thread(self) -> None:
        """Start a background thread to periodically clean up expired items."""
        def cleanup_worker():
            while True:
                time.sleep(300)  # Run every 5 minutes
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        now = time.time()
        with self.lock:
            # Get keys of expired items
            expired_keys = [
                key for key, (_, timestamp) in self.cache.items()
                if now - timestamp > self.ttl_seconds
            ]

            # Remove expired items
            for key in expired_keys:
                self._remove_item(key)

            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired items from cache")


class VectorCache:
    """
    Cache for vector embeddings to avoid redundant embedding generation.
    """

    def __init__(self, max_size: int = 10000, ttl_days: int = 30):
        """
        Initialize the vector cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_days: Time to live in days for cached embeddings
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_days * 24 * 3600
        self.cache: Dict[str, Tuple[List[float], float]] = {}  # hash -> (embedding, timestamp)
        self.lru_keys: List[str] = []
        self.lock = threading.RLock()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None if not found or expired
        """
        key = self._hash_text(text)

        with self.lock:
            if key not in self.cache:
                return None

            embedding, timestamp = self.cache[key]

            # Check if embedding has expired
            if time.time() - timestamp > self.ttl_seconds:
                # Remove expired embedding
                self._remove_item(key)
                return None

            # Update LRU order
            self._update_lru(key)

            return embedding

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache embedding for text.

        Args:
            text: Text for which the embedding was generated
            embedding: The embedding to cache
        """
        key = self._hash_text(text)

        with self.lock:
            # If key exists, update it
            if key in self.cache:
                self._remove_item(key)

            # Check if cache is full
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self._evict_lru()

            # Add new item
            self.cache[key] = (embedding, time.time())
            self.lru_keys.append(key)

    def _hash_text(self, text: str) -> str:
        """Generate a hash for the text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _remove_item(self, key: str) -> None:
        """Remove an item from the cache."""
        if key in self.cache:
            del self.cache[key]
            self.lru_keys.remove(key)

    def _update_lru(self, key: str) -> None:
        """Update the LRU order for a key."""
        if key in self.lru_keys:
            self.lru_keys.remove(key)
            self.lru_keys.append(key)

    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if self.lru_keys:
            lru_key = self.lru_keys[0]
            self._remove_item(lru_key)

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.lru_keys.clear()