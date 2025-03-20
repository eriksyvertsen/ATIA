"""
Lightweight file-based vector store for embeddings.

Provides vector storage and similarity search capabilities without external dependencies.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class FileVectorStore:
    """
    File-based vector store that persists embeddings to disk and provides similarity search.

    A lightweight alternative to Pinecone for Replit environments.
    """

    def __init__(self, storage_dir: str = "data/vectors", dimension: int = 1536):
        """
        Initialize the file vector store.

        Args:
            storage_dir: Directory to store vector data
            dimension: Dimension of stored vectors (default: 1536 for OpenAI embeddings)
        """
        self.storage_dir = storage_dir
        self.dimension = dimension
        self.index_file = os.path.join(storage_dir, "vector_index.json")
        self.embeddings_file = os.path.join(storage_dir, "embeddings.pkl")

        # In-memory index
        self.index = {}  # id -> metadata
        self.vectors = {}  # id -> embedding

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Load existing data if available
        self._load_data()

        logger.info(f"Initialized file vector store with {len(self.index)} vectors")

    def _load_data(self) -> None:
        """Load vector data from disk."""
        try:
            # Load index
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)

            # Load embeddings
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.vectors = pickle.load(f)

            logger.info(f"Loaded {len(self.index)} vectors from disk")
        except Exception as e:
            logger.error(f"Error loading vector data: {e}")
            self.index = {}
            self.vectors = {}

    def _save_data(self) -> None:
        """Save vector data to disk."""
        try:
            # Save index
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)

            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.vectors, f)

            logger.info(f"Saved {len(self.index)} vectors to disk")
        except Exception as e:
            logger.error(f"Error saving vector data: {e}")

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """
        Insert or update vectors in the store.

        Args:
            vectors: List of vector objects with id, values, and metadata
        """
        for vector in vectors:
            vector_id = vector.get('id')
            values = vector.get('values')
            metadata = vector.get('metadata', {})

            if not vector_id or not values:
                logger.warning(f"Skipping invalid vector: {vector}")
                continue

            # Store the vector and metadata
            self.index[vector_id] = metadata
            self.vectors[vector_id] = np.array(values)

        # Save to disk
        self._save_data()

    def query(self, vector: List[float], top_k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Perform similarity search for the closest vectors.

        Args:
            vector: Query vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results

        Returns:
            Results in Pinecone-compatible format
        """
        if not self.vectors:
            return {"matches": []}

        query_vector = np.array(vector)

        # Calculate cosine similarity for all vectors
        scores = {}
        for vec_id, vec_values in self.vectors.items():
            # Cosine similarity = dot product / (norm(a) * norm(b))
            dot_product = np.dot(query_vector, vec_values)
            norm_a = np.linalg.norm(query_vector)
            norm_b = np.linalg.norm(vec_values)

            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_a * norm_b)

            scores[vec_id] = similarity

        # Sort by similarity (highest first)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Format results like Pinecone
        matches = []
        for vec_id, score in sorted_results:
            match = {
                "id": vec_id,
                "score": float(score),
            }

            if include_metadata and vec_id in self.index:
                match["metadata"] = self.index[vec_id]

            matches.append(match)

        return {"matches": matches}

    def delete(self, ids: List[str] = None, delete_all: bool = False) -> None:
        """
        Delete vectors from the store.

        Args:
            ids: List of vector IDs to delete
            delete_all: If True, delete all vectors
        """
        if delete_all:
            self.index = {}
            self.vectors = {}
            self._save_data()
            logger.info("Deleted all vectors")
            return

        if ids:
            for vec_id in ids:
                if vec_id in self.index:
                    del self.index[vec_id]
                if vec_id in self.vectors:
                    del self.vectors[vec_id]

            self._save_data()
            logger.info(f"Deleted {len(ids)} vectors")

    def count_vectors(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.vectors)
    