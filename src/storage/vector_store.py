"""
Vector storage using FAISS for similarity search.
"""
import faiss
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

import config


class MultiModalVectorStore:
    """Vector store for managing multi-modal embeddings using FAISS."""

    def __init__(self):
        self.indices: Dict[str, faiss.Index] = {}  # modality -> FAISS index
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}  # modality -> list of metadata
        self.dimension = config.EMBEDDING_DIMENSION

    def add_vectors(
        self,
        modality: str,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Add vectors to the store for a specific modality.

        Args:
            modality: Modality type ('text' or 'table')
            embeddings: Numpy array of embeddings (num_vectors x dimension)
            metadata: List of metadata dicts for each vector
        """
        if len(embeddings) == 0:
            return

        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Create index if it doesn't exist
        if modality not in self.indices:
            self.indices[modality] = faiss.IndexFlatL2(self.dimension)
            self.metadata[modality] = []

        # Add vectors to index
        self.indices[modality].add(embeddings)

        # Add metadata
        self.metadata[modality].extend(metadata)

    def query_similar(
        self,
        modality: str,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Query for similar vectors.

        Args:
            modality: Modality type to search in
            query_vector: Query vector (1D array of dimension)
            k: Number of results to return

        Returns:
            List of (index, distance, metadata) tuples
        """
        if modality not in self.indices or self.indices[modality].ntotal == 0:
            return []

        # Ensure query vector is 2D and float32
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        # Search
        k = min(k, self.indices[modality].ntotal)
        distances, indices = self.indices[modality].search(query_vector, k)

        # Compile results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata[modality]):
                results.append((
                    int(idx),
                    float(distance),
                    self.metadata[modality][idx]
                ))

        return results

    def get_all_vectors(self, modality: str) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Get all vectors and metadata for a modality.

        Args:
            modality: Modality type

        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if modality not in self.indices or self.indices[modality].ntotal == 0:
            return None, []

        # Reconstruct vectors from index
        num_vectors = self.indices[modality].ntotal
        embeddings = faiss.rev_swig_ptr(
            self.indices[modality].get_xb(),
            num_vectors * self.dimension
        ).reshape(num_vectors, self.dimension)

        return embeddings, self.metadata[modality]

    def get_num_vectors(self, modality: str) -> int:
        """
        Get number of vectors stored for a modality.

        Args:
            modality: Modality type

        Returns:
            Number of vectors
        """
        if modality not in self.indices:
            return 0
        return self.indices[modality].ntotal

    def save(self, filename_prefix: str) -> None:
        """
        Save indices and metadata to disk.

        Args:
            filename_prefix: Prefix for saved files
        """
        save_dir = config.VECTOR_STORE_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        for modality, index in self.indices.items():
            # Save FAISS index
            index_path = save_dir / f"{filename_prefix}_{modality}.faiss"
            faiss.write_index(index, str(index_path))

            # Save metadata
            metadata_path = save_dir / f"{filename_prefix}_{modality}_metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata[modality], f)

    def load(self, filename_prefix: str) -> bool:
        """
        Load indices and metadata from disk.

        Args:
            filename_prefix: Prefix of saved files

        Returns:
            True if loaded successfully, False otherwise
        """
        load_dir = config.VECTOR_STORE_DIR

        try:
            # Find all index files with this prefix
            for modality in ["text", "table"]:
                index_path = load_dir / f"{filename_prefix}_{modality}.faiss"
                metadata_path = load_dir / f"{filename_prefix}_{modality}_metadata.pkl"

                if index_path.exists() and metadata_path.exists():
                    # Load FAISS index
                    self.indices[modality] = faiss.read_index(str(index_path))

                    # Load metadata
                    with open(metadata_path, "rb") as f:
                        self.metadata[modality] = pickle.load(f)

            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def clear(self) -> None:
        """Clear all indices and metadata."""
        self.indices.clear()
        self.metadata.clear()
