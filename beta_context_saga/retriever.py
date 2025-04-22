import os
from typing import TypeVar

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Define a type variable for context ID
ContextId = TypeVar("ContextId", str, int)


class Retriever:
    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        persist_dir: str = "data/chroma",
    ):
        """Initialize the retriever with a model name and disk persistence."""
        # Load the Sentence Transformer model
        self.model = SentenceTransformer(model_name)
        # Get embedding dimension from the model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Ensure persistence directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize ChromaDB with disk persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection for storing embeddings
        self.collection = self.client.get_or_create_collection(
            name="memory_vectors",
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # In-memory cache of contexts for faster lookups
        self._contexts_cache = {}

    def add_context(self, context: str, context_id: ContextId) -> None:
        """Add a single context to the retriever."""
        # Get embedding
        embedding = self.get_embedding(context).tolist()

        # Add to ChromaDB
        self.collection.upsert(
            ids=[str(context_id)],  # Ensure string format for ChromaDB
            embeddings=[embedding],
            metadatas=[{"text": context}],
            documents=[context],
        )

        # Update cache
        self._contexts_cache[context_id] = context

    def add_contexts(self, contexts: list[str], context_ids: list[ContextId]) -> None:
        """Add multiple contexts to the retriever."""
        if len(contexts) != len(context_ids):
            raise ValueError("Length of contexts and context_ids must match")

        # Get embeddings for all contexts
        embeddings = [self.get_embedding(ctx).tolist() for ctx in contexts]

        # Add to ChromaDB (ensuring ids are strings)
        self.collection.upsert(
            ids=[str(ctx_id) for ctx_id in context_ids],
            embeddings=embeddings,
            metadatas=[{"text": ctx} for ctx in contexts],
            documents=contexts,
        )

        # Update cache
        for i, context_id in enumerate(context_ids):
            self._contexts_cache[context_id] = contexts[i]

    def remove_context(self, context_id: ContextId) -> bool:
        """Remove a context by id."""
        try:
            self.collection.delete(ids=[str(context_id)])
            if context_id in self._contexts_cache:
                del self._contexts_cache[context_id]
            return True
        except Exception:
            return False

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text."""
        # Compute the embedding, normalizing for consistency with retrieval
        tensor = self.model.encode(text, normalize_embeddings=True)
        # Convert PyTorch tensor to numpy array if needed
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        # If already numpy array or other format that can be converted
        return np.asarray(tensor)

    def retrieve_context(self, query: str, k: int = 5) -> list[tuple[ContextId, str, float]]:
        """
        Retrieve the top-k most relevant contexts given a query.

        Returns a list of tuples containing (context_id, context_text, similarity_score)
        """
        # Get collection count to handle empty case
        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query).tolist()

        # Limit k to available documents
        k = min(k, collection_count)

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results to match expected output
            formatted_results = []

            # Check if we have valid results
            if (
                not results
                or "ids" not in results
                or not results["ids"]
                or len(results["ids"]) == 0
                or not results["documents"]
                or not results["distances"]
            ):
                return []

            for i in range(len(results["ids"][0])):
                context_id_str = results["ids"][0][i]
                # Try to convert to int if possible, otherwise keep as string
                context_id = self._convert_context_id(context_id_str)
                context = results["documents"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - (distance / 2.0)  # Convert distance to similarity score

                # Update cache
                self._contexts_cache[context_id] = context

                formatted_results.append((context_id, context, similarity))

            return formatted_results
        except Exception:
            # If any error occurs during retrieval, return empty results
            return []

    def retrieve_by_embedding(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> list[tuple[ContextId, str, float]]:
        """
        Retrieve the top-k most relevant contexts given a query embedding.

        Returns a list of tuples containing (context_id, context_text, similarity_score)
        """
        # Get collection count to handle empty case
        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        # Limit k to available documents
        k = min(k, collection_count)

        # Ensure embedding is in the right format
        if not isinstance(query_embedding, list):
            query_embedding = query_embedding.tolist()

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results to match expected output
            formatted_results = []

            # Check if we have valid results
            if (
                not results
                or "ids" not in results
                or not results["ids"]
                or len(results["ids"]) == 0
                or not results["documents"]
                or not results["distances"]
            ):
                return []

            for i in range(len(results["ids"][0])):
                context_id_str = results["ids"][0][i]
                # Try to convert to int if possible, otherwise keep as string
                context_id = self._convert_context_id(context_id_str)
                context = results["documents"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - (distance / 2.0)  # Convert distance to similarity score

                # Update cache
                self._contexts_cache[context_id] = context

                formatted_results.append((context_id, context, similarity))

            return formatted_results
        except Exception:
            # If any error occurs during retrieval, return empty results
            return []

    def _convert_context_id(self, context_id_str: str) -> ContextId:
        """Convert a context ID string to the appropriate type (int or str)."""
        try:
            # Try to convert to int
            return int(context_id_str)
        except (ValueError, TypeError):
            # If conversion fails, return as string
            return context_id_str
