import json
import os
import sqlite3
from datetime import UTC, datetime

import numpy as np

from beta_context_saga.models import MemoryItem
from beta_context_saga.retriever import Retriever


class MemoryDatabase:
    """SQLite3-based memory database with vector search and tagging capabilities."""

    def __init__(
        self,
        db_path: str = "data/memory.db",
        retriever: Retriever | None = None,
        embedding_dim: int = 384,  # Default dimension for embeddings
    ):
        """Initialize the memory database with SQLite and vector retriever.

        Args:
            db_path: Path to the SQLite database file
            retriever: Optional Retriever instance for vector embeddings
            embedding_dim: Dimension of embedding vectors (used if retriever not provided)
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path

        # Initialize the retriever for vector embeddings if not provided
        self.retriever = retriever or Retriever()
        self.embedding_dim = embedding_dim if retriever is None else self.retriever.embedding_dim

        # Create tables if they don't exist
        self._create_tables()

    def _get_connection(self):
        """Get a database connection for the current thread."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Create memories table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP,
                expires_at TIMESTAMP,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 5.0,
                metadata TEXT,
                deleted_at TIMESTAMP
            )
            """)

            # Create embeddings table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
            """)

            # Create tags table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            """)

            # Create memory_tags junction table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (memory_id, tag_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )
            """)

            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """Close the database connection."""
        # No connection to close as we create them per method
        pass

    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close()

    def _vector_to_blob(self, vector: np.ndarray) -> bytes:
        """Convert a numpy vector to a blob for storage."""
        return vector.tobytes()

    def _blob_to_vector(self, blob: bytes) -> np.ndarray:
        """Convert a blob back to a numpy vector."""
        return np.frombuffer(blob, dtype=np.float32)

    def add_memory(self, memory: MemoryItem) -> int:
        """Add a memory item to the database with its embedding and tags."""
        conn = self._get_connection()
        cursor = conn.cursor()
        memory_id = None

        try:
            # Calculate expiration timestamp
            expires_at = None
            if memory.expires_in:
                expires_at = (memory.created_at + memory.expires_in).isoformat()

            # Convert metadata to JSON if present
            metadata_json = json.dumps(memory.metadata) if memory.metadata else None

            # Insert memory
            cursor.execute(
                """
            INSERT INTO memories
            (created_at, updated_at, expires_at, source, content, importance, metadata, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat() if memory.updated_at else None,
                    expires_at,
                    memory.source,
                    memory.content,
                    memory.importance,
                    metadata_json,
                    memory.deleted_at.isoformat() if memory.deleted_at else None,
                ),
            )

            memory_id = cursor.lastrowid

            # Generate and store embedding
            embedding = self.retriever.get_embedding(memory.content)
            cursor.execute(
                """
            INSERT INTO embeddings (memory_id, vector) VALUES (?, ?)
            """,
                (memory_id, self._vector_to_blob(embedding)),
            )

            # Process tags
            for tag in memory.tags:
                # Get or create tag
                cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                tag_id = cursor.fetchone()[0]

                # Link memory to tag
                cursor.execute(
                    """
                INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)
                """,
                    (memory_id, tag_id),
                )

            conn.commit()
            return memory_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def update_memory(self, memory: MemoryItem) -> bool:
        """Update an existing memory item."""
        conn = self._get_connection()
        cursor = conn.cursor()
        success = False

        try:
            # Check if memory exists
            cursor.execute("SELECT 1 FROM memories WHERE id = ?", (memory.id,))
            if not cursor.fetchone():
                return False

            # Calculate expiration timestamp
            expires_at = None
            if memory.expires_in:
                expires_at = (memory.created_at + memory.expires_in).isoformat()

            # Convert metadata to JSON if present
            metadata_json = json.dumps(memory.metadata) if memory.metadata else None

            # Update memory
            memory.updated_at = datetime.now(UTC)
            cursor.execute(
                """
            UPDATE memories SET
                updated_at = ?,
                expires_at = ?,
                source = ?,
                content = ?,
                importance = ?,
                metadata = ?,
                deleted_at = ?
            WHERE id = ?
            """,
                (
                    memory.updated_at.isoformat(),
                    expires_at,
                    memory.source,
                    memory.content,
                    memory.importance,
                    metadata_json,
                    memory.deleted_at.isoformat() if memory.deleted_at else None,
                    memory.id,
                ),
            )

            # Update embedding if content changed
            embedding = self.retriever.get_embedding(memory.content)
            cursor.execute(
                """
            UPDATE embeddings SET vector = ? WHERE memory_id = ?
            """,
                (self._vector_to_blob(embedding), memory.id),
            )

            # Update tags
            # First, remove all existing tag associations
            cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory.id,))

            # Then add the new tags
            for tag in memory.tags:
                # Get or create tag
                cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                tag_id = cursor.fetchone()[0]

                # Link memory to tag
                cursor.execute(
                    """
                INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)
                """,
                    (memory.id, tag_id),
                )

            conn.commit()
            success = True
            return success
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        """Retrieve a memory item by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        memory = None

        try:
            # Get memory
            cursor.execute(
                """
            SELECT * FROM memories WHERE id = ? AND (deleted_at IS NULL OR deleted_at > ?)
            """,
                (memory_id, datetime.now(UTC).isoformat()),
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Get tags for this memory
            cursor.execute(
                """
            SELECT t.name FROM tags t
            JOIN memory_tags mt ON t.id = mt.tag_id
            WHERE mt.memory_id = ?
            """,
                (memory_id,),
            )
            tags = [tag[0] for tag in cursor.fetchall()]

            # Parse expiration if present
            expires_in = None
            if row["expires_at"]:
                expires_at = datetime.fromisoformat(row["expires_at"])
                created_at = datetime.fromisoformat(row["created_at"])
                expires_in = expires_at - created_at

            # Parse metadata if present
            metadata = json.loads(row["metadata"]) if row["metadata"] else None

            # Create memory item
            memory = MemoryItem(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                expires_in=expires_in,
                source=row["source"],
                content=row["content"],
                importance=row["importance"],
                tags=tags,
                metadata=metadata,
                deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
            )

            return memory
        finally:
            conn.close()

    def delete_memory(self, memory_id: int, hard_delete: bool = False) -> bool:
        """Delete a memory item.

        Args:
            memory_id: ID of the memory to delete
            hard_delete: If True, permanently delete; if False, soft delete by marking deleted_at

        Returns:
            True if successful, False if memory not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        success = False

        try:
            # Check if memory exists
            cursor.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,))
            if not cursor.fetchone():
                return False

            if hard_delete:
                # Hard delete - remove from database
                cursor.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
                cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                # Soft delete - mark as deleted
                cursor.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (datetime.now(UTC).isoformat(), memory_id),
                )

            conn.commit()
            success = True
            return success
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def list_memories(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "DESC",
        include_deleted: bool = False,
    ) -> list[MemoryItem]:
        """List memory items with pagination and sorting.

        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
            sort_by: Field to sort by (created_at, updated_at, importance, id)
            sort_order: Sort order (ASC or DESC)
            include_deleted: Whether to include soft-deleted memories

        Returns:
            List of memory items
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        memories = []

        try:
            # Validate sort parameters to prevent SQL injection
            valid_sort_fields = ["created_at", "updated_at", "importance", "id"]
            if sort_by not in valid_sort_fields:
                sort_by = "created_at"

            if sort_order not in ["ASC", "DESC"]:
                sort_order = "DESC"

            # Construct the query
            query = "SELECT id FROM memories "
            if not include_deleted:
                query += "WHERE deleted_at IS NULL "
            query += f"ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?"

            # Get memory IDs
            cursor.execute(query, (limit, offset))
            memory_ids = [row[0] for row in cursor.fetchall()]

            # Get full memory items
            for memory_id in memory_ids:
                memory = self.get_memory(memory_id)
                if memory:
                    memories.append(memory)

            return memories
        finally:
            conn.close()

    def search_by_text(self, query: str, limit: int = 10) -> list[tuple[MemoryItem, float]]:
        """Search for memories by text similarity.

        Args:
            query: Text to search for
            limit: Maximum number of results to return

        Returns:
            List of (memory_item, similarity_score) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []

        try:
            # Get query embedding
            query_embedding = self.retriever.get_embedding(query)

            # Get all non-deleted memory embeddings
            cursor.execute(
                """
            SELECT m.id, e.vector FROM memories m
            JOIN embeddings e ON m.id = e.memory_id
            WHERE m.deleted_at IS NULL
            """
            )
            rows = cursor.fetchall()

            # Calculate similarity scores
            scores = []
            for row in rows:
                memory_id = row["id"]
                vector_blob = row["vector"]
                memory_embedding = self._blob_to_vector(vector_blob)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                scores.append((memory_id, similarity))

            # Sort by similarity (descending) and get top results
            scores.sort(key=lambda x: x[1], reverse=True)
            top_scores = scores[:limit]

            # Get full memory items for top results
            for memory_id, similarity in top_scores:
                memory = self.get_memory(memory_id)
                if memory:
                    results.append((memory, similarity))

            return results
        finally:
            conn.close()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def search_by_tags(
        self, tags: list[str], require_all: bool = False, limit: int = 100
    ) -> list[MemoryItem]:
        """Search for memories by tags.

        Args:
            tags: List of tags to search for
            require_all: If True, require all tags; if False, require any tag
            limit: Maximum number of results to return

        Returns:
            List of memory items
        """
        if not tags:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()
        memories = []

        try:
            # Construct tag placeholders
            placeholders = ", ".join(["?"] * len(tags))

            # Construct the query based on whether all tags are required
            if require_all:
                # Get memories that have ALL the specified tags
                query = f"""
                SELECT m.id, COUNT(DISTINCT t.name) as tag_count
                FROM memories m
                JOIN memory_tags mt ON m.id = mt.memory_id
                JOIN tags t ON mt.tag_id = t.id
                WHERE t.name IN ({placeholders})
                    AND m.deleted_at IS NULL
                GROUP BY m.id
                HAVING tag_count = ?
                LIMIT ?
                """
                cursor.execute(query, tags + [len(tags), limit])
            else:
                # Get memories that have ANY of the specified tags
                query = f"""
                SELECT m.id
                FROM memories m
                JOIN memory_tags mt ON m.id = mt.memory_id
                JOIN tags t ON mt.tag_id = t.id
                WHERE t.name IN ({placeholders})
                    AND m.deleted_at IS NULL
                GROUP BY m.id
                LIMIT ?
                """
                cursor.execute(query, tags + [limit])

            memory_ids = [row[0] for row in cursor.fetchall()]

            # Get full memory items
            for memory_id in memory_ids:
                memory = self.get_memory(memory_id)
                if memory:
                    memories.append(memory)

            return memories
        finally:
            conn.close()

    def list_all_tags(self) -> list[tuple[str, int]]:
        """List all tags and their occurrence counts.

        Returns:
            List of (tag_name, count) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []

        try:
            cursor.execute(
                """
            SELECT t.name, COUNT(mt.memory_id) as count
            FROM tags t
            JOIN memory_tags mt ON t.id = mt.tag_id
            JOIN memories m ON mt.memory_id = m.id
            WHERE m.deleted_at IS NULL
            GROUP BY t.name
            ORDER BY count DESC
            """
            )
            results = [(row[0], row[1]) for row in cursor.fetchall()]
            return results
        finally:
            conn.close()

    def delete_tag(self, tag: str) -> bool:
        """Delete a tag and remove all its associations.

        Args:
            tag: Tag name to delete

        Returns:
            True if successful, False if tag not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        success = False

        try:
            # Get tag ID
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
            row = cursor.fetchone()
            if not row:
                return False

            tag_id = row[0]

            # Delete tag associations
            cursor.execute("DELETE FROM memory_tags WHERE tag_id = ?", (tag_id,))

            # Delete tag
            cursor.execute("DELETE FROM tags WHERE id = ?", (tag_id,))

            conn.commit()
            success = True
            return success
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def cleanup_expired_memories(self) -> int:
        """Delete memories that have expired.

        Returns:
            Number of deleted memories
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        count = 0

        try:
            # Find expired memories
            current_time = datetime.now(UTC).isoformat()
            cursor.execute(
                """
            SELECT id FROM memories
            WHERE expires_at IS NOT NULL
                AND expires_at < ?
                AND deleted_at IS NULL
            """,
                (current_time,),
            )
            expired_ids = [row[0] for row in cursor.fetchall()]

            # Mark them as deleted
            for memory_id in expired_ids:
                cursor.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?", (current_time, memory_id)
                )
                count += 1

            conn.commit()
            return count
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_all_memories(self, limit: int = 100) -> list[MemoryItem]:
        """Get all memories, potentially filtered by attribute.

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of memory items
        """
        return self.list_memories(limit=limit)
