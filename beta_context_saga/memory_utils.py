from datetime import UTC, datetime, timedelta

from beta_context_saga.memory_db import MemoryDatabase
from beta_context_saga.models import MemoryItem


def create_test_memory_items() -> list[MemoryItem]:
    """Create some test memory items for demonstration."""
    return [
        MemoryItem(
            id=0,  # Will be assigned by the database
            created_at=datetime.now(UTC),
            expires_in=timedelta(days=30),
            source="user",
            content="Python is a high-level programming language with dynamic\
                  typing and garbage collection.",
            tags=["programming", "python", "language"],
            importance=8.0,
            metadata={"category": "knowledge", "verified": True},
        ),
        MemoryItem(
            id=0,  # Will be assigned by the database
            created_at=datetime.now(UTC),
            expires_in=timedelta(days=60),
            source="system",
            content="SQLite is a C library that provides a lightweight disk-based database.",
            tags=["programming", "database", "sqlite"],
            importance=7.5,
            metadata={"category": "knowledge", "verified": True},
        ),
        MemoryItem(
            id=0,  # Will be assigned by the database
            created_at=datetime.now(UTC),
            expires_in=timedelta(days=14),
            source="user",
            content="Remember to finish the vector search implementation by next Friday.",
            tags=["todo", "project", "vector-search"],
            importance=9.0,
            metadata={
                "category": "reminder",
                "deadline": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
            },
        ),
        MemoryItem(
            id=0,  # Will be assigned by the database
            created_at=datetime.now(UTC),
            expires_in=timedelta(days=365),
            source="user",
            content="Machine learning is a field of study that gives computers the ability to learn\
                  without being explicitly programmed.",
            tags=["machine-learning", "ai", "computer-science"],
            importance=8.5,
            metadata={"category": "knowledge", "verified": True},
        ),
        MemoryItem(
            id=0,  # Will be assigned by the database
            created_at=datetime.now(UTC),
            expires_in=timedelta(days=7),
            source="system",
            content="Meeting with the dev team scheduled for Monday at 10 AM to discuss vector \
                database implementation.",
            tags=["meeting", "calendar", "vector-search"],
            importance=7.0,
            metadata={"category": "calendar", "location": "Conference Room B"},
        ),
    ]


def initialize_database(memory_db: MemoryDatabase) -> list[int]:
    """Initialize the database with test data and return the assigned IDs."""
    memory_items = create_test_memory_items()
    memory_ids = []

    for item in memory_items:
        memory_id = memory_db.add_memory(item)
        memory_ids.append(memory_id)
        print(f"Added memory item with ID: {memory_id}")

    return memory_ids


def demonstrate_memory_operations() -> None:
    """Demonstrate basic memory operations."""
    db = MemoryDatabase()
    print("Initializing database with test items...")
    memory_ids = initialize_database(db)

    # Demonstrate fetching a memory item
    print("\n----- Fetching Memory Item -----")
    if memory_ids:
        memory = db.get_memory(memory_ids[0])
        if memory:
            print(f"Retrieved memory: {memory.content}")
            print(f"Tags: {memory.tags}")
            print(f"Importance: {memory.importance}")

    # Demonstrate updating a memory item
    print("\n----- Updating Memory Item -----")
    if memory_ids:
        memory = db.get_memory(memory_ids[0])
        if memory:
            memory.content += " It's one of the most popular programming languages."
            memory.importance = 9.0
            memory.tags.append("popular")
            success = db.update_memory(memory)
            print(f"Update status: {'Success' if success else 'Failed'}")

            # Verify update
            updated_memory = db.get_memory(memory_ids[0])
            if updated_memory:
                print(f"Updated content: {updated_memory.content}")
                print(f"Updated tags: {updated_memory.tags}")

    # Demonstrate listing all memories
    print("\n----- Listing All Memories -----")
    memories = db.list_memories(limit=10)
    for memory in memories:
        print(f"ID: {memory.id}, Content: {memory.content[:50]}...")

    # Demonstrate tag operations
    print("\n----- Listing All Tags -----")
    tags = db.list_all_tags()
    for tag, count in tags:
        print(f"Tag: {tag}, Count: {count}")

    print("\n----- Search by Tags -----")
    vector_search_memories = db.search_by_tags(["vector-search"])
    print(f"Found {len(vector_search_memories)} memories with 'vector-search' tag:")
    for memory in vector_search_memories:
        print(f"ID: {memory.id}, Content: {memory.content[:50]}...")

    # Demonstrate vector search
    print("\n----- Vector Search -----")
    query = "database implementation with vectors"
    search_results = db.search_by_text(query)
    print(f"Search results for query: '{query}'")
    for memory, similarity in search_results:
        print(f"ID: {memory.id}, Similarity: {similarity:.4f}")
        print(f"Content: {memory.content[:100]}...")
        print("---")

    # Clean up (optional)
    if memory_ids:
        print("\n----- Deleting a Memory Item -----")
        delete_id = memory_ids[-1]
        success = db.delete_memory(delete_id)
        print(f"Deleted memory {delete_id}: {'Success' if success else 'Failed'}")

        # Verify deletion
        memories_after = db.list_memories(limit=10)
        print(f"Memories after deletion: {len(memories_after)}")

    print("\nClosing database connection...")
    db.close()


if __name__ == "__main__":
    demonstrate_memory_operations()
