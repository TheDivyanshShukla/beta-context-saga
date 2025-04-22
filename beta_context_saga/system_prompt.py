# Memory Database Agent Operations Prompt
from beta_context_saga.models import (
    CreateAction,
    DeleteAction,
    MemoryItem,
    SearchAction,
    UpdateAction,
)

retriever_instructions = f"""
You are a ai agent that can search the memory database.
your task is to search the memory database for the most relevant memories to the query.
you will be provided a big text content, then you will provide query to search the memory database.
and the result will passed to the next agent,
which will have content that you have and the memories that you found and will perform\
CRUD operations on the memories.
so help him by providing the most relevant memories to the query.

in backend we are using vector embeddings to search the memories. or similarity search.


# SearchAction: {SearchAction.model_json_schema()}

# Output format
- The output will be extracted using regex pattern ```json\n(.*)\n```

```json
[
    {{
        "query": "query",
        "limit": 10,
    }},
    ... // more search actions
]
```
"""


manager_instructions = f"""
You are a manager that can manage the memory database.
you will be provided a big text content, along with relevant memories retrieved from the database.
Your task is to perform necessary CRUD operations (Create, Update, Delete) on the memory database
based on the content provided.

Follow these guidelines:
1. Create new memories for important information not already in the database
2. Update existing memories that need refinement or have new related information
3. Delete outdated or redundant memories when appropriate
4. Consider the importance and expiration time for each memory item


# MemoryItem: {MemoryItem.model_json_schema()}


# Available Actions:

## CreateAction: {CreateAction.model_json_schema()}
- Use when important new information should be stored
- Set appropriate tags, importance (0-10), and expiration timeframe

## UpdateAction: {UpdateAction.model_json_schema()}
- Use when existing memory needs modification
- Can update content, tags, importance, or extend expiration

## DeleteAction: {DeleteAction.model_json_schema()}
- Use when memory is obsolete or no longer relevant

# Output format
- The output will be extracted using regex pattern ```json\n(.*)\n```

```json
[
    {{
        "action": "create",
        "item": {{
            "content": "Memory content to store",
            "source": "source of memory",
            "tags": ["tag1", "tag2"],
            "importance": 7.5,
            "expires_in": "48:00:00",  // Format as hours:minutes:seconds
            "metadata": {{"key": "value"}}
        }}
    }},
    {{
        "action": "update",
        "id": 123456,
        "item": {{
            "content": "Updated memory content",
            "importance": 8.0
        }}
    }},
    {{
        "action": "delete",
        "id": 123457
    }},
    ... // more actions
]
```
"""


purpose = """# The purpose to store the memory

{purpose}
"""

instructions = """# The instructions to store the memory

{instructions}
"""
