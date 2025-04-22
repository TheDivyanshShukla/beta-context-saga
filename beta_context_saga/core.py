import hashlib
import json
import os
import pickle
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from token_count import TokenCount
from tqdm import tqdm

from beta_context_saga.memory_db import MemoryDatabase
from beta_context_saga.models import MemoryItem, SearchAction
from beta_context_saga.system_prompt import manager_instructions, retriever_instructions

load_dotenv()


class TokenCounter:
    """Utility class to count tokens in text."""

    def __init__(self):
        """Initialize with a specific model's tokenizer."""
        self.model_name = "gpt-4o"
        self.token_count = TokenCount(model_name=self.model_name)

    def num_tokens_from_string(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return self.token_count.num_tokens_from_string(text)

    def split_text_by_tokens(
        self, text: str, max_tokens: int = 24000, overlap_tokens: int = 1
    ) -> list[str]:
        """Split text into chunks of approximately max_tokens with overlap."""
        if not text:
            return []

        # If text is small enough, return it as a single chunk
        total_tokens = self.num_tokens_from_string(text)
        if total_tokens <= max_tokens:
            return [text]

        chunks = []
        text_length = len(text)
        position = 0

        while position < text_length:
            # Take a chunk of text
            end_position = position + max_tokens * 4  # Roughly estimate characters per token
            if end_position >= text_length:
                end_position = text_length

            chunk = text[position:end_position]

            # Check token count and adjust if necessary
            chunk_tokens = self.num_tokens_from_string(chunk)

            # Binary search to find the right size chunk
            if chunk_tokens > max_tokens:
                # Chunk is too large, reduce size
                left, right = position, end_position
                while left < right:
                    mid = (left + right) // 2
                    test_chunk = text[position:mid]
                    test_tokens = self.num_tokens_from_string(test_chunk)

                    if test_tokens <= max_tokens:
                        left = mid + 1
                    else:
                        right = mid

                # Find the largest chunk that fits
                chunk = text[position : left - 1]
                end_position = left - 1

            chunks.append(chunk)

            # Move position for next chunk, accounting for overlap
            if end_position >= text_length:
                break

            # Calculate how many tokens to backtrack for overlap
            overlap_chars = min(len(chunk), int(overlap_tokens * 4))  # Rough character estimate
            position = end_position - overlap_chars

        return chunks


class MemoryAgent:
    """Base class for memory agents with common functionality."""

    def __init__(self, llm: BaseChatModel, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt

    def invoke(self, content: str) -> str:
        """Invoke the agent with content and return the response."""
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=content)]

        response = self.llm.invoke(messages)
        return response.content


class RetrieverAgent(MemoryAgent):
    """Agent responsible for searching the memory database."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, retriever_instructions)

    def parse_response(self, response: str) -> list[SearchAction]:
        """Parse the response from the agent to extract search actions."""
        try:
            # Extract JSON from response
            json_content = self._extract_json(response)
            actions = []

            for item in json_content:
                # Create a SearchAction from each item
                actions.append(
                    SearchAction(
                        action="search", query=item.get("query"), limit=item.get("limit", 10)
                    )
                )

            # If no valid actions were extracted, create a default one
            if not actions:
                print("No valid search actions found, creating default search action")
                return [SearchAction(action="search", query="", limit=10)]

            return actions
        except Exception as e:
            print(f"Error parsing retriever response: {e}")
            # Return a default search action if parsing fails
            return [SearchAction(action="search", query="", limit=10)]

    def _extract_json(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON content from response text."""
        try:
            # Try multiple extraction patterns

            # Pattern 1: Find content between triple backticks and json
            if "```json" in text:
                import re

                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
                if json_match:
                    json_str = json_match.group(1).strip()
                    return json.loads(json_str)

            # Pattern 2: Find any JSON array in the response
            import re

            json_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # Pattern 3: Just try to load the entire text as JSON
            try:
                return json.loads(text)
            except Exception:
                pass

            # If no JSON found, raise an exception
            raise ValueError("No valid JSON found in response")
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            print(f"Text response: {text[:100]}...")  # Print part of the response for debugging
            return []


class ManagerAgent(MemoryAgent):
    """Agent responsible for performing CRUD operations on the memory database."""

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, manager_instructions)

    def parse_response(self, response: str) -> list[dict[str, Any]]:
        """Parse the response from the agent to extract memory actions."""
        try:
            # Extract JSON from response
            json_content = self._extract_json(response)
            return json_content
        except Exception as e:
            print(f"Error parsing manager response: {e}")
            return []

    def _extract_json(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON content from response text."""
        try:
            # Try multiple extraction patterns

            # Pattern 1: Find content between triple backticks and json
            if "```json" in text:
                import re

                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
                if json_match:
                    json_str = json_match.group(1).strip()
                    return json.loads(json_str)

            # Pattern 2: Find any JSON array in the response
            import re

            json_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # Pattern 3: Just try to load the entire text as JSON
            try:
                return json.loads(text)
            except Exception:
                pass

            # If no JSON found, return empty list with a warning rather than raising exception
            print("Warning: No valid JSON found in response. Using default empty list.")
            return []
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            print(f"Text response: {text[:100]}...")  # Print part of the response for debugging
            return []


class ContextSaga:
    """Main class that orchestrates the retriever and manager agents."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        db: MemoryDatabase | None = None,
        instructions: str = "",
        purpose: str = "",
        max_tokens_per_chunk: int = 24000,
        token_overlap: int = 100,
        checkpoint_dir: str = "checkpoints",
    ):
        self.llm = llm or self.default_llm()
        self.db = db or MemoryDatabase()
        self.instructions = instructions
        self.purpose = purpose
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.token_overlap = token_overlap
        self.token_counter = TokenCounter()
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize agents
        self.retriever_agent = RetrieverAgent(self.llm)
        self.manager_agent = ManagerAgent(self.llm)

    @staticmethod
    def default_llm():
        """Create a default LLM instance."""
        return ChatOpenAI(model="gpt-4o")

    def process_content(self, content: str) -> dict[str, Any]:
        """Process content through both agents and update the memory database."""
        # Step 1: Use retriever agent to search for relevant memories
        print("Invoking retriever agent...")
        retriever_response = self.retriever_agent.invoke(content)
        print(f"Retriever response received (length: {len(retriever_response)})")

        search_actions = self.retriever_agent.parse_response(retriever_response)

        # Step 2: Execute search actions
        search_results = []
        for action in search_actions:
            # If query is empty, use the content as the query
            query = action.query if action.query else content
            print(
                f"Searching for: {query[:50]}..." if len(query) > 50 else f"Searching for: {query}"
            )

            memories = self.db.search_by_text(query, limit=action.limit)
            for memory, score in memories:
                search_results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "importance": memory.importance,
                        "tags": memory.tags,
                        "score": score,
                        "created_at": memory.created_at.isoformat(),
                        "source": memory.source,
                    }
                )

        print(f"Found {len(search_results)} memories")

        # Step 3: Pass content and search results to manager agent
        manager_content = f"""
Content to process:
{content}

Relevant memories from database ({len(search_results)} results):
{json.dumps(search_results, indent=2)}
"""

        print("Invoking manager agent...")
        manager_response = self.manager_agent.invoke(manager_content)
        print(f"Manager response received (length: {len(manager_response)})")

        memory_actions = self.manager_agent.parse_response(manager_response)
        print(f"Parsed {len(memory_actions)} memory actions")

        # Step 4: Execute memory actions
        results = {"created": [], "updated": [], "deleted": []}

        # If no memory actions were found, create one based on the input content
        if not memory_actions:
            print("No memory actions found, creating a default create action")
            # Create a default memory item from the content
            try:
                memory_item = MemoryItem(
                    content=content[:100],
                    source="auto-generated",
                    tags=["auto-generated"],
                    importance=5.0,
                    expires_in=timedelta(hours=24),
                    metadata={"auto_generated": True},
                )

                # Add to database
                memory_id = self.db.add_memory(memory_item)
                results["created"].append(memory_id)
                print(f"Created default memory item with ID: {memory_id}")
            except Exception as e:
                print(f"Error creating default memory: {e}")

        # Process memory actions with tqdm progress bar
        print("Applying memory actions to database...")
        for action_data in tqdm(memory_actions, desc="Processing memory actions"):
            action_type = action_data.get("action")

            if action_type == "create":
                try:
                    item_data = action_data.get("item", {})

                    # Set default values for missing fields
                    content = item_data.get("content", "")
                    if not content:
                        print("Warning: Empty content in create action, skipping")
                        continue

                    # Convert expires_in string to timedelta with better error handling
                    expires_in_str = item_data.get("expires_in", "24:00:00")
                    try:
                        hours, minutes, seconds = map(int, expires_in_str.split(":"))
                        expires_in = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                    except (ValueError, TypeError):
                        print(f"Invalid expires_in format: {expires_in_str}, using default")
                        expires_in = timedelta(hours=24)

                    # Create memory item
                    memory_item = MemoryItem(
                        content=content,
                        source=item_data.get("source", "system"),
                        tags=item_data.get("tags", []),
                        importance=float(item_data.get("importance", 5.0)),
                        expires_in=expires_in,
                        metadata=item_data.get("metadata", {}),
                    )

                    # Add to database
                    memory_id = self.db.add_memory(memory_item)
                    results["created"].append(memory_id)
                except Exception as e:
                    print(f"Error creating memory: {e}")

            elif action_type == "update":
                try:
                    memory_id = action_data.get("id")
                    item_data = action_data.get("item", {})

                    # Get existing memory
                    memory = self.db.get_memory(memory_id)
                    if memory:
                        # Update fields
                        for key, value in item_data.items():
                            if hasattr(memory, key):
                                setattr(memory, key, value)

                        # Update in database
                        success = self.db.update_memory(memory)
                        if success:
                            results["updated"].append(memory_id)
                except Exception as e:
                    print(f"Error updating memory: {e}")

            elif action_type == "delete":
                try:
                    memory_id = action_data.get("id")
                    success = self.db.delete_memory(memory_id)
                    if success:
                        results["deleted"].append(memory_id)
                except Exception as e:
                    print(f"Error deleting memory: {e}")

        return {"search_results": search_results, "actions": memory_actions, "results": results}

    def process_large_content(self, content: str) -> dict[str, Any]:
        """Process large content by breaking it into chunks and processing each chunk."""
        # Check if content exceeds max token limit
        token_count = self.token_counter.num_tokens_from_string(content)
        print(f"Token count: {token_count}")

        if token_count <= self.max_tokens_per_chunk:
            # If content is small enough, process normally
            print(f"Processing content with {token_count} tokens in a single chunk")
            return self.process_content(content)

        # Split content into chunks
        print(f"Content has {token_count} tokens, splitting into chunks...")
        chunks = self.token_counter.split_text_by_tokens(
            content, max_tokens=self.max_tokens_per_chunk, overlap_tokens=self.token_overlap
        )
        print(f"Split content into {len(chunks)} chunks")

        # Process each chunk
        overall_results = {
            "search_results": [],
            "actions": [],
            "results": {"created": [], "updated": [], "deleted": []},
        }

        for i, chunk in enumerate(chunks):
            print(
                f"\nProcessing chunk {i + 1}/{len(chunks)} with "
                f"{self.token_counter.num_tokens_from_string(chunk)} tokens"
            )
            chunk_results = self.process_content(chunk)

            # Combine results
            overall_results["search_results"].extend(chunk_results["search_results"])
            overall_results["actions"].extend(chunk_results["actions"])
            overall_results["results"]["created"].extend(chunk_results["results"]["created"])
            overall_results["results"]["updated"].extend(chunk_results["results"]["updated"])
            overall_results["results"]["deleted"].extend(chunk_results["results"]["deleted"])

        # Deduplicate search results by ID
        seen_ids = set()
        deduplicated_results = []

        for result in overall_results["search_results"]:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                deduplicated_results.append(result)

        overall_results["search_results"] = deduplicated_results

        print("\nFinished processing all chunks")
        print(f"Total search results: {len(overall_results['search_results'])}")
        print(f"Total actions: {len(overall_results['actions'])}")
        print(f"Total created: {len(overall_results['results']['created'])}")
        print(f"Total updated: {len(overall_results['results']['updated'])}")
        print(f"Total deleted: {len(overall_results['results']['deleted'])}")

        return overall_results

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash of the file content for checkpointing."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_checkpoint_path(self, file_path: str) -> str:
        """Get the path to the checkpoint file."""
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        return os.path.join(self.checkpoint_dir, f"{file_name}_{file_hash}_checkpoint.pkl")

    def _save_checkpoint(
        self, checkpoint_path: str, chunk_index: int, overall_results: dict[str, Any]
    ) -> None:
        """Save a checkpoint with the current processing state."""
        checkpoint_data = {"chunk_index": chunk_index, "overall_results": overall_results}

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print(f"Checkpoint saved after chunk {chunk_index}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_path: str) -> dict[str, Any] | None:
        """Load a checkpoint if it exists."""
        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            print(
                f"Checkpoint loaded: processed {checkpoint_data['chunk_index'] + 1}"
                " chunks previously"
            )
            return checkpoint_data
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def process_file(self, file_path: str, resume: bool = True) -> dict[str, Any]:
        """
        Process a file by reading its content and processing it in chunks, with resume capability.
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return {
                    "search_results": [],
                    "actions": [],
                    "results": {"created": [], "updated": [], "deleted": []},
                }

            # Get checkpoint path
            checkpoint_path = self._get_checkpoint_path(file_path)

            # Try to load checkpoint if resume is enabled
            checkpoint_data = None
            if resume:
                checkpoint_data = self._load_checkpoint(checkpoint_path)

            # If no checkpoint or resume disabled, start from the beginning
            if checkpoint_data is None:
                print(f"Reading file: {file_path}")
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Process the content with checkpointing
                return self.process_large_content_with_checkpoints(
                    content, file_path, checkpoint_path
                )
            else:
                # Resume from checkpoint
                return self.resume_processing_from_checkpoint(
                    file_path, checkpoint_data, checkpoint_path
                )

        except Exception as e:
            print(f"Error processing file: {e}")
            return {
                "search_results": [],
                "actions": [],
                "results": {"created": [], "updated": [], "deleted": []},
            }

    def resume_processing_from_checkpoint(
        self, file_path: str, checkpoint_data: dict[str, Any], checkpoint_path: str
    ) -> dict[str, Any]:
        """Resume processing a file from a checkpoint."""
        try:
            # Extract data from checkpoint
            chunk_index = checkpoint_data["chunk_index"]
            overall_results = checkpoint_data["overall_results"]

            # Read the file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Split content into chunks
            token_count = self.token_counter.num_tokens_from_string(content)
            print(f"Token count: {token_count}")

            chunks = self.token_counter.split_text_by_tokens(
                content, max_tokens=self.max_tokens_per_chunk, overlap_tokens=self.token_overlap
            )
            print(f"Split content into {len(chunks)} chunks")
            print(f"Resuming from chunk {chunk_index + 1}/{len(chunks)}")

            # Process remaining chunks with tqdm progress
            remaining_chunks = chunks[chunk_index + 1 :]
            for i, chunk in enumerate(tqdm(remaining_chunks, desc="Processing remaining chunks")):
                actual_index = chunk_index + 1 + i
                print(
                    f"\nProcessing chunk {actual_index + 1}/{len(chunks)} with "
                    f"{self.token_counter.num_tokens_from_string(chunk)} tokens"
                )

                try:
                    chunk_results = self.process_content(chunk)

                    # Combine results
                    overall_results["search_results"].extend(chunk_results["search_results"])
                    overall_results["actions"].extend(chunk_results["actions"])
                    overall_results["results"]["created"].extend(
                        chunk_results["results"]["created"]
                    )
                    overall_results["results"]["updated"].extend(
                        chunk_results["results"]["updated"]
                    )
                    overall_results["results"]["deleted"].extend(
                        chunk_results["results"]["deleted"]
                    )

                    # Save checkpoint after each chunk
                    self._save_checkpoint(checkpoint_path, actual_index, overall_results)

                except Exception as e:
                    print(f"Error processing chunk {actual_index + 1}: {e}")

            # Deduplicate search results by ID
            seen_ids = set()
            deduplicated_results = []

            for result in overall_results["search_results"]:
                if result["id"] not in seen_ids:
                    seen_ids.add(result["id"])
                    deduplicated_results.append(result)

            overall_results["search_results"] = deduplicated_results

            print("\nFinished processing all chunks")
            print(f"Total search results: {len(overall_results['search_results'])}")
            print(f"Total actions: {len(overall_results['actions'])}")
            print(f"Total created: {len(overall_results['results']['created'])}")
            print(f"Total updated: {len(overall_results['results']['updated'])}")
            print(f"Total deleted: {len(overall_results['results']['deleted'])}")

            # Remove checkpoint after successful completion
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print("Processing completed successfully, checkpoint removed")

            return overall_results

        except Exception as e:
            print(f"Error resuming from checkpoint: {e}")
            return checkpoint_data["overall_results"]  # Return whatever we have so far

    def process_large_content_with_checkpoints(
        self, content: str, file_path: str, checkpoint_path: str
    ) -> dict[str, Any]:
        """Process large content with checkpointing capability."""
        # Check if content exceeds max token limit
        token_count = self.token_counter.num_tokens_from_string(content)
        print(f"Token count: {token_count}")

        if token_count <= self.max_tokens_per_chunk:
            # If content is small enough, process normally
            print(f"Processing content with {token_count} tokens in a single chunk")
            return self.process_content(content)

        # Split content into chunks
        print(f"Content has {token_count} tokens, splitting into chunks...")
        chunks = self.token_counter.split_text_by_tokens(
            content, max_tokens=self.max_tokens_per_chunk, overlap_tokens=self.token_overlap
        )
        print(f"Split content into {len(chunks)} chunks")

        # Process each chunk with checkpointing
        overall_results = {
            "search_results": [],
            "actions": [],
            "results": {"created": [], "updated": [], "deleted": []},
        }

        # Use tqdm to create a progress bar for chunks
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            print(
                f"\nProcessing chunk {i + 1}/{len(chunks)} with "
                f"{self.token_counter.num_tokens_from_string(chunk)} tokens"
            )

            try:
                chunk_results = self.process_content(chunk)

                # Combine results
                overall_results["search_results"].extend(chunk_results["search_results"])
                overall_results["actions"].extend(chunk_results["actions"])
                overall_results["results"]["created"].extend(chunk_results["results"]["created"])
                overall_results["results"]["updated"].extend(chunk_results["results"]["updated"])
                overall_results["results"]["deleted"].extend(chunk_results["results"]["deleted"])

                # Save checkpoint after each chunk
                self._save_checkpoint(checkpoint_path, i, overall_results)

            except Exception as e:
                print(f"Error processing chunk {i + 1}: {e}")

        # Deduplicate search results by ID
        seen_ids = set()
        deduplicated_results = []

        for result in overall_results["search_results"]:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                deduplicated_results.append(result)

        overall_results["search_results"] = deduplicated_results

        print("\nFinished processing all chunks")
        print(f"Total search results: {len(overall_results['search_results'])}")
        print(f"Total actions: {len(overall_results['actions'])}")
        print(f"Total created: {len(overall_results['results']['created'])}")
        print(f"Total updated: {len(overall_results['results']['updated'])}")
        print(f"Total deleted: {len(overall_results['results']['deleted'])}")

        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Processing completed successfully, checkpoint removed")

        return overall_results
