import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """
    Fundamental unit of memory representing a single piece of information.

    MemoryItem is used to store discrete pieces of information with associated metadata,
    expiration time, and importance ranking. These items form the building blocks of
    both short-term and working memory structures.
    """

    id: int = Field(
        default_factory=lambda: uuid.uuid4().int,
        description="The unique identifier for the memory item (auto generated)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the memory item was created",
    )
    updated_at: datetime | None = Field(
        default=None, description="Timestamp when the memory item was last updated"
    )
    expires_in: timedelta = Field(
        description="Time period after which this memory item should expire"
    )
    source: str = Field(description="Origin of the memory (user input, system event, etc.)")
    content: str = Field(description="The actual content/information stored in this memory item")
    tags: list[str] = Field(
        default_factory=list,
        description="Categorical labels to help organize and retrieve memory items",
    )
    importance: float = Field(
        ge=0.0,
        le=10.0,
        default=5.0,
        description="Ranking of importance (0-10) affecting retention and retrieval priority",
    )
    metadata: dict | None = Field(
        default_factory=dict,
        description="Additional structured data associated with this memory item",
    )

    # internal
    deleted_at: datetime | None = Field(
        default=None,
        exclude=True,
        description="Timestamp when this item was marked as deleted (for soft deletion)",
    )


class CreateAction(BaseModel):
    action: Literal["create"] = "create"
    item: dict = Field(..., description="The memory item to create")


class UpdateAction(BaseModel):
    action: Literal["update"] = "update"
    id: int
    item: dict = Field(..., description="The memory item to update using dict.update()")


class DeleteAction(BaseModel):
    action: Literal["delete"] = "delete"
    id: int = Field(..., description="The id of the memory item to delete")


class SearchAction(BaseModel):
    action: Literal["search"] = "search"
    query: str = Field(..., description="The query to search for")
    limit: int = Field(..., description="The maximum number of results to return")


if __name__ == "__main__":
    ...
