import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    source_authority_order: list[str] | None = Field(
        default=None,
        description=(
            "Most-to-least-authoritative source labels for this collection "
            "— e.g. ['Annual Report', 'Press Release']. Not "
            "hardcoded anywhere as universally correct; omit to use the "
            "built-in default order."
        ),
    )


class CollectionResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    source_authority_config: dict[str, object]
    created_at: datetime

    model_config = {"from_attributes": True}
