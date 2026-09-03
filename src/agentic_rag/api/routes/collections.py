from fastapi import APIRouter
from sqlalchemy import select

from agentic_rag.api.dependencies.db import DbSession
from agentic_rag.api.schemas.collections import CollectionCreate, CollectionResponse
from agentic_rag.storage.models import Collection

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(body: CollectionCreate, db: DbSession) -> Collection:
    source_authority_config = (
        {"order": body.source_authority_order} if body.source_authority_order else {}
    )
    collection = Collection(
        name=body.name,
        description=body.description,
        source_authority_config=source_authority_config,
    )
    db.add(collection)
    await db.flush()
    await db.commit()
    await db.refresh(collection)
    return collection


@router.get("", response_model=list[CollectionResponse])
async def list_collections(db: DbSession) -> list[Collection]:
    result = await db.scalars(select(Collection).order_by(Collection.created_at.desc()))
    return list(result.all())
