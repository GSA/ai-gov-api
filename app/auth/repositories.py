from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.models import APIKey
from app.auth.schemas import APIKeyCreate
import hashlib

class APIKeyRepository:
    def __init__(self, session:AsyncSession):
        self.session = session

    async def create(self, token: APIKeyCreate) -> APIKey:
        new_api_key = APIKey(**token.model_dump())

        self.session.add(new_api_key)
        return new_api_key
            
    async def get_by_api_key_value(self, provided_key: str) -> APIKey | None:
        '''Api keys are not stored. Given an API key, first get it's hash and use that for the query'''
        hashed_key = hashlib.sha256(provided_key.encode('utf-8')).hexdigest()

        result = await self.session.execute(
            select(APIKey).where(APIKey.hashed_key == hashed_key)
        )
        return result.scalars().one_or_none()
    
    async def get_by_id_with_manager(self, api_key_id: int) -> APIKey | None:
        """Fetches an APIKey and eagerly loads its manager (User)."""
        stmt = (
            select(APIKey)
            .where(APIKey.id == api_key_id)
            .options(
                joinedload(APIKey.manager)
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()
 
