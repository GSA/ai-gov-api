from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.models import APIKey
from app.auth.schemas import APIKeyCreate, APIKeyOut
import hashlib

class APIKeyRepository:
    def __init__(self, session:AsyncSession):
        self.session = session

    async def create(self, token: APIKeyCreate) -> APIKeyOut:
        new_api_key = APIKey(**token.model_dump())

        async with self.session.begin():
            self.session.add(new_api_key)
            await self.session.commit()
            
        await self.session.refresh(new_api_key)
        return APIKeyOut.model_validate(new_api_key)

    async def get_by_api_key_value(self, provided_key: str) -> APIKeyOut | None:
        '''Api keys are not stored. Given an API key, first get it's hash and use that for the query'''
        hashed_key = hashlib.sha256(provided_key.encode('utf-8')).hexdigest()

        async with self.session.begin():
            result = await self.session.execute(
                select(APIKey).where(APIKey.hashed_key == hashed_key)
            )
            result = result.scalars().first()
            if result is not None:
                return APIKeyOut.model_validate(result)

