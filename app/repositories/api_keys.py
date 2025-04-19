from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.entities.api_key import APIKey
from app.schema.api_key import APIKeyCreate, APIKeyOut
from pydantic import UUID4

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

    async def get_by_api_key_value(self, key_value: UUID4) -> APIKeyOut | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(APIKey).where(APIKey.key_value == key_value)
            )
            result = result.scalars().first()
            if result is not None:
                return APIKeyOut.model_validate(result)

