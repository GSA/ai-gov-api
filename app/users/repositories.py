from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.models import User
from app.users.schemas import UserCreate

class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        
    async def create(self,user:UserCreate) -> User:
        new_user = User(**user.model_dump())
        async with self.session.begin():
            self.session.add(new_user)           
        await self.session.refresh(new_user)
        return new_user

    async def get(self, user_id: str) -> User:
        async with self.session.begin():
            result = await self.session.execute(
                select(User).where(User.id==user_id)
            )
        return result.scalars().first()