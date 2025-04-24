from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.models import User
from app.users.schemas import UserCreate, UserOut

class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        
    async def create(self,user:UserCreate) -> UserOut:
        new_user = User(**user.model_dump())
        async with self.session.begin():
            self.session.add(new_user)           
        await self.session.refresh(new_user)
        return UserOut.model_validate(new_user)

    async def get(self, user_id: str) -> UserOut | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(User).where(User.id==user_id)
            )
            result = result.scalars().first()
            if result is not None:
                return UserOut.model_validate(result)
    
    async def get_by_email(self, email: str) -> UserOut | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(User).where(User.email==email)
            )
            result = result.scalars().first()

            if result is not None:
                return UserOut.model_validate(result)
