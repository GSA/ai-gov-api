from fastapi import APIRouter, Depends
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.schemas import UserCreate, UserOut, UserUpdate
from app.users.repositories import UserRepository
from app.auth.dependencies import RequiresScope
from app.auth.schemas import Scope
from app.db.session import get_db_session



router = APIRouter()

@router.post("/create")
async def create_user(
    user: UserCreate, 
    api_key=Depends(RequiresScope([Scope.ADMIN])),
    session: AsyncSession = Depends(get_db_session)
) -> UserOut:
    user_repo = UserRepository(session)
    async with session.begin():
        new_user = await user_repo.create(user)
    
    await session.refresh(new_user) 
    return UserOut.model_validate(new_user)
        
@router.post("/update/{email}")
async def converse(
    update: UserUpdate,
    email:EmailStr,
    api_key=Depends(RequiresScope([Scope.ADMIN])),
    session: AsyncSession = Depends(get_db_session),
) -> UserOut:
    user_repo = UserRepository(session)
    async with session.begin():
        updated_user_orm = await user_repo.update(email=email, user=update)
    await session.refresh(updated_user_orm) 
    return UserOut.model_validate(updated_user_orm)
        
