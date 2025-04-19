from pydantic import BaseModel, ConfigDict, UUID4, EmailStr
from datetime import datetime

from app.auth.roles import Role

class UserBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    email: EmailStr
    role: Role
    is_active: bool = True

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    pass

class UserOut(UserBase):
    id: UUID4 
    created_at: datetime
    updated_at: datetime