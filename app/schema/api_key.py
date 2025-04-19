from pydantic import BaseModel, ConfigDict, UUID4 
from datetime import datetime

from app.auth.scopes import Scope

class APIKeyBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    manager_id: UUID4
    scopes: list[Scope] = []
    key_value:UUID4
    is_active: bool = True
    expires_at: datetime | None = None

class APIKeyCreate(APIKeyBase):
    pass

class APIKeyUpdate(APIKeyBase):
    pass

class APIKeyOut(APIKeyBase):
    id: int 
    created_at: datetime
    updated_at: datetime