from datetime import datetime
from fastapi import HTTPException, status, Depends, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.db.session import get_db_session
from app.schema.api_key import APIKeyOut
from app.auth.scopes import Scope
from app.repositories.api_keys import APIKeyRepository
from pydantic import UUID4

logger = structlog.get_logger()

api_key_header:UUID4 = APIKeyHeader(name="X-API-Key")


async def valid_api_key(
    api_key_header:str=Security(api_key_header), 
    session: AsyncSession=Depends(get_db_session)
    ) -> APIKeyOut:
    '''
    Auth dependency injection that looks for the API key and returns the user.
    If the API key does not exist, raises 401.
    '''
    api_key = await APIKeyRepository(session).get_by_api_key_value(api_key_header)

    if api_key is None or not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key"
        )
    if api_key.expires_at is not None:
        now = datetime.now()
        if api_key.expires_at <= now:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is expired"
            ) 

    return api_key
        

class RequiresScope:
    def __init__(self, scopes: list[Scope]):
        self.scopes = set(scopes)

    def __call__(
        self,
        api_key: APIKeyOut = Depends(valid_api_key)
        ) -> APIKeyOut:

        if not self.scopes:
            return api_key
        
        if self.scopes.issubset(set(api_key.scopes)):
            return api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not Authorized"
        ) 
