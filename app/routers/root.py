from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
import structlog

from app.db.session import async_session


log = structlog.get_logger()

router = APIRouter()


@router.get("/health")
async def healthcheck():
    async with async_session() as session:
        try:
            result = await session.execute(text('SELECT 1'))
            result.scalar_one
        except NoResultFound as e:
            log.error(f"Database health check failed: SELECT 1 returned no result. {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Database health check failed: Core query returned no result. {str(e)}"
            )
        except SQLAlchemyError as e: 
            log.error(f"Database health check failed: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Database connection error: {str(e)}"
            )
        except Exception as e: #
            log.error(f"An unexpected error occurred during health check: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {str(e)}"
            )
        return {"status": True}