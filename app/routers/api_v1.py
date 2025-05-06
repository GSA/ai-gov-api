from typing import List
from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import RequiresScope, valid_api_key
from app.auth.schemas import Scope
from app.providers.base import LLMModel
from app.providers.dependencies import Backend
from app.providers.exceptions import InputDataError
from app.config.settings import get_settings
from app.schema.open_ai import ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest


router = APIRouter()

@router.get("/models")
async def models(
    settings=Depends(get_settings),
    api_key=Depends(valid_api_key)
) -> List[LLMModel]:
    return  [backend for _, backend in settings.backend_map.values()]


@router.post("/chat/completions")
async def converse(
    req: ChatCompletionRequest, 
    api_key=Depends(RequiresScope([Scope.MODELS_INFERENCE])),
    backend=Depends(Backend('chat'))
) -> ChatCompletionResponse:
    try:
        return await backend.invoke_model(req)
    except InputDataError as e:
        error_detail = {"error": "Bad Request", "message": str(e)}
        if e.field_name:
            error_detail["field"] = e.field_name
        raise HTTPException(status_code=400, detail=error_detail)



@router.post("/embeddings")
async def embeddings(
    req: EmbeddingRequest,
    api_key=Depends(RequiresScope([Scope.MODELS_EMBEDDING])),
    backend=Depends(Backend('embedding'))
):
    return  await backend.embeddings(req)
