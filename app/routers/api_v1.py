from typing import List
from fastapi import APIRouter, Depends

from app.auth.dependencies import RequiresScope, valid_api_key
from app.auth.schemas import Scope
from app.backends.base import LLMModel
from app.backends.dependencies import Backend
from app.config.settings import get_settings
from app.schema.open_ai import ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest

router = APIRouter()

@router.get("/models")
async def models(
    settings=Depends(get_settings),
    api_key=Depends(valid_api_key)
) -> List[LLMModel]:
    return  [backend for _, backend in settings.backend_map.values()]


@router.post("/chat")
async def converse(
    req: ChatCompletionRequest, 
    api_key=Depends(RequiresScope([Scope.MODELS_INFERENCE])),
    backend=Depends(Backend('chat'))
) -> ChatCompletionResponse:
    return await backend.invoke_model(req)

@router.post("/embeddings")
async def embeddings(
    req: EmbeddingRequest,
    api_key=Depends(RequiresScope([Scope.MODELS_EMBEDDING])),
    backend=Depends(Backend('embedding'))
):
    return  await backend.embeddings(req)
