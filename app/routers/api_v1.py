from fastapi import APIRouter, Depends

from app.auth.dependencies import RequiresScope
from app.backends.dependencies import Backend
from app.auth.schemas import Scope
from app.schema.open_ai import ChatCompletionRequest, ChatCompletionResponse
from app.schema.open_ai import CohereRequest

router = APIRouter()


@router.post("/chat")
async def converse(
    req: ChatCompletionRequest, 
    api_key=Depends(RequiresScope([Scope.MODELS_INFERENCE])),
    backend=Depends(Backend('chat'))
) -> ChatCompletionResponse:
    print("api key:", api_key.model_dump())
    return await backend.invoke_model(req)

@router.post("/embeddings")
async def embeddings(
    req: CohereRequest,
    api_key=Depends(RequiresScope([Scope.MODELS_EMBEDDING])),
    backend=Depends(Backend('embedding'))
):
    return  await backend.embeddings(req)
