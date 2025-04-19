from fastapi import APIRouter, HTTPException, Depends

from app.auth.dependencies import RequiresScope
from app.auth.scopes import Scope
from app.services.llm_models import invoke_converse_model, get_embeddings
from app.schema.open_ai import ChatCompletionRequest, ChatCompletionResponse
from app.schema.bedrock import  CohereRequest
from app.schema.conversion import convert_open_ai_completion_bedrock, convert_bedrock_response_open_ai


router = APIRouter()

@router.post("/chat")
async def converse(req: ChatCompletionRequest, api_key=Depends(RequiresScope([Scope.MODELS_INFERENCE]))
) -> ChatCompletionResponse:
    bedrock_req = convert_open_ai_completion_bedrock(req)
    output = await invoke_converse_model(bedrock_req)
    return convert_bedrock_response_open_ai(output)

@router.post("/embeddings")
async def embeddings(req: CohereRequest, api_key=Depends(RequiresScope([Scope.MODELS_EMBEDDING]))):
    try:
        output = await get_embeddings(req)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
