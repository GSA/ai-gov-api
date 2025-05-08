from ..core.chat_schema import ChatRepsonse
from ..core.embed_schema import EmbeddingResponse as Core_EmbeddingResponse
from app.providers.open_ai.schemas import ChatCompletionUsage, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionResponseMessage, EmbeddingResponse, EmbeddingData, EmbeddingUsage

def core_chat_response_to_openai(resp: ChatRepsonse) -> ChatCompletionResponse:
    return ChatCompletionResponse(
       model=resp.model,
       created=resp.created,
       choices=[
           ChatCompletionChoice(
               index=idx,
               message=ChatCompletionResponseMessage(content=c.content)
           ) for idx, c in enumerate(resp.choices)
       ],
       usage=ChatCompletionUsage.model_validate(resp.usage, from_attributes=True)
    )

def core_embed_response_to_openai(resp: Core_EmbeddingResponse) -> EmbeddingResponse:
    return EmbeddingResponse(
        model=resp.model,
        data=[EmbeddingData.model_validate(data, from_attributes=True) for data in resp.data],
        usage=EmbeddingUsage.model_validate(resp.usage, from_attributes=True)
    )
