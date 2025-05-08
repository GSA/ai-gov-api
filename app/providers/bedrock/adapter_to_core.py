from datetime import datetime

from ..core.chat_schema import ChatRepsonse, CompletionUsage, Response
from ..core.embed_schema import EmbeddingResponse, EmbeddingData, EmbeddingUsage
from .converse_schemas import ConverseResponse
from .cohere_embedding_schemas import CohereRepsonse

def bedrock_chat_response_to_core(resp: ConverseResponse, model:str) -> ChatRepsonse:
    return ChatRepsonse(
        model=model,
        created=datetime.now(),
        choices=[
            Response(content=r.text)
            for r in resp.output['message'].content
        ],
        usage=CompletionUsage(
            prompt_tokens=resp.usage.inputTokens,
            completion_tokens=resp.usage.outputTokens,
            total_tokens=resp.usage.totalTokens
        )
    )

def bedorock_embed_reposonse_to_core(resp: CohereRepsonse, model:str, token_count) -> EmbeddingResponse:
    return EmbeddingResponse(
        model=model,
        data=[EmbeddingData(index=idx, embedding=data) for idx, data in enumerate(resp.embeddings['float'])],
        usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count)
    )