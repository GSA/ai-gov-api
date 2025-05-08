from datetime import datetime
from typing import List
from ..core.chat_schema import ChatRepsonse, CompletionUsage, Response
from ..core.embed_schema import EmbeddingResponse, EmbeddingData, EmbeddingUsage
from vertexai.generative_models import GenerationResponse
from vertexai.language_models import TextEmbedding


def convert_chat_vertex_response(resp: GenerationResponse, model:str) -> ChatRepsonse:
        usage = CompletionUsage(
            prompt_tokens=resp.usage_metadata.prompt_token_count,
            completion_tokens=resp.usage_metadata.candidates_token_count,
            total_tokens=resp.usage_metadata.total_token_count, 
        )   
        choices = [Response(content=candidate.content.parts[0].text) for i, candidate in enumerate(resp.candidates)]
        
        return ChatRepsonse(
            created=datetime.now(),
            model=model,
            choices=choices,
            usage=usage
        )

def vertex_embed_reposonse_to_core(embeddings: List[TextEmbedding], model:str) -> EmbeddingResponse:
    token_count = sum(int(emb.statistics.token_count) for emb in embeddings if emb.statistics)
    usage = EmbeddingUsage(
        prompt_tokens=token_count,
        total_tokens=token_count
    )
    return EmbeddingResponse(
        model=model,
        data=[EmbeddingData(index=idx, embedding=data.values) for idx, data in enumerate(embeddings)],
        usage=usage
    )