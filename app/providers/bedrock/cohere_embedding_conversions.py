from app.schema.open_ai import EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage
from .cohere_embedding_schemas import CohereRequest

def convert_openai_request(req: EmbeddingRequest) -> CohereRequest:
    texts = req.input if isinstance(req.input, list) else [req.input]

    return CohereRequest(
        model=req.model,
        texts=texts,
        input_type=req.input_type,  # type: ignore[arg-type]
        embedding_types=[req.encoding_format]
    )

def convert_openai_repsonse(res, token_count:int, model_id:str) -> EmbeddingResponse:
    return EmbeddingResponse(
        object="list",
        data = [
            EmbeddingData(
                object = "embedding",
                embedding=emb, 
                index=i
            )
            for i, emb in enumerate(e for nested in res['embeddings'].values() for e in nested)
        ],
        model=model_id,
        usage = EmbeddingUsage(promptTokens=token_count, totalTokens=token_count) 
    )
