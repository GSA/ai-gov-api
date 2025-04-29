from app.schema.open_ai import EmbeddingRequest
from .cohere_embedding_schemas import CohereRequest

def convert_openai_request(req: EmbeddingRequest) -> CohereRequest:
    texts = req.input if isinstance(req.input, list) else [req.input]

    return CohereRequest(
        model=req.model,
        texts=texts,
        input_type=req.input_type,  # type: ignore[arg-type]
        embedding_types=[req.encoding_format]
    )
