from typing import List, Literal, Optional
from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]
    encoding_format: Literal['float']
    dimensions: Optional[int] = None
    input_type: Optional[Literal[
        "search_document",       
        "search_query",          
        "classification",
        "clustering",
        "semantic_similarity",
    ]] = None


class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data:List[EmbeddingData]
    model:str
    usage: EmbeddingUsage