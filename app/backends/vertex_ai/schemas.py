from typing import List
from pydantic import BaseModel

class VertexEmbeddingInput(BaseModel):
    input: List[str]
    