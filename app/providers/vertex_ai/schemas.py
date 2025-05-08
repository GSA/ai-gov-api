# Google Vertex provides typed python objects
# which we can use until it introduces issues

from typing import List, Optional
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingInput

class EmbedRequest(BaseModel):
    texts: List[str | TextEmbeddingInput]
    auto_truncate: bool = True
    output_dimensionality: Optional[int] = None
