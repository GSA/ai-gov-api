from functools import singledispatch
from typing import List

from vertexai.language_models import TextEmbeddingInput
from vertexai.generative_models import Part, Content
from ..core.chat_schema import (
    Message,
    SystemMessage,
    TextPart,
    ImagePart,
    FilePart,
)
from ..core.embed_schema import EmbedRequest as CoreEmbedRequest
from ..vertex_ai.schemas import EmbedRequest

@singledispatch
def _part_to_vtx(part) -> Part:
    raise TypeError(f"No converter for {type(part)}")

@_part_to_vtx.register
def _(part: TextPart) -> Part:
    return Part.from_text(part.text)

@_part_to_vtx.register
def _(part: ImagePart) -> Part:
    return Part.from_data(data= part.bytes_, mime_type=f"image/{part.file_type}")

@_part_to_vtx.register
def _(part: FilePart) -> Part:
    return Part.from_data(data=part.bytes_, mime_type="application/pdf")
    

def convert_core_messages(messages: list[Message]) -> List[Content]:
    vertex_history:List[Content] = []

    for idx, message in enumerate(messages):
        vertex_role = "model" if message.role == "assistant" else message.role
  
        if isinstance(message, SystemMessage):
            # vertex doesn't have system messages they recommend using user/assistant pairs
            system_message = '\n'.join(content.text for content in message.content)
            vertex_history.append(Content(role="user", parts=[Part.from_text(system_message)]))
            vertex_history.append(Content(role="model", parts=[Part.from_text("Okay, I will follow these instructions.")]))

            continue
        # Multimodal content list
        for part in message.content:
            vertex_history.append(Content(role=vertex_role,parts=[_part_to_vtx(part)]))

    return vertex_history


def convert_embedding_request(req: CoreEmbedRequest) -> EmbedRequest:

    type_map = {
        "search_document": "RETRIEVAL_DOCUMENT",
        "search_query": "RETRIEVAL_QUERY",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
        "semantic_similarity": "SEMANTIC_SIMILARITY",
    }

    input_type = type_map.get(req.input_type) if req.input_type is not None else None

    return EmbedRequest(
        auto_truncate =True,
        output_dimensionality=req.dimensions,
        texts = [
            TextEmbeddingInput(text=text, task_type=input_type) 
            for text in req.input
        ]
    )

    
    