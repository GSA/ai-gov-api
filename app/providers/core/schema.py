'''
This is an intermediate format representing an abstract chat request.
It's purpose it to insulate from changes to the OpenAI API and prevent
re-writing conversion code.

OpenAI requests will be converted into this form and wll concrete representations
should convert from this.

OpenAI -> IR -> |-> Bedrock
                |-> Vertex
                | -> Others
'''
from typing import List, Literal, Annotated, Optional
from pydantic import BaseModel, Field


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str

class ImagePart(BaseModel):
    kind: Literal["image"] = "image"
    bytes_: bytes = Field(..., alias="bytes")

class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    mime_type: str
    bytes_: bytes

ContentPart = Annotated[TextPart | ImagePart | FilePart, Field(discriminator="kind")]

class Message(BaseModel):
    role: Literal["user", "assistant", "system"] 
    parts: List[ContentPart]


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
