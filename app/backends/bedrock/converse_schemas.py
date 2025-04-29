'''
Types for the Bedrock Converse API
https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html

These are the destination types for conversions from public input appropriate for
passing to Bedrocks's converse() method as well as source types to convert back
to the output of the public API.
'''

from typing import Optional, Union, List, Literal

from pydantic import BaseModel, Field, NonNegativeInt, ConfigDict
from pydantic.alias_generators import to_camel



class ImageSource(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # alias because pydantic does not allow python types as properties
    data: bytes = Field(..., description="Raw image data bytes.", alias="bytes")

class ImagePayload(BaseModel):
    format: Literal['jpeg', 'png', 'gif', 'webp'] = Field(..., description="Image format.")
    source: ImageSource

class ContentImageBlock(BaseModel):
    image: ImagePayload

class SystemContentBlock(BaseModel):
    # Bedrock allows other fields here that we're ignoring for now
    # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemContentBlock.html
    text: str

class ContentTextBlock(BaseModel):
    text: str

class DocumentSource(BaseModel):
    data: bytes = Field(..., description="Raw document data bytes.", alias="bytes")

class DocumentPayload(BaseModel):
    format: Literal['pdf', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'txt', 'md'] = Field(..., description="Document format.")
    name: str = Field(..., description="Name of the document.")
    source: DocumentSource

class ContentDocumentBlock(BaseModel):
    document: DocumentPayload

ContentBlock = Union[ContentTextBlock, ContentImageBlock, ContentDocumentBlock]

# it's not clear how to deal with OpenAI's other possible roles
BedrockMessageRole = Literal["user", "assistant"]

class Message(BaseModel):
    role: BedrockMessageRole
    content: List[ContentBlock]

class InferenceConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel
    )
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=None, ge=0, le=1, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description="Top-p sampling")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Optional list of stop sequences")

class ConverseRequest(BaseModel):
    model_config = ConfigDict(
        # Allows parsing input with either 'inference_config' or 'inferenceConfig'
        populate_by_name=True,
        alias_generator=to_camel
    )
    model_id: str = Field(..., exclude=True)
    messages: List[Message]
    system: Optional[List[SystemContentBlock]] = Field(default=None, description="Optional list of system prompts")
    inference_config: Optional[InferenceConfig] = Field(default=None, serialization_alias="inferenceConfig")

class ConverseResponseUsage(BaseModel):
    inputTokens: NonNegativeInt
    outputTokens: NonNegativeInt
    totalTokens: NonNegativeInt

class ConverseResponseOutput(BaseModel):
    role: Literal["assistant"]
    content: List[ContentTextBlock]

class ConverseResponse(BaseModel):
        output: dict[Literal["message"], ConverseResponseOutput]
        usage: ConverseResponseUsage
