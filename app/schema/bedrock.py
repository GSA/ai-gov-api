from pydantic import BaseModel, Field, NonNegativeInt
from typing import List, Literal, Optional, Union

from app.config.settings import available_models

# Types that model Bedrock's converse API.
# These won't be exposed to the users, but will be 
# used when converting to formats approriate for
# working with Bedrocks

# Document Block
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html

class ContentDocument(BaseModel):
    format: Literal['pdf', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'txt', 'md']
    name: str
    source: bytes

class ContentText(BaseModel):
    text: str


ContentBlock = Union[ContentText, ContentDocument]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: List[ContentBlock]


class InferenceConfig(BaseModel):
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, ge=0, le=1, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Top-p sampling")
    stop_sequences: Optional[List[str]] = Field(None, description="Optional list of stop sequences")


class ConverseRequest(BaseModel):
    # TODO: pull this as an enum from settings
    model_id: available_models = Field(..., exclude=True)
    messages: List[Message]
    inferenceConfig: Optional[InferenceConfig] = None


class ConverseResponseUsage(BaseModel):
    inputTokens: NonNegativeInt
    outputTokens: NonNegativeInt
    totalTokens: NonNegativeInt

class ConverseResponseOutput(BaseModel):
    role: Literal["assistant"]
    content: List[ContentText]

class ConverseResponse(BaseModel):
        output: dict[Literal["message"], ConverseResponseOutput]
        usage: ConverseResponseUsage


# TODO: validate text lengths? 0 characters - 2048 characters
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html

class CohereRequest(BaseModel):
    texts: List[str] = Field(..., min_length=0, max_length=96)
    input_type: Literal["search_document", "search_query", "classification", "clustering"]
    truncate: Literal["NONE", "START", "END"] = None
    embedding_types: List[Literal["float", "int8", "uint8", "binary", "ubinary"]] = None
