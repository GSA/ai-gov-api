from pydantic import BaseModel, Field, confloat, ConfigDict, NonNegativeInt
from typing import Literal, Optional, Union, List
from datetime import datetime

"""
The api's chat interface is modeled after the OpenAI Chat Completion API:
https://platform.openai.com/docs/api-reference/chat

This schema expresses this API as Pydantic classes and should be considered the source of truth
of which parts of the Chap Completion API we support. 

Other backends will depend on this schema to convert to other backend formats.
"""


class ImageUrl(BaseModel):
    """
    Defines the structure for an image URL input.
    To simplify egress concerns, we don't supprt HTTPS urls at the moment.
    """
    model_config = ConfigDict(extra="ignore")

    url: str = Field(..., description="The base64 encoded image data URI.")
    detail: Optional[Literal["auto", "low", "high"]] = Field("auto", description="Specifies the detail level of the image.")

class TextContentPart(BaseModel):
    """Represents a text part in a multimodal content list."""
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"] = "text"
    text: str

class ImageContentPart(BaseModel):
    """Represents an image part in a multimodal content list."""
    model_config = ConfigDict(extra="ignore")

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl

ContentPart = Union[TextContentPart, ImageContentPart]

class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant","system", "function", "tool"]
    content: Union[str, List[ContentPart], None] = Field(description="The content of the message. Can be a string, a list of content parts (for multimodal input), or None (this is intended for tool calls).")
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    #model_config = ConfigDict(extra="ignore") 
    model: str = Field(..., description="The model to use for chat completion")
    messages: List[ChatCompletionMessage] = Field(..., description="A list of messages from the conversation so far")
    
    temperature: Optional[confloat(ge=0, le=2)] = Field(
        default=None,
        description="What sampling temperature: between 0 and 2"
    )
    
    top_p: Optional[confloat(ge=0, le=1)] = Field(
        default=None,
        description="An alternative to sampling with temperature, called nucleus sampling"
    )
    
    n: Optional[int] = Field(
        default=None,
        description="How many chat completion choices to generate for each input message"
    )
    
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent"
    )
    
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate"
    )
    
    presence_penalty: Optional[confloat(ge=-2.0, le=2.0)] = Field(
        default=0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far"
    )
    
    frequency_penalty: Optional[confloat(ge=-2.0, le=2.0)] = Field(
        default=0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far"
    )
    
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user"
    )

class ChatCompletionUsage(BaseModel):
    """Report of token use for a particular call"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponseMessage(BaseModel):
    """The LLM repsonse"""
    role: Literal["assistant"]
    content: str

class ChatCompletionChoice(BaseModel):
    index: NonNegativeInt
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop"]


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    object: Literal["chat.completion"]
    created: datetime
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


### Embedding Requests Models ###
# This is not used at the moment — it's not clear how to convert to Bedrock's Cohere model

class EmbeddingRequest(BaseModel):
    """
    Represents the request payload for the OpenAI Embeddings API.
    Reference: https://platform.openai.com/docs/api-reference/embeddings/create
    """
    input: Union[str, List[str]] = Field(
        ..., 
        description="Input text to embed, encoded as a string or array of strings. Each input must not exceed the max input tokens for the model."
    )
    model: str = Field(
        ...,
        description="ID of the model to use (e.g., 'text-embedding-3-small', 'text-embedding-ada-002')."
    )
    encoding_format: Optional[Literal['float', 'base64']] = Field(
        default='float',
        alias="encodingFormat", # Uhg, the API's camelCase parameter
        description="The format to return the embeddings in. Can be 'float' or 'base64'."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="The number of dimensions the resulting output embeddings should have. Only supported in some models."
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user, which can help OpenAI monitor and detect abuse."
    )

    model_config = ConfigDict(
        populate_by_name=True, 
        extra='ignore' 
    )
  

# --- Embedding Response Models ---

class EmbeddingData(BaseModel):
    """
    A single embedding object within the response data array.
    """
    object: Literal["embedding"] = Field(..., description="The object type, always 'embedding'.")
    embedding: Union[List[float], str] = Field(..., description="The embedding vector, which is a list of floats or a base64 string depending on 'encoding_format'.")
    index: int = Field(..., description="The index of the embedding in the list, corresponding to the input index.")

    model_config = ConfigDict(
        extra='ignore'
    )

class EmbeddingUsage(BaseModel):
    """
    Represents the token usage information for the embedding request.
    """
    prompt_tokens: int = Field(..., alias="promptTokens", description="The number of tokens in the prompt.")
    total_tokens: int = Field(..., alias="totalTokens", description="The total number of tokens used in the request (prompt + completion).")


    model_config = ConfigDict(
        populate_by_name=True, # Allows using aliases during instantiation from JSON
        extra='ignore'
    )

class EmbeddingResponse(BaseModel):
    """
    Represents the successful response payload from the OpenAI Embeddings API.
    See: https://platform.openai.com/docs/api-reference/embeddings/create
    """
    object: Literal["list"] = Field(..., description="The object type, typically 'list'.")
    data: List[EmbeddingData] = Field(..., description="A list of embedding objects, one for each input.")
    model: str = Field(..., description="The ID of the model used for generating embeddings.")
    usage: EmbeddingUsage = Field(..., description="Usage statistics for the request.")


    model_config = ConfigDict(
        populate_by_name=True, # Needed for nested aliases (like EmbeddingUsage)
        extra='ignore'
    )
   

# Bedrock does not have a unified embedding API and the Cohere request format
# has input_type, which is not part of the OpenAI api.
# This means at the moment we don't have a canonical interface for embeddings.
# For now, just use cohere
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html

class CohereRequest(BaseModel):
    model: str = Field(..., exclude=True)
    texts: List[str] = Field(..., min_length=0, max_length=96)
    input_type: Literal["search_document", "search_query", "classification", "clustering"]
    truncate: Literal["NONE", "START", "END"] = None
    embedding_types: List[Literal["float", "int8", "uint8", "binary", "ubinary"]] = None
