from pydantic import BaseModel, Field, confloat, ConfigDict, NonNegativeInt
from typing import Literal, Optional, Union
from datetime import datetime
from app.config.settings import available_models


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant","system", "function", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: available_models = Field(..., description="The model to use for chat completion")
    messages: list[ChatCompletionMessage] = Field(..., description="A list of messages comprising the conversation so far")
    
    temperature: Optional[confloat(ge=0, le=2)] = Field(
        default=None,
        description="What sampling temperature to use, between 0 and 2"
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
    
    stop: Optional[Union[str, list[str]]] = Field(
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
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponseMessage(BaseModel):
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
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
