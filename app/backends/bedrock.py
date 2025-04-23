'''
backends.bedrock
~~~~~~~~~~~~~~~~

This module is responsible for accepting requests in the form of OpenAI Chat Completions, 
interacting with various bedrock models, and returning an well-formed response.

It has thee part:

1. Getting its settings. This uses Pydantic's `env_nested_delimiter`, which allows defining nested
   variables such as: BEDROCK_MODELS__CLAUDE_3_5_SONNET__ARN to set the right nested setting
2. Converting to and from the OpenAI spec
3. Implementing a subclass of BackendBase

'''


from datetime import datetime
import json
import structlog
from typing import Optional, Union, List, Literal, get_args, cast

import aioboto3
from aiobotocore.config import AioConfig

from pydantic import BaseModel, Field, NonNegativeInt, model_validator, ValidationError, ConfigDict
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict


from app.schema.open_ai import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionChoice,
    ChatCompletionUsage,
    TextContentPart,
    ImageContentPart,
    CohereRequest
)
from app.backends.base import BackendBase, LLMModel
from .utils import parse_data_uri

log = structlog.get_logger()

# Settings
# This is mostly about getting secrets
# but also makes it easy to define supported 
# in on location

class BedrockModel(BaseModel):  
    name: str
    id: str
    arn: str
    capability: Literal['chat', 'embedding']
 

class BedrockModelsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        extra='ignore',
        env_nested_delimiter="__"
    )

    claude_3_5_sonnet: BedrockModel = BedrockModel(
        name="Claude 3.5 Sonnet",
        id="claude_3_5_sonnet",
        capability="chat",
        arn="",                         
    )
    llama3211b: BedrockModel = BedrockModel(
        name="Llama 3.2 11B",
        id="llama3211b",
        capability="chat",
        arn="",                         
    )
    cohere_english_v3: BedrockModel = BedrockModel(
        name="Cohere English Ebmeddings",
        id="cohere_english_v3",
        capability="embedding",
        arn="",                         
    )


# Handle requests to Bedrock
# `models` is a required property and will be used to register models
# to ensure the correct instance handles requests. This is how the 
# application knows which backends serve which models.

class BedRockBackend(BackendBase):
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file='.env',extra='ignore', env_file_encoding='utf-8', env_nested_delimiter="__" )
        bedrock_assume_role: str = Field(default=...)
        aws_default_region: str = Field(default=...)
        bedrock_models: BedrockModelsSettings = BedrockModelsSettings()
        
        @model_validator(mode="after")
        def ensure_arns_present(self):
            missing = [
                key
                for key, model in self.bedrock_models.__dict__.items()
                if isinstance(model, BedrockModel) and not model.arn
            ]
            if missing:
                raise ValueError(
                    "Missing ARN for: " + ", ".join(missing)
                )
            return self


    
    def __init__(self):
        self.settings = self.Settings()
        self.retry_config = AioConfig(
            retries={"max_attempts": 5, "mode": "standard",},
            region_name=self.settings.aws_default_region
        )


    @property
    def models(self):
        return [LLMModel(**v) for v in self.settings.bedrock_models.model_dump().values()]


    async def invoke_model(self, payload: ChatCompletionRequest):
        converted = convert_open_ai_completion_bedrock(payload)
        session = aioboto3.Session()
        async with session.client("bedrock-runtime", config=self.retry_config) as client:
            body = converted.model_dump(exclude_none=True, by_alias=True)
            arn = getattr(self.settings.bedrock_models, converted.model_id).arn
            
            body['modelId'] = arn
            response = await client.converse(**body)
            log.info("bedrock metrics", model=converted.model_id, **response['metrics'])
        
            res = ConverseResponse(**response)
            return convert_bedrock_response_open_ai(res)


    async def embeddings(self, payload: CohereRequest):
        session = aioboto3.Session()
        async with session.client("bedrock-runtime", config=self.retry_config) as client:
            body = payload.model_dump_json(exclude_none=True)
            modelId = getattr(self.settings.bedrock_models, payload.model).arn 
            
            response = await client.invoke_model(
                body=body,
                modelId=modelId,
                accept = '*/*',
                contentType = 'application/json'
                )
            
            headers = response['ResponseMetadata']['HTTPHeaders']
            latency = headers['x-amzn-bedrock-invocation-latency']
            log.info("embedding", latency=latency, model=modelId)
            response_body = json.loads(await response.get("body").read())

            return response_body


# Types that model Bedrock's converse API.
# These won't be exposed to the users, but will be 
# used when converting to formats approriate for
# working with Bedrocks

# Document Block
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html


# Base structure for image source
class ImageSource(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: bytes = Field(..., description="Raw image data bytes.", alias="bytes")

# Payload for image content block
class ImagePayload(BaseModel):
    format: Literal['jpeg', 'png', 'gif', 'webp'] = Field(..., description="Image format.")
    source: ImageSource

# Specific content block for an image
class ContentImageBlock(BaseModel):
    image: ImagePayload

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
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, ge=0, le=1, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Top-p sampling")
    stop_sequences: Optional[List[str]] = Field(None, description="Optional list of stop sequences")

class ConverseRequest(BaseModel):
    model_config = ConfigDict(
        # Allows parsing input with either 'inference_config' or 'inferenceConfig'
        populate_by_name=True,
        alias_generator=to_camel
    )
    model_id: str = Field(..., exclude=True)
    messages: List[Message]
    inference_config: Optional[InferenceConfig] = Field(None, serialization_alias="inferenceConfig")

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




### Conversions OpenAI <-> Bedrock Converse

def convert_open_ai_stop(stop: Optional[Union[str, list[str]]] = None) -> Optional[list[str]]:
    # open AI allows this to be a string or a list of strings
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop

def convert_open_ai_messages(messages: list[ChatCompletionMessage]) -> List[Message]:
    """Converts OpenAI messages to Bedrock Converse API messages, handling multimodal content."""
    bedrock_messages = []
    for openai_msg in messages:

        if openai_msg.role == "system":
            log.warning("Warning: Skipping system message during conversion as Bedrock Converse API handles system prompts differently or not at all within 'messages'.")
            continue

        bedrock_content_blocks: List[ContentBlock] = []

        if isinstance(openai_msg.content, str):
            # Simple text content
            if openai_msg.content.strip(): 
                 bedrock_content_blocks.append(ContentTextBlock(text=openai_msg.content))
        elif isinstance(openai_msg.content, list):
            # Multimodal content list
            for part in openai_msg.content:
                if isinstance(part, TextContentPart):
                    if part.text.strip(): # Avoid empty blocks
                        bedrock_content_blocks.append(ContentTextBlock(text=part.text))
                elif isinstance(part, ImageContentPart):
                    if part.image_url.url.startswith("data:image"):
                        try:
                            image_data = parse_data_uri(part.image_url.url)
                            # Validate format against Bedrock's Literal
                            format_annotation = ImagePayload.model_fields['format'].annotation
                            allowed_formats = get_args(format_annotation) 

                            if image_data["format"] not in allowed_formats:
                                log.warning(f"Skipping image with unsupported format '{image_data['format']}'. Supported: jpeg, png, gif, webp.")
                                continue
                            source = ImageSource(bytes=image_data["data"])
                            payload = ImagePayload(format=image_data["format"], source=source)
                            bedrock_content_blocks.append(ContentImageBlock(image=payload))
                        except ValidationError as e:
                             log.warning(f"skipping image due to validation error: {e}")
                        except ValueError as e:
                            log.warning(f"Skipping image due to parsing error: {e}")

                    else:
                        # Let's not fetch images from the internet right now.
                        log.warn(f"Warning: Skipping image with HTTPS URL ({part.image_url.url[:50]}...). Conversion only supports Base64 data URIs.")
       
        if bedrock_content_blocks:
             # Ensure role is valid for Bedrock Message
             # this ignores OpenAI's other roles, not sure it this is the way
            format_annotation = Message.model_fields['role'].annotation
            bedrock_roles = get_args(format_annotation) 
            if openai_msg.role not in bedrock_roles:
                 log.warning(f"Skipping message with unsupported role '{openai_msg.role}' for Bedrock Converse.")
                 continue
        
            bedrock_role = cast(BedrockMessageRole, openai_msg.role)
            bedrock_msg = Message(role=bedrock_role, content=bedrock_content_blocks)
            bedrock_messages.append(bedrock_msg)
        elif openai_msg.role == 'user': 
            log.wanring("Warning: Skipping empty user message after conversion.")

    return bedrock_messages

def convert_open_ai_completion_bedrock(chat_completion: ChatCompletionRequest) -> ConverseRequest:
    """Converts an OpenAI ChatCompletionRequest to an AWS Bedrock ConverseRequest."""
    
    bedrock_messages = convert_open_ai_messages(chat_completion.messages)
    # Handle cases where messages might become empty after conversion (e.g., only system messages)
    if not bedrock_messages:
        raise ValueError("Cannot create Bedrock Converse request: No valid user/assistant messages found after conversion.")

    inference_conf = None
    # Check if any inference parameters are set in the OpenAI request
    if any([chat_completion.max_tokens is not None,
            chat_completion.temperature is not None,
            chat_completion.top_p is not None,
            chat_completion.stop is not None]):
        inference_conf = InferenceConfig(
            max_tokens=chat_completion.max_tokens,
            temperature=chat_completion.temperature,
            top_p=chat_completion.top_p,
            stop_sequences=convert_open_ai_stop(chat_completion.stop)
        )

    return ConverseRequest (
        model_id=chat_completion.model, # Note: Bedrock uses specific model IDs like 'anthropic.claude-3-sonnet-20240229-v1:0'
        messages=bedrock_messages,
        inference_config=inference_conf # Use snake_case here, Pydantic alias handles JSON output
    )

def convert_bedrock_response_open_ai(response: ConverseResponse) -> ChatCompletionResponse:
    choices =  []
    for i, content in enumerate(response.output['message'].content):
        message = ChatCompletionResponseMessage(
            role="assistant",
            content=content.text
        )
        choices.append(ChatCompletionChoice(
            index=i,
            message=message,
            finish_reason="stop"
        ))
    
    usage = ChatCompletionUsage(
        prompt_tokens=response.usage.inputTokens,
        completion_tokens=response.usage.outputTokens,
        total_tokens=response.usage.totalTokens 
    )

    return ChatCompletionResponse(
        object="chat.completion",
        created=datetime.now(),
        model="how do we get this?",
        choices=choices,
        usage=usage
    )


