import base64
import binascii

from datetime import datetime
from typing import Optional, Union, List, get_args, cast

import structlog

from app.providers.utils import parse_data_uri

from app.schema.open_ai import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionChoice,
    ChatCompletionUsage,
    TextContentPart,
    ImageContentPart,
    FileContentPart,
    #EmbeddingRequest
)
from ..exceptions import InvalidBase64DataError, InputDataError
from .converse_schemas import (
    Message,
    ContentBlock,
    ContentTextBlock,
    ContentImageBlock,
    SystemContentBlock,
    ImagePayload,
    BedrockMessageRole,
    ImageSource,
    ConverseRequest,
    InferenceConfig,
    ConverseResponse,
    DocumentSource,
    DocumentPayload,
    ContentDocumentBlock
)

log = structlog.get_logger()
### Conversions OpenAI <-> Bedrock Converse

def convert_open_ai_stop(stop: Optional[Union[str, list[str]]] = None) -> Optional[list[str]]:
    # open AI allows this to be a string or a list of strings
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop

def convert_open_ai_messages(messages: list[ChatCompletionMessage]) -> tuple[List[Message], Optional[List[SystemContentBlock]]]:
    """Converts OpenAI messages to Bedrock Converse API messages, handling multimodal content."""
    bedrock_messages = []
    bedrock_system_message: List[SystemContentBlock] | None = []
    
    for idx, openai_msg in enumerate(messages):
        if openai_msg.role == "system":
            # collect system messages which live at a level higher in bedrock
            if isinstance(openai_msg.content, str):
                bedrock_system_message.append(SystemContentBlock(text=openai_msg.content))
            elif isinstance(openai_msg.content, list):
                for message in openai_msg.content:
                    if message.type == "text":
                        bedrock_system_message.append(SystemContentBlock(text=message.text))
            continue

        bedrock_content_blocks: List[ContentBlock] = []

        if isinstance(openai_msg.content, str):
            # Simple text content
            if openai_msg.content.strip(): 
                 bedrock_content_blocks.append(ContentTextBlock(text=openai_msg.content))
        elif isinstance(openai_msg.content, list):
            # Multimodal content list
            for part in openai_msg.content:
                match part:
                    case TextContentPart(text=text):
                        if text.strip(): # Avoid empty blocks
                            bedrock_content_blocks.append(ContentTextBlock(text=text))
                    
                    case FileContentPart(file=file):
                        
                        data_bytes = file.file_data
                       
                        doc_source = DocumentSource(bytes=data_bytes)
                        payload = DocumentPayload(source=doc_source, name="test", format='pdf')
                        bedrock_content_blocks.append(ContentDocumentBlock(document=payload))
                    
                    case ImageContentPart(image_url=image_url):
                        if image_url.url.startswith("data:image"):
                            try:
                                image_data = parse_data_uri(image_url.url)
                            except InputDataError as e:
                                raise InvalidBase64DataError(
                                    f"Invalid base64 encoding at message index '{idx}': {e}",
                                    field_name=part.type,
                                    original_exception=e
                                ) from e 
                            # Validate format against Bedrock's Literal
                            format_annotation = ImagePayload.model_fields['format'].annotation
                            allowed_formats = get_args(format_annotation) 

                            if image_data["format"] not in allowed_formats:
                                log.warning(f"Skipping image with unsupported format '{image_data['format']}'. Supported: jpeg, png, gif, webp.")
                                continue
                            source = ImageSource(bytes=image_data["data"])
                            payload = ImagePayload(format=image_data["format"], source=source)
                            bedrock_content_blocks.append(ContentImageBlock(image=payload))
                        
                        else:
                            raise InvalidBase64DataError(
                                f"Invalid base64 encoding at message index '{idx}': image url must begin with 'data:image'",
                                field_name=part.type
                            )
       
        if bedrock_content_blocks:
            # Ensure role is valid for Bedrock Message
            # OpenAI allows other roles,but we left them off the OpenAI models
            # so this probably not needed at the moment.
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
    
    if not bedrock_system_message:
        bedrock_system_message = None
        
    return bedrock_messages, bedrock_system_message

def convert_open_ai_completion_bedrock(chat_completion: ChatCompletionRequest) -> ConverseRequest:
    """Converts an OpenAI ChatCompletionRequest to an AWS Bedrock ConverseRequest."""
    
    bedrock_messages, bedrock_system_message = convert_open_ai_messages(chat_completion.messages)
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
        system=bedrock_system_message,
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