from datetime import datetime
from typing import Optional, Union

from app.schema.open_ai import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionChoice,
    ChatCompletionUsage,
)
from .bedrock import ContentText, Message, InferenceConfig, ConverseRequest, ConverseResponse



def convert_open_ai_stop(stop: Optional[Union[str, list[str]]] = None) -> Optional[list[str]]:
    # open AI allows this to be a string or a list of strings
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop

def convert_open_ai_messages(messages: list[ChatCompletionMessage]):
    result = []
    for chat_completion in messages:
        content_text = ContentText(text = chat_completion.content)
        message = Message(content = [content_text], role=chat_completion.role)
        result.append(message)
    return result

def convert_open_ai_completion_bedrock(chat_completion: ChatCompletionRequest) -> ConverseRequest:
    return ConverseRequest (
        model_id = chat_completion.model,
        inferenceConfig = InferenceConfig(
            max_tokens=chat_completion.max_tokens,
            temperature=chat_completion.temperature,
            top_p = chat_completion.top_p,
            stop_sequences = convert_open_ai_stop(chat_completion.stop)
        ),
        messages = convert_open_ai_messages(chat_completion.messages)
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
