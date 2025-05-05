import pytest

from app.backends.bedrock.converse_schemas import (
     ConverseRequest, 
     Message, 
     ContentTextBlock,
     InferenceConfig,
)
from vertexai.generative_models import (
    Part,
    Content
)

from app.schema.open_ai import ChatCompletionRequest, ChatCompletionMessage, ImageContentPart, ImageUrl


@pytest.fixture(scope="module") 
def open_ai_example_with_system():
    return ChatCompletionRequest(
        model="claude_3_5_sonnet",
        messages=[
            ChatCompletionMessage(role="system", content="You speak only Pirate!"),
            ChatCompletionMessage(role="user", content="Hello!"),
            ChatCompletionMessage(role="system", content="You no longer speak only Pirate."),

        ],
        temperature=0,
        max_tokens=300
)

@pytest.fixture(scope="module") 
def open_ai_example():
    return ChatCompletionRequest(
        model="claude_3_5_sonnet",
        messages=[
            ChatCompletionMessage(role="user", content="Hello!"),
            ChatCompletionMessage(role="assistant", content="Hello! How can I assist you?")
        ],
        temperature=0,
        max_tokens=300
)

@pytest.fixture(scope="module") 
def open_ai_example_image(request):
    return ChatCompletionRequest(
        model="claude_3_5_sonnet",
        messages=[
            ChatCompletionMessage(role="user", content=[
                ImageContentPart(image_url=ImageUrl(url=request.param, detail="auto"))
            ])
        ],
        temperature=0,
        max_tokens=300
)

@pytest.fixture(scope="module") 
def bedrock_example():
    return ConverseRequest(
        model_id= "claude_3_5_sonnet",
        messages = [
            Message(role="user", content=[ContentTextBlock(text="Hello!")]),
            Message(role="assistant", content=[ContentTextBlock(text="Hello! How can I assist you?")])
        ],
        inference_config=InferenceConfig(temperature=0, max_tokens=300)
    )

@pytest.fixture(scope="module") 
def vertex_history():
    return [
        Content(role="user", parts=[Part.from_text("Hello!")]),
        Content(role="model", parts=[Part.from_text("Hello! How can I assist you?")])
    ]

@pytest.fixture(scope="module") 
def vertex_system():
    return [
        Content(role="user", parts=[
            Part.from_text("You speak only Pirate!"),
            Part.from_text("You no longer speak only Pirate.")
        ]),
        Content(role="model", parts=[Part.from_text("Okay")]),
        Content(role="user", parts=[Part.from_text("Hello!")])
    ]

@pytest.fixture(scope="module")
def bedrock_response():
    return  {
        'ResponseMetadata': {
            'RequestId': '479603d5-3abf-41ba-ba26-631167223a23', 
            'HTTPStatusCode': 200, 
            'HTTPHeaders': {'date': 'Tue, 08 Apr 2025 19:12:22 GMT', 'content-type': 'application/json', 'content-length': '215', 'connection': 'keep-alive', 'x-amzn-requestid': '479603d5-3abf-41ba-ba26-631167223a23'}, 
            'RetryAttempts': 0
        }, 
        'output': {
            'message': {
                'role': 'assistant', 
                'content': [{'text': 'Hi there! How can I help you today?'}]
            }
        },
        'stopReason': 'end_turn', 
        'usage': {'inputTokens': 9, 'outputTokens': 13, 'totalTokens': 22}, 
        'metrics': {'latencyMs': 871}
    }
