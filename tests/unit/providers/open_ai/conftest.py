import pytest
from datetime import datetime

from app.providers.open_ai.schemas import (
    ChatCompletionRequest,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    TextContentPart,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
    ImageContentPart,
    ImageUrl, 
    FileContentPart,
    FileContent
)

@pytest.fixture
def openai_chat_request():
    return ChatCompletionRequest(
        model="test-model",
        messages = [
            UserMessage(content=[TextContentPart(text="Hello!")]),
            AssistantMessage(content=[TextContentPart(text="Hello! How can I assist you?")])
        ]
    )

@pytest.fixture
def openai_full_chat_request():
    return ChatCompletionRequest(
        model="test-model",
        messages = [
            SystemMessage(content=[TextContentPart(text="Speak Pirate!")]),
            UserMessage(content=[TextContentPart(text="Hello!")]),
            AssistantMessage(content=[TextContentPart(text="Hello! How can I assist you?")])
        ],
        temperature=1.0,
        top_p=.5,
        max_tokens=1000,
        stream=True,
        stop=["stop", "STOP"]

    )

@pytest.fixture
def openai_chat_reponse():
    return ChatCompletionResponse(
        model="test-model",
        created=datetime(2024, 12, 25),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content='It was the afternoon of my eighty-first birthday, and I was in bed…'
                    )
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=10, completion_tokens=12, total_tokens=22)
    )


@pytest.fixture(scope="module") 
def open_ai_example_image(request):
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            UserMessage(role="user", content=[
                ImageContentPart(image_url=ImageUrl(url=request.param, detail="auto"))
            ])
        ]
    )

@pytest.fixture(scope="module") 
def open_ai_example_file(request):
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            UserMessage(role="user", content=[
                FileContentPart(file=FileContent(file_data=request.param))
            ])
        ],
        temperature=0,
        max_tokens=300
    )