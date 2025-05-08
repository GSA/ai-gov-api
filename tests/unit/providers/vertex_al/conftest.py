import pytest
from app.providers.vertex_ai.schemas import EmbeddingRequest, TextEmbeddingInput
from vertexai.generative_models import Part,Content, GenerationResponse
from vertexai.language_models import TextEmbedding
from vertexai.language_models._language_models import TextEmbeddingStatistics


@pytest.fixture(scope="module") 
def vertex_history():
    return [
        Content(role="user", parts=[Part.from_text("Hello!")]),
        Content(role="model", parts=[Part.from_text("Hello! How can I assist you?")])
    ]

@pytest.fixture(scope="module") 
def vertex_full_history():
    return [
        Content(role="user", parts=[Part.from_text("Speak Pirate!")]),
        Content(role="model", parts=[Part.from_text("Okay, I will follow these instructions.")]),
        Content(role="user", parts=[Part.from_text("Hello!")]),
        Content(role="model", parts=[Part.from_text("Hello! How can I assist you?")])
    ]

@pytest.fixture(scope="module")
def vertex_chat_response():
    return GenerationResponse.from_dict({   
        "createTime": "2024-12-25T00:01:00Z",
        "model_version": "test-model",
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "It was the afternoon of my eighty-first birthday, and I was in bed…"}]}
                }
        ],
        "usage_metadata":{
            "promptTokenCount":10, 
            "candidatesTokenCount":12,
            "totalTokenCount":22
        }
    })

@pytest.fixture()
def vertex_embed_request():
    return EmbeddingRequest(
        texts = [
            TextEmbeddingInput(text="this is a test", task_type="RETRIEVAL_DOCUMENT"),
            TextEmbeddingInput(text="something else", task_type="RETRIEVAL_DOCUMENT")
        ]
    )

@pytest.fixture()
def vertex_embedding_output():
    return [
        TextEmbedding(
            values=[-0.1, 0.2, -0.5],
            statistics=TextEmbeddingStatistics(token_count=11, truncated=False),
            ),
        TextEmbedding(
            values=[0.4, 0.2, 0.5],
            statistics=TextEmbeddingStatistics(token_count=1, truncated=False),
        )
    ]
