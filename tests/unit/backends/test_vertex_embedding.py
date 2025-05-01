import pytest

from app.backends.vertex_ai.conversions import convert_open_ai_embedding, convert_vertex_embedding_response
from app.schema.open_ai import EmbeddingRequest
from vertexai.language_models import TextEmbedding
from vertexai.language_models._language_models import TextEmbeddingStatistics

def test_vertex_embedding_conversion():
    req = EmbeddingRequest(
        input = ["this is a test", "something else"],
        model = "text-embedding-005",
        input_type = 'search_document'
    )
    converted = convert_open_ai_embedding(req)
    assert converted[0].text == "this is a test"
    assert converted[1].text == "something else"
    assert converted[0].task_type == "RETRIEVAL_DOCUMENT"
    assert converted[1].task_type == "RETRIEVAL_DOCUMENT"


def test_vertex_single_embedding_conversion():
    req = EmbeddingRequest(
        input = "this is a test",
        model = "text-embedding-005",
        input_type = 'search_query'
    )
    converted = convert_open_ai_embedding(req)
    assert converted[0].text == "this is a test"
    assert converted[0].task_type == "RETRIEVAL_QUERY"


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

def test_vertex_embedding_response_conversion(vertex_embedding_output):
    model_id = "some-model"

    res = convert_vertex_embedding_response(vertex_embedding_output, model_id)
    assert res.data[0].embedding == [-0.1, 0.2, -0.5]
    assert res.data[1].embedding == [0.4, 0.2, 0.5]

def test_vertex_embedding_response_token_count(vertex_embedding_output):
    model_id = "some-model"
    res = convert_vertex_embedding_response(vertex_embedding_output, model_id)
    assert res.usage.total_tokens == 12
    assert res.usage.prompt_tokens == 12