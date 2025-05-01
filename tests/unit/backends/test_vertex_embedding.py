from app.backends.vertex_ai.conversions import convert_open_ai_embedding
from app.schema.open_ai import EmbeddingRequest

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