from datetime import datetime
from unittest import mock

from app.providers.vertex_ai.adapter_to_core import convert_chat_vertex_response, vertex_embed_reposonse_to_core

@mock.patch("app.providers.vertex_ai.adapter_to_core.datetime")
def test_vertex_chat_reposonse_to_core(mock_datetime, core_chat_reponse, vertex_chat_response):
    mock_datetime.now.return_value = datetime(2024, 12, 25)
    converted = convert_chat_vertex_response(vertex_chat_response, model="test-model")
    assert core_chat_reponse == converted


def test_vertex_embed_reposonse_to_core(vertex_embedding_output, core_embed_response):
    converted = vertex_embed_reposonse_to_core(vertex_embedding_output, model="test-model")
    assert converted == core_embed_response