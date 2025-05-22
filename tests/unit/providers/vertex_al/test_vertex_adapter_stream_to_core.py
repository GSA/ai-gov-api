import pytest
from uuid import UUID
from datetime import datetime
import app.providers.core.chat_schema as core
from app.providers.vertex_ai.adapter_to_core import vertex_stream_response_to_core
from vertexai.generative_models import FinishReason

# -- Mocks --

class MockPart:
    def __init__(self, text=None):
        self.text = text

class MockContent:
    def __init__(self, parts):
        self.parts = parts

class MockCandidate:
    def __init__(self, text=None, finish_reason=None):
        self.content = MockContent([MockPart(text)] if text else [])
        self.finish_reason = finish_reason

class MockVertexResponse:
    def __init__(self, candidates):
        self.candidates = candidates


async def make_stream(responses):
    for r in responses:
        yield r

@pytest.mark.asyncio
async def test_single_content_chunk_with_role_and_finish_reason():
    vertex_stream = make_stream([
        MockVertexResponse([
            MockCandidate(text="Hello", finish_reason=FinishReason.STOP)
        ])
    ])
    results:list[core.StreamResponse] = [resp async for resp in vertex_stream_response_to_core(vertex_stream, model_id="vertex-ai-001")]

    assert len(results) == 3

    role_chunk = results[0]
    assert role_chunk.choices[0].delta is not None
    assert role_chunk.choices[0].delta.role == "assistant"

    content_chunk = results[1]
    assert content_chunk.choices[0].delta is not None
    assert content_chunk.choices[0].delta.content == "Hello"

    finish_chunk = results[2]
    assert finish_chunk.choices[0].finish_reason == "stop" 
    assert finish_chunk.choices[0].delta is not None
    assert finish_chunk.choices[0].delta.model_dump(exclude_none=True) == {}

    for r in results:
        assert UUID(r.id.replace("chatcmpl-", ""), version=4)
        assert isinstance(r.created, datetime)


@pytest.mark.asyncio
async def test_ignores_empty_candidates():
    vertex_stream = make_stream([
        MockVertexResponse([MockCandidate()])
    ])
    results = [r async for r in vertex_stream_response_to_core(vertex_stream, "model")]

    assert results == []


@pytest.mark.asyncio
async def test_multiple_candidates_and_index_tracking():
    vertex_stream = make_stream([
        MockVertexResponse([
            MockCandidate(text="First", finish_reason=None),
            MockCandidate(text="Second", finish_reason="STOP")
        ])
    ])
    results = [r async for r in vertex_stream_response_to_core(vertex_stream, "model")]

    # Expecting:
    # 0: role chunk for idx 0
    # 1: content chunk for idx 0
    # 2: role chunk for idx 1
    # 3: content chunk for idx 1
    # 4: finish chunk for idx 1
    roles = [r for r in results if r.choices[0].delta is not None and r.choices[0].delta.role == "assistant"]
    assert len(roles) == 2
    assert [r.choices[0].index for r in roles] == [0, 1]


@pytest.mark.asyncio
async def test_multiple_parts_in_candidate():
    class MultiPartCandidate:
        def __init__(self):
            self.content = MockContent([
                MockPart("Hello"), MockPart(" world")
            ])
            self.finish_reason = None

    vertex_stream = make_stream([
        MockVertexResponse([MultiPartCandidate()])
    ])
    results = [r async for r in vertex_stream_response_to_core(vertex_stream, "model")]

    contents = [r.choices[0].delta.content for r in results if r.choices[0].delta is not None and r.choices[0].delta.content]
    assert contents == ["Hello", " world"]  
