import pytest

import app.providers.bedrock.converse_schemas as br
import app.providers.core.chat_schema as core
from app.providers.bedrock.adapter_to_core import bedrock_chat_stream_response_to_core

def test_message_start_event_conversion():
    raw_json_chunk_dict = {
        "messageStart": {
            "role": "assistant"
        }
    }
    validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)

    converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
    converted = next(converted)
    assert isinstance(converted, core.StreamResponse)
    assert converted.choices[0].delta is not None
    assert converted.choices[0].delta.role == "assistant"
    assert converted.choices[0].delta.content == ""
    assert converted.model == "some_model"

def test_content_block_start_event_conversion():
    raw_json_chunk_dict = {
        
        "contentBlockStart": { 
            "contentBlockIndex": 3,
            "start": {"text": {}}
        },
    }
    validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)
    converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
    with pytest.raises(StopIteration):
        converted = next(converted)
    

def test_content_block_delta_event_conversion():
    raw_json_chunk_dict = {
        
        "contentBlockDelta": { 
            "contentBlockIndex": 2,
            "delta": { 
                "text": "Hello"
             }
        }
    }
    validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)
    converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
    converted = next(converted)
    assert isinstance(converted, core.StreamResponse)
    assert converted.choices[0].delta is not None
    assert converted.choices[0].delta.content == "Hello"

def test_content_block_stop_event_conversion():
    raw_json_chunk_dict = {
        "contentBlockStop": { 
            "contentBlockIndex": 10
        }
    }
    validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)
    converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
    with pytest.raises(StopIteration):
        converted = next(converted)


def test_message_stop_event_conversion():
    raw_json_chunk_dict = {
         "messageStop": { 
            "additionalModelResponseFields": {},
            "stopReason": "end_message"
   },
    }
    validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)
    converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
    converted = next(converted)
    assert isinstance(converted, core.StreamResponse)
    assert converted.choices[0].delta is not None
    assert converted.choices[0].finish_reason == "end_message"


def test_metadata_conversion():
    raw_json_chunk_dict = {
        "metadata": { 
            "metrics": { 
                "latencyMs": 762
            },
            "usage": { 
                "inputTokens": 20,
                "outputTokens": 23,
                "totalTokens": 43
            }   
        }
    }
    validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)
    converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
    converted = next(converted)
    assert isinstance(converted, core.CompletionUsage)
    assert converted.prompt_tokens == 20
    assert converted.completion_tokens == 23
    assert converted.total_tokens == 43