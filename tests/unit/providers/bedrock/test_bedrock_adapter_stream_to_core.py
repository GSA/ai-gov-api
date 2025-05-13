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
    assert isinstance(converted, core.StreamResponse)

# def test_message_start_event_conversion():
#     raw_json_chunk_dict = {
#     "messageStart": {
#         "role": "assistant"
#     }
#     }
#     validated_wrapper_object = br.ConverseStreamChunk.model_validate(raw_json_chunk_dict)

#     converted = bedrock_chat_stream_response_to_core(resp=validated_wrapper_object, id="abc", model="some_model")
#     assert converted is None