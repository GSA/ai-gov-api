from typing import cast 
import pytest 

from app.backends.bedrock.converse_conversions import (
    convert_open_ai_completion_bedrock, 
    convert_bedrock_response_open_ai,
    ConverseResponse,
    ContentImageBlock
)

from app.backends.exceptions import InvalidBase64DataError


def test_convert_open_ai_request(bedrock_example, open_ai_example):
    '''Should convert a valid OpenAI ChatCompletion requests to a Bedrock Converse Request'''
    converted = convert_open_ai_completion_bedrock(open_ai_example)
    assert converted.model_dump(by_alias=True) == bedrock_example.model_dump(by_alias=True)

def test_convert_open_ai_request_with_system_prompt(bedrock_example, open_ai_example_with_system):
    '''Should extract system prompts from OpenAi content blocks into single list'''
    converted = convert_open_ai_completion_bedrock(open_ai_example_with_system)
    converted_dict = converted.model_dump(by_alias=True)
    assert converted_dict['system'] == [
        {'text': 'You speak only Pirate!'},
        {'text': 'You no longer speak only Pirate.'}
    ]
    assert converted_dict['messages'] == [{'role': 'user', 'content': [{'text': 'Hello!'}]}]

@pytest.mark.parametrize(
    ("open_ai_example_image", "expected_exc", "msg_part"),
    [
        pytest.param("abci23", InvalidBase64DataError, "Invalid base64 encoding"),
        pytest.param("data:image/xls", InvalidBase64DataError, "Invalid or unsupported image data URI format."),
        pytest.param("data:image/jpeg;base64,abcde", InvalidBase64DataError, "Invalid base64 encoding"),
        pytest.param("data:image/jpeg;base64,abcd=", None, [b'i\xb7\x1d', "jpeg"]),

    ],
    indirect=("open_ai_example_image",)
)
def test_convert_open_ai_request_with_image(open_ai_example_image, expected_exc, msg_part):
    ''' It should produce the correct bytes for good image formats or raise appropriate exception'''
    if expected_exc is not None:
        with pytest.raises(expected_exc) as exc_info:
            convert_open_ai_completion_bedrock(open_ai_example_image)
        assert msg_part in str(exc_info.value)
    else:
        data_bytes, format = msg_part
        converted = convert_open_ai_completion_bedrock(open_ai_example_image)
        assert len(converted.messages) == 1
        imageblock:ContentImageBlock = cast(ContentImageBlock, converted.messages[0].content[0])
        assert imageblock.image.format == format
        assert imageblock.image.source.data == data_bytes

def test_convert_camel_case(open_ai_example):
    '''The bedrock model uses camelCase in spots. Make sure we correctly serialize'''
    converted = convert_open_ai_completion_bedrock(open_ai_example)
    serialized = converted.model_dump(by_alias=True)
    assert "inferenceConfig" in serialized
    assert "maxTokens" in serialized['inferenceConfig']

def test_bedrock_response(bedrock_response):
    response = ConverseResponse(**bedrock_response)
    converted = convert_bedrock_response_open_ai(response)
    assert converted.choices[0].index == 0
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content == "Hi there! How can I help you today?"

