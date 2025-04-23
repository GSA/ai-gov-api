
from app.backends.bedrock import (
    convert_open_ai_completion_bedrock, 
    convert_bedrock_response_open_ai,
    ConverseResponse
)


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

