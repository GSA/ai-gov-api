from app.schema.conversion import convert_open_ai_completion_bedrock, convert_bedrock_response_open_ai
from app.schema.bedrock import ConverseRequest, Message, ContentText, InferenceConfig, ConverseResponse
from app.schema.open_ai import ChatCompletionRequest, ChatCompletionMessage


open_ai_example = ChatCompletionRequest(
    model="claude_3_5_sonnet",
    messages=[
        ChatCompletionMessage(role="user", content="Hello!"),
        ChatCompletionMessage(role="assistant", content="Hello! How can I assist you?")
    ],
    temperature=0
)

bedrock_example = ConverseRequest(
    model_id= "claude_3_5_sonnet",
    messages = [
        Message(role="user", content=[ContentText(text="Hello!")]),
        Message(role="assistant", content=[ContentText(text="Hello! How can I assist you?")])
    ],
    inferenceConfig=InferenceConfig(temperature=0)
)

def test_convert_open_ai_request():
    converted = convert_open_ai_completion_bedrock(open_ai_example)
    assert converted.model_dump() == bedrock_example.model_dump()



bedrock_response = {
    'ResponseMetadata': {
        'RequestId': '479603d5-3abf-41ba-ba26-631167223a23', 
        'HTTPStatusCode': 200, 
        'HTTPHeaders': {'date': 'Tue, 08 Apr 2025 19:12:22 GMT', 'content-type': 'application/json', 'content-length': '215', 'connection': 'keep-alive', 'x-amzn-requestid': '479603d5-3abf-41ba-ba26-631167223a23'}, 
        'RetryAttempts': 0
    }, 
    'output': {
        'message': {
            'role': 'assistant', 
            'content': [{'text': 'Hi there! How can I help you today?'}]
        }
    },
    'stopReason': 'end_turn', 
    'usage': {'inputTokens': 9, 'outputTokens': 13, 'totalTokens': 22}, 
    'metrics': {'latencyMs': 871}
}
def test_bedrock_response():
    response = ConverseResponse(**bedrock_response)
    converted = convert_bedrock_response_open_ai(response)
    assert converted.choices[0].index == 0
    assert converted.choices[0].message.role == "assistant"
    assert converted.choices[0].message.content == "Hi there! How can I help you today?"

