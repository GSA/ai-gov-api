from app.providers.bedrock.adapter_from_core import core_to_bedrock

def test_core_request_to_bedrock(core_chat_request, bedrock_chat_request):
    bedrock = core_to_bedrock(core_chat_request)
    assert bedrock == bedrock_chat_request


def test_core_full_request_to_bedrock(core_full_chat_request, bedrock_full_chat_request):
    bedrock = core_to_bedrock(core_full_chat_request)
    assert bedrock == bedrock_full_chat_request