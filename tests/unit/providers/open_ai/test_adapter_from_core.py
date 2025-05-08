from app.providers.open_ai.adapter_from_core import core_chat_response_to_openai

def test_core_chat_response_to_openai(openai_chat_reponse, core_chat_reponse):
    converted = core_chat_response_to_openai(core_chat_reponse)
    assert converted == openai_chat_reponse