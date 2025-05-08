from app.providers.adaptors.open_ai_to_core import openai_chat_request_to_core

def test_open_ai_to_core(open_ai_example_with_system, core_request_with_system):
    core_request = openai_chat_request_to_core(open_ai_example_with_system)
    assert core_request == core_request_with_system