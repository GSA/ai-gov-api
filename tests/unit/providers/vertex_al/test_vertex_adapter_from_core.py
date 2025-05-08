from app.providers.vertex_ai.adapter_from_core import convert_core_messages, convert_embedding_request

def test_vertex_message_conversion(core_chat_request, vertex_history):
    messages = convert_core_messages(core_chat_request.messages)
    assert all(converted.text == fixture.text 
                for converted, fixture in zip(messages, vertex_history))
    assert all(converted.role == fixture.role 
               for converted, fixture in zip(messages, vertex_history))
    

def test_vertex_system_message_conversion(core_full_chat_request, vertex_full_history):
    messages = convert_core_messages(core_full_chat_request.messages)
    assert all(converted.text == fixture.text 
                for converted, fixture in zip(messages, vertex_full_history))
    assert all(converted.role == fixture.role 
               for converted, fixture in zip(messages, vertex_full_history))
    

def test_embedding_request_conversion(core_embed_request, vertex_embed_request):
    converted = convert_embedding_request(core_embed_request)
    assert converted == vertex_embed_request

