from app.providers.adaptors.core_to_vertex import convert_core_messages

def test_vertex_message_conversion(core_request_basic, vertex_history):
    messages = convert_core_messages(core_request_basic.messages)
    assert all(converted.text == fixture.text 
                for converted, fixture in zip(messages, vertex_history))
    assert all(converted.role == fixture.role 
               for converted, fixture in zip(messages, vertex_history))
    

def test_vertex_system_prompt_conversion(core_request_with_system, vertex_system):
    '''Vertex system prompts should be added as a user/model pair to the beginning'''
    messages = convert_core_messages(core_request_with_system.messages)
    assert all(converted.role == fixture.role 
              for converted, fixture in zip(messages, vertex_system))
    assert all(converted.text == fixture.text 
               for converted, fixture in zip(messages[0].parts, vertex_system[0].parts))
