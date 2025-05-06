from app.providers.vertex_ai.conversions import convert_open_ai_messages

def test_vertex_message_conversion(open_ai_example, vertex_history):
    messages = convert_open_ai_messages(open_ai_example.messages)

    ## Google doesn't implement __eq__ on their model objects ?!?
    assert all(converted.text == fixture.text 
               for converted, fixture in zip(messages, vertex_history))
    assert all(converted.role == fixture.role 
               for converted, fixture in zip(messages, vertex_history))
    

def test_vertex_system_prompt_conversion(open_ai_example_with_system, vertex_system):
    '''Vertex system prompts should be added as a user/model pair to the beginning'''
    messages = convert_open_ai_messages(open_ai_example_with_system.messages)
    assert all(converted.role == fixture.role 
              for converted, fixture in zip(messages, vertex_system))
    assert all(converted.text == fixture.text 
               for converted, fixture in zip(messages[0].parts, vertex_system[0].parts))
