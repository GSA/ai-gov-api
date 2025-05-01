from datetime import datetime
from typing import  List

import structlog

from vertexai.generative_models import (
    Part,
    Content,
    GenerationResponse
)

from vertexai.language_models import TextEmbeddingInput

from app.schema.open_ai import (
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionMessage,
    TextContentPart,
    ImageContentPart,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    EmbeddingData
)

from app.backends.utils import parse_data_uri

log = structlog.get_logger()

def convert_vertex_response(resp: GenerationResponse) -> ChatCompletionResponse:
        usage = ChatCompletionUsage(
            prompt_tokens=resp.usage_metadata.prompt_token_count,
            completion_tokens=resp.usage_metadata.candidates_token_count,
            total_tokens=resp.usage_metadata.total_token_count, 
        )   
        choices = []
        for i, candidate in enumerate(resp.candidates):
            message = ChatCompletionResponseMessage(
                role="assistant",
                # we will need to revist for multimodal output
                content=candidate.content.parts[0].text
            )
            choices.append(ChatCompletionChoice(
                index=i,
                message=message,
                finish_reason="stop"
            ))
        
        return ChatCompletionResponse(
            object="chat.completion",
            created=datetime.now(),
            model=resp._raw_response.model_version,
            choices=choices,
            usage=usage
        )

def to_vertex_role(role: str): 
    if role == "user":
        return "user"
    elif role == "assistant":
        return "model"
    else:
        log.warning(f"Unsupported role '{role}' found. Skipping message.")
        # maybe this should raise
        return role


def convert_open_ai_messages(messages: list[ChatCompletionMessage]) -> List[Content]:
    vertex_history:List[Content] = []
    system_prompt_parts:List[Part]  = []

    for openai_msg in messages:
        if openai_msg.role == "system":
            # collect system messages which are addeds as a user message in vertex
            if isinstance(openai_msg.content, str):
                system_prompt_parts.append(Part.from_text(openai_msg.content))
            elif isinstance(openai_msg.content, list):
                system_message = '\n'.join(content.text for content in openai_msg.content if content.type=="text")
                system_prompt_parts.append(Part.from_text(system_message))
            continue

        vertex_role = to_vertex_role(openai_msg.role)

        if isinstance(openai_msg.content, str):
            # Simple text content
            if openai_msg.content.strip(): 
                vertex_history.append(Content(role=vertex_role, parts=[Part.from_text(openai_msg.content)]))

        elif isinstance(openai_msg.content, list):
            # Multimodal content list
            for part in openai_msg.content:
                if isinstance(part, TextContentPart):
                    if part.text.strip(): # Avoid empty blocks
                        vertex_history.append(Content(role=vertex_role, parts=[Part.from_text(part.text)]))
                elif isinstance(part, ImageContentPart):
                    if part.image_url.url.startswith("data:image"):
                        try:
                            image_data = parse_data_uri(part.image_url.url)
                            image_part = Part.from_data(data=image_data['data'], mime_type=f"image/{image_data['format']}")
                            vertex_history.append(Content(role=vertex_role, parts=[image_part]))
                        except ValueError as e:
                            log.warning(f"Skipping image due to parsing error: {e}")
                    else:
                        # Let's not fetch images from the internet right now.
                        log.warn(f"Warning: Skipping image with HTTPS URL ({part.image_url.url[:50]}...). Conversion only supports Base64 data URIs.")


    if system_prompt_parts:
        # Gemini does not have system prompts in conversations. Google recommends
        # adding a context prompt at the beginning with s model acknoledgment.
        vertex_history.insert(0, Content(role="user", parts=system_prompt_parts))
        vertex_history.insert(1, Content(role="model", parts=[Part.from_text("Okay, I will follow these instructions.")]))

    return vertex_history

def convert_open_ai_embedding(req: EmbeddingRequest) -> List[TextEmbeddingInput]:
    task_type_mapping = {
        "search_document": "RETRIEVAL_DOCUMENT",
        "search_query": "RETRIEVAL_QUERY",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
        "semantic_similarity": "SEMANTIC_SIMILARITY",
    }
    input_type=None
    
    if req.input_type:
        input_type = task_type_mapping.get(req.input_type)

    input = [req.input] if isinstance(req.input, str) else req.input
    
    requests = []
    for text in input:
        requests.append(TextEmbeddingInput(
            text=text,
            task_type=input_type
        ))

    return requests


def convert_vertex_embedding_response(res, model_id:str) -> EmbeddingResponse:
    token_count = sum(int(emb.statistics.token_count) for emb in res)
    usages = EmbeddingUsage(
        promptTokens=token_count,
        totalTokens=token_count
    )
    
    return EmbeddingResponse(
        object="list",
        data=[
            EmbeddingData(
                object = "embedding",
                embedding=emb.values, 
                index=i
            )
            for i, emb in enumerate(res)
        ],
        model=model_id,
        usage=usages
    )