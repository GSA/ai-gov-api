from functools import singledispatch
from typing import List, Tuple

from ..core.chat_schema import (
    ChatRequest,
    Message,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    TextPart,
    ImagePart,
    FilePart
)
from ..core.embed_schema import EmbedRequest

import app.providers.bedrock.converse_schemas as br 
from app.providers.bedrock.cohere_embedding_schemas import CohereRequest

AgentMessage = AssistantMessage | UserMessage

def extract_system_messages(messages:List[Message]) -> Tuple[List[SystemMessage],List[AgentMessage]]:
    system: List[SystemMessage]  = []
    other: List[AgentMessage] = []
    for m in messages:
        if m.role == "system":
            system.append(m)
        else:
            other.append(m)
    return system, other

@singledispatch
def _part_to_br(part) -> br.ContentBlock:
    raise TypeError(f"No converter for {type(part)}")

@_part_to_br.register
def _(part: TextPart) -> br.ContentTextBlock:
    return br.ContentTextBlock(text=part.text)

@_part_to_br.register
def _(part: ImagePart) -> br.ContentImageBlock:
    return br.ContentImageBlock(
        image=br.ImagePayload(
            format="jpeg",
            source=br.ImageSource(bytes=part.bytes_)
        )
    )

@_part_to_br.register
def _(part: FilePart) -> br.ContentDocumentBlock:
    return br.ContentDocumentBlock(
        document=br.DocumentPayload(
            format="pdf",
            name="",
            source=br.DocumentSource(bytes=part.bytes_)
        )
    )


def core_to_bedrock(req: ChatRequest) -> br.ConverseRequest:
    system_messages, messages = extract_system_messages(req.messages)
    return br.ConverseRequest(
        model_id=req.model,
        messages=[
            br.Message(
                role= m.role,
                content=[_part_to_br(p) for p in m.content],
            )
            for m in messages if m.content
        ],
        system=[
            br.SystemContentBlock(text=p.text) 
            for m in system_messages for p in m.content
        ], 
        inference_config=br.InferenceConfig(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_sequences=req.stop
        )
    )


def core_embed_request_to_bedrock(req: EmbedRequest) -> CohereRequest:
    return CohereRequest(
        model=req.model,
        texts=req.input,
        input_type=req.input_type, # type: ignore[arg-type]
        embedding_types=[req.encoding_format]
    )
