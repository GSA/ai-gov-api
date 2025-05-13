from datetime import datetime
from functools import singledispatch

from ..core.chat_schema import ChatRepsonse, CompletionUsage, Response, StreamResponse, StreamResponseChoice, StreamResponseDelta
from ..core.embed_schema import EmbeddingResponse, EmbeddingData, EmbeddingUsage
from .converse_schemas import ConverseResponse, ConverseStreamChunk, MetadataEvent, MessageStartEvent, ContentBlockStartEvent, ContentBlockDeltaEvent, ContentBlockStopEvent, MessageStopEvent
from .cohere_embedding_schemas import CohereRepsonse

def bedrock_chat_response_to_core(resp: ConverseResponse, model:str) -> ChatRepsonse:
    return ChatRepsonse(
        model=model,
        created=datetime.now(),
        choices=[
            Response(content=r.text)
            for r in resp.output['message'].content
        ],
        usage=CompletionUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.total_tokens
        )
    )

def bedorock_embed_reposonse_to_core(resp: CohereRepsonse, model:str, token_count) -> EmbeddingResponse:
    return EmbeddingResponse(
        model=model,
        data=[EmbeddingData(index=idx, embedding=data) for idx, data in enumerate(resp.embeddings['float'])],
        usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count)
    )


# --------- Handle the menagerie of stream chunk types

@singledispatch
def _usage_to_oai(part: ConverseStreamChunk) -> CompletionUsage | None:
    return None

@_usage_to_oai.register
def _(part: MetadataEvent) -> CompletionUsage:
    return CompletionUsage(
            prompt_tokens=part.usage.input_tokens,
            completion_tokens=part.usage.output_tokens,
            total_tokens=part.usage.total_tokens,
    )
    
@singledispatch
def _event_to_oai(part: ConverseStreamChunk) -> StreamResponseChoice | None:
    raise TypeError(f"No converter for {type(part)}")

@_event_to_oai.register
def _(part: MessageStartEvent) -> StreamResponseChoice | None:
    return StreamResponseChoice(
        index=0,
        delta=StreamResponseDelta(
            role="assistant",
            content=""
        )
    )

@_event_to_oai.register
def _(part: ContentBlockStartEvent) -> StreamResponseChoice | None:
    # This can also be a tool call
    return None

@_event_to_oai.register
def _(part: ContentBlockDeltaEvent) -> StreamResponseChoice | None:
    return StreamResponseChoice(
        index=0, # were does part.content_block_delta.delta.index fit?
        delta=StreamResponseDelta(
            content=part.content_block_delta.delta.text
        )
    )

@_event_to_oai.register
def _(part: ContentBlockStopEvent) -> StreamResponseChoice | None:
   return None
   
@_event_to_oai.register
def _(part: MessageStopEvent) -> StreamResponseChoice | None:
    # This can also be a tool call
    return StreamResponseChoice(
        index=0,
        delta=StreamResponseDelta(),
        finish_reason=part.message_stop.stop_reason
    )


def bedrock_chat_stream_response_to_core(resp: ConverseStreamChunk, model:str, id:str) -> StreamResponse | None:
    print("resp: ", resp)
    print("root:", type(resp.root))
    event = _event_to_oai(resp.root)
    print("event:", event)
    if event is None: 
        return None
    return StreamResponse(
        id=id,
        object="chat.completion.chunk",
        model=model,
        created=datetime.now(),
        choices=[event],
        usage=_usage_to_oai(resp.root)
    )
