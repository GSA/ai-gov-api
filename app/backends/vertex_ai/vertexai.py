from typing import Literal, Any, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel, TextEmbedding

from app.schema.open_ai import ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest
from app.backends.base import BackendBase, LLMModel

from .conversions import (
    convert_vertex_response,
    convert_open_ai_messages,
    convert_open_ai_embedding,
    convert_vertex_embedding_response
)

log = structlog.get_logger()


class VertexModel(LLMModel):  
    name: str
    id: str
    capability: Literal['chat', 'embedding']
 

class VertexModelsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        extra='ignore',
        env_nested_delimiter="__"
    )
    gemini_2_0_flash:VertexModel = VertexModel(
        name="Gemini 2.0 Flash",
        id="gemini-2.0-flash",
        capability="chat"
    )
    text_embedding_005:VertexModel = VertexModel(
        name="Text embedding 005",
        id="text-embedding-005",
        capability="embedding"
    )


class VertexBackend(BackendBase):
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file='.env',extra='ignore', env_file_encoding='utf-8', env_nested_delimiter="__" )
        vertex_project_id:str = Field(default=...)
        vertex_models:VertexModelsSettings = VertexModelsSettings()

    
    def __init__(self):
        self.settings = self.Settings()
        vertexai.init(project=self.settings.vertex_project_id, location="us-central1")

    @property
    def models(self):
        return [LLMModel(**v) for v in self.settings.vertex_models.model_dump().values()]


    async def invoke_model(self, payload: ChatCompletionRequest) -> ChatCompletionResponse:
        model_id = payload.model
        # check that model is valid before moving on
        model = GenerativeModel(model_id)
    
        vertex_history = convert_open_ai_messages(payload.messages)

        temperature = payload.temperature
        max_tokens = payload.max_tokens
        top_p = payload.top_p
        stop_sequences = payload.stop

        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        elif stop_sequences is not None and not isinstance(stop_sequences, list):
            log.warning("'stop' parameter should be a string or list of strings. Ignoring.")
            stop_sequences = None

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
            # candidate_count=payload.n #  we could put OpenAI's n paramter here if we wanted to
            # but it's not available on Bedrock
        )
 
        response = await model.generate_content_async(
            vertex_history,
            generation_config=generation_config
        )
        
        return convert_vertex_response(response)
  
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
    async def embeddings(self, payload: EmbeddingRequest): 
        model = TextEmbeddingModel.from_pretrained(payload.model)
        parameters: dict[str, Any] = {
            "texts":  convert_open_ai_embedding(payload)
        }
        if payload.dimensions:
            parameters['output_dimensionality'] = payload.dimensions
            
        response: List[TextEmbedding] = await model.get_embeddings_async(**parameters)
        print(response)
        return convert_vertex_embedding_response(response, model_id=payload.model)