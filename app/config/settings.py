from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.providers.base import Backend, LLMModel
from app.providers.bedrock.bedrock import BedRockBackend
from app.providers.vertex_ai.vertexai import VertexBackend

# Register Providers
backend_instances: List[Backend]  = [
    BedRockBackend(),
    VertexBackend()
]

_backend_map:dict[str,tuple[Backend, LLMModel]] ={}

for backend in backend_instances:
    for model in backend.models:
        _backend_map[model.id] = backend, model

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra="ignore", env_file_encoding='utf-8', env_nested_delimiter="__" )
    env:str = Field(default=...)
    log_level: str = Field(default=...)

    postgres_connection: str = Field(default=...)
    database_echo: bool = False
    
    bedrock_assume_role: str = Field(default=...)
    aws_default_region: str = Field(default=...)

    backend_map:dict[str,tuple[Backend, LLMModel]]


@lru_cache()
def get_settings() -> Settings:
    return Settings(backend_map=_backend_map)

