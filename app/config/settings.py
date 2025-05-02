from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.backends.base import BackendBase
from app.backends.bedrock.bedrock import BedRockBackend
from app.backends.vertex_ai.vertexai import VertexBackend

# Register backends
backend_instances = [
    BedRockBackend(),
    VertexBackend()
]

# TODO revist this if we have more capabilities; this probably won't scale
BACKEND_MAP:dict[str,tuple[BackendBase, Literal['chat', 'embedding']]] ={}

for backend in backend_instances:
    for model in backend.models:
        BACKEND_MAP[model.id] = backend, model.capability

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra="ignore", env_file_encoding='utf-8', env_nested_delimiter="__" )
    env:str = Field(default=...)
    log_level: str = Field(default=...)

    postgres_connection: str = Field(default=...)
    database_echo: bool = False
    
    bedrock_assume_role: str = Field(default=...)
    aws_default_region: str = Field(default=...)

settings = Settings()
