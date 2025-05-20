import os
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
    log_level:str = Field(default=...)

    db_user:str =  Field(default=...)
    db_pass:str =  Field(default=...)
    db_endpoint:str =  Field(default=...)
    db_name:str =  Field(default=...)
    db_port:str =  Field(default=...)
    
    database_echo: bool = False
    google_application_credentials: str = Field(default=...)
    
    bedrock_assume_role:str = Field(default=...)
    aws_default_region:str = Field(default=...)

    backend_map:dict[str,tuple[Backend, LLMModel]]

    @property
    def postgres_connection(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_pass}"
            f"@{self.db_endpoint}:{self.db_port}/{self.db_name}"
        )



@lru_cache()
def get_settings() -> Settings:
    settings = Settings(backend_map=_backend_map)
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

    return settings

