from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.backends.base import BackendBase
from app.backends.bedrock import BedRockBackend


# Register backends
backend_instances = [
    BedRockBackend()
]

# TODO revist this if we have more capabilities; this probably won't scale
BACKEND_MAP:dict[str,tuple[BackendBase, Literal['chat', 'embedding']]] ={}

bedrock_backend = BedRockBackend()
for backend in backend_instances:
    for model in backend.models:
        BACKEND_MAP[model.id] = bedrock_backend, model.capability

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra="ignore", env_file_encoding='utf-8', env_nested_delimiter="__" )
    env:str #= "dev"
    log_level: str# = "INFO"

    postgres_connection: str
    database_echo: bool = False
    
    bedrock_assume_role: str
    aws_default_region: str

settings = Settings()
