from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Annotated, Literal

class BedrockModel(BaseModel):  
    name: str
    arn: str
 

available_models = Annotated[
    Literal[
        'claude_3_5_sonnet',
        'llama_3_2_11B'
    ],
    "This will be used to validate model types on chat input"
]

class BedrockModelsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        nested_model_default_partial_update=True,
        extra='ignore',
        env_nested_delimiter="__"
    )

    claude_3_5_sonnet: BedrockModel = BedrockModel(
        name="Claude 3.5 Sonnet",
        arn="",                         
    )
    llama3211b: BedrockModel = BedrockModel(
        name="Llama 3.2 11B",
        arn="",                         
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', env_nested_delimiter="__" )
    env:str #= "dev"
    log_level: str# = "INFO"

    postgres_connection: str
    database_echo: bool = False
    
    bedrock_assume_role: str
    aws_default_region: str
    bedrock_models: BedrockModelsSettings = BedrockModelsSettings()
    
    cohere_embed_model_id: str

    @model_validator(mode="after")
    def ensure_arns_present(self):
        missing = [
            key
            for key, model in self.bedrock_models.__dict__.items()
            if isinstance(model, BedrockModel) and not model.arn
        ]
        if missing:
            raise ValueError(
                "Missing ARN for: " + ", ".join(missing)
            )
        return self

settings = Settings()
