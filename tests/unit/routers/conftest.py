import pytest
from typing import Generator, TypeVar
from uuid import uuid4
from datetime import datetime
from fastapi.testclient import TestClient

from app.main import app
from app.config.settings import Settings, get_settings
from app.auth.dependencies import valid_api_key
from app.auth.schemas import APIKeyOut 
from app.providers.base import LLMModel, Backend

T = TypeVar("T")
YieldFixture = Generator[T, None, None]

@pytest.fixture(scope="session") 
def mock_llm_models() -> list[LLMModel]:
    return [
        LLMModel(name="Test Model", id="test_model", capability="chat"),
        LLMModel(name="Test Embed", id="test_embed_model", capability="embedding")
    ]

@pytest.fixture(scope="session") 
def mock_backend(mock_llm_models: list[LLMModel]) -> Backend:
    class TestBackend(Backend):
        def __init__(self, models_data: list[LLMModel]):
            self._models = models_data

        @property
        def models(self):
            return self._models

    return TestBackend(mock_llm_models)

@pytest.fixture(scope="module") 
def mock_settings(mock_backend: Backend, mock_llm_models: list[LLMModel]) -> Settings:
    backend_map_config = {
        mock_llm_models[0].id: (mock_backend, mock_llm_models[0]),
        mock_llm_models[1].id: (mock_backend, mock_llm_models[1]),
    }
    return Settings(backend_map=backend_map_config)

@pytest.fixture(scope="function") 
def mock_valid_api_key_out() -> APIKeyOut:
    return APIKeyOut(
        manager_id=uuid4(),
        hashed_key="mock_hashed_xyz", 
        key_prefix="test",
        id=123, 
        created_at=datetime.now() 
    )

@pytest.fixture(scope="function") 
def client(mock_settings: Settings, mock_valid_api_key_out: APIKeyOut) ->  YieldFixture:
    def get_settings_override() -> Settings:
        return mock_settings

    def valid_api_key_override() -> APIKeyOut:
        return mock_valid_api_key_out

    app.dependency_overrides[get_settings] = get_settings_override
    app.dependency_overrides[valid_api_key] = valid_api_key_override

    yield TestClient(app) 

    app.dependency_overrides.clear()
