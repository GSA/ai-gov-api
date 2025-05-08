from fastapi import HTTPException, Depends
from app.config.settings import get_settings
from app.providers.open_ai.schemas import ChatCompletionRequest
from app.providers.open_ai.schemas import EmbeddingRequest
from app.providers.base import Backend as BackendBase

class Backend:
    '''Dependency injection for routes'''
    def __init__(self, capability: str):
        self.capability = capability

    def __call__(self, req: ChatCompletionRequest | EmbeddingRequest, settings=Depends(get_settings)) -> BackendBase:
        model_id = req.model

        backend, model = settings.backend_map.get(model_id, (None, None))

        if not backend or not model:
            raise HTTPException(status_code=422, detail=f"Model '{model_id}' is not supported by this API.",)
        elif model.capability != self.capability:
            raise HTTPException(status_code=422, detail=f"This endpoint not does support {self.capability} with the model '{model_id}'.",)

        return backend
