from fastapi import HTTPException
from app.config.settings import BACKEND_MAP
from app.schema.open_ai import ChatCompletionRequest
from app.schema.open_ai import EmbeddingRequest

class Backend:
    '''Dependency injection for routes'''
    def __init__(self, capability: str):
        self.capability = capability

    def __call__(self, req: ChatCompletionRequest | EmbeddingRequest):
        model_id = req.model

        backend, model = BACKEND_MAP.get(model_id, (None, None))

        if not backend or not model:
            raise HTTPException(status_code=422, detail=f"Model '{model_id}' is not supported by this API.",)
        elif model.capability != self.capability:
            raise HTTPException(status_code=422, detail=f"This endpoint not does support {self.capability} with the model '{model_id}'.",)

        return backend
