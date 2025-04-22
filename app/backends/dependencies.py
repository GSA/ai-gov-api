from typing import Union
from fastapi import HTTPException
from app.config.settings import BACKEND_MAP
from app.schema.open_ai import ChatCompletionRequest
from app.schema.open_ai import CohereRequest

class Backend:
    def __init__(self, capability: str):
        self.capability = capability

    def __call__(self, req: Union[ChatCompletionRequest, CohereRequest]):
        model_id = req.model

        backend, capability = BACKEND_MAP.get(model_id, (None, None))

        if not backend:
            raise HTTPException(status_code=422, detail=f"Model '{model_id}' is not supported by this API.",)
        elif capability != self.capability:
            raise HTTPException(status_code=422, detail=f"This endpoint does support chat with the model '{model_id}'.",)

        return backend

