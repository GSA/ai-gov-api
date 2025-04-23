"""
BackendBase
~~~~~~~~~~~

This is an abstract class that represents a provider like Bedrock.
The only required field models, which should return a list of LLMModel
instances to communicate which models this backend can serve and what
capabilities they have (currently just chat and embedding).
"""

from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel
from app.schema.open_ai import ChatCompletionRequest, ChatCompletionResponse, CohereRequest

class LLMModel(BaseModel):
    name: str
    id: str
    capability: Literal['chat', 'embedding']

class BackendBase(ABC):

    async def invoke_model(self, payload:ChatCompletionRequest) -> ChatCompletionResponse:
        """Handles chat completion requests. Raises NotImplementedError if not supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support chat completions.")

    async def embeddings(self, payload:CohereRequest):
        """Handles requests for embeddings. Raises NotImplementedError if not supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings.")


    @property  
    @abstractmethod 
    def models(self) -> list[LLMModel]:
        pass

