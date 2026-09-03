"""Cliente de LLM unificado, asíncrono y con streaming.

Permite usar OpenAI o Anthropic bajo una misma interfaz (`BaseLLMClient`)
seleccionando el proveedor con `AsyncLLMManager`.
"""

from .base import BaseLLMClient, LLMClientError
from .manager import AsyncLLMManager
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider

__all__ = [
    "AsyncLLMManager",
    "BaseLLMClient",
    "ChatMessage",
    "LLMClientError",
    "LLMConfig",
    "ModelResponse",
    "Provider",
]

__version__ = "1.0.0"
