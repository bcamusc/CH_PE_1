"""`AsyncLLMManager`: una sola clase para hablar con cualquier proveedor.

Selecciona el cliente concreto (OpenAI, Anthropic o Kimi) según la
configuración: o bien la pasas explícita (`LLMConfig`) o bien la arma
desde variables de entorno con `AsyncLLMManager.desde_env()`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .kimi_client import KimiClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider

# Modelos sugeridos cuando no se define LLM_MODEL en el .env
MODELOS_SUGERIDOS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-haiku-4-5",
    Provider.KIMI: "kimi-k2.6",
}

_CLIENTES: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.KIMI: KimiClient,
}


class AsyncLLMManager:
    """Capa única para OpenAI, Anthropic y Kimi, elegidos por configuración.

    Uso típico:

        manager = AsyncLLMManager.desde_env()          # lee el .env
        respuesta = await manager.generate(
            [ChatMessage(role="user", content="Hola")]
        )
        async for trozo in manager.stream(mensajes):
            print(trozo, end="")
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Intercambiabilidad: se instancia la clase del proveedor elegido.
        self._cliente: BaseLLMClient = _CLIENTES[config.provider](config)

    # ------------------------------------------------------------------
    # Información básica
    # ------------------------------------------------------------------
    @property
    def provider(self) -> Provider:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    # ------------------------------------------------------------------
    # API pública (delega en el cliente concreto)
    # ------------------------------------------------------------------
    async def generate(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> ModelResponse:
        """Respuesta completa. Nunca lanza: ante error, `ModelResponse.error`."""
        return await self._cliente.generate(mensajes, **extra)

    async def stream(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> AsyncIterator[str]:
        """Streaming de fragmentos. Ante error lanza `LLMClientError`."""
        async for fragmento in self._cliente.stream(mensajes, **extra):
            yield fragmento

    # ------------------------------------------------------------------
    # Construcción desde variables de entorno
    # ------------------------------------------------------------------
    @classmethod
    def desde_env(
        cls,
        provider: Provider | str | None = None,
        model: str | None = None,
    ) -> "AsyncLLMManager":
        """Construye el manager leyendo el entorno (ideal tras `load_dotenv()`).

        Variables usadas:
            LLM_PROVIDER       "openai", "anthropic" o "kimi" (default: openai)
            OPENAI_API_KEY     key de OpenAI
            ANTHROPIC_API_KEY  key de Anthropic
            KIMI_API_KEY       key de Kimi (Moonshot AI)
            LLM_MODEL          modelo (opcional; hay uno sugerido)
            LLM_TEMPERATURE    temperatura (default 0.7)
            LLM_MAX_TOKENS     máx. tokens (default 1024)

        Si el proveedor activo no tiene su API key, `LLMConfig` lanza un
        `ValueError` con un mensaje claro.
        """
        nombre = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        try:
            prov = Provider(nombre)
        except ValueError as exc:
            raise ValueError(
                f"LLM_PROVIDER inválido: '{nombre}'. Valores válidos: "
                f"{[p.value for p in Provider]}."
            ) from exc

        modelo = (model or os.getenv("LLM_MODEL") or "").strip() or MODELOS_SUGERIDOS[prov]

        api_keys = {
            f"{p.value}_api_key": (os.getenv(f"{p.value.upper()}_API_KEY") or "").strip() or None
            for p in Provider
        }

        config = LLMConfig(
            provider=prov,
            model=modelo,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            **api_keys,
        )
        return cls(config)
