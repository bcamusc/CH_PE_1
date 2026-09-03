"""Cliente asíncrono para OpenAI (usa `AsyncOpenAI`).

Puntos clave de la implementación:

- Nada de esto bloquea el event loop: `AsyncOpenAI` + `await`.
- El SDK actual prefiere `max_completion_tokens` sobre el viejo
  `max_tokens` (que quedó deprecado en los modelos nuevos), así que el
  campo genérico `max_tokens` de la config se mapea a ese parámetro.
- `stream=True` + `async for` para el modo streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from .base import BaseLLMClient, ErrorNormalizado, fusionar_extra
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider


def _usage_openai(usage) -> dict[str, int] | None:
    """Normaliza `CompletionUsage` a un dict simple (o None)."""
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


class OpenAIClient(BaseLLMClient):
    """Implementación concreta para OpenAI."""

    provider = Provider.OPENAI

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        import openai

        self._cliente = openai.AsyncOpenAI(
            api_key=self._api_key(),
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    # ------------------------------------------------------------------
    # Implementación de la interfaz base
    # ------------------------------------------------------------------
    async def _pedir_completos(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> ModelResponse:
        params = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in mensajes],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
        }
        params.update(fusionar_extra(extra, params))

        respuesta = await self._cliente.chat.completions.create(**params)
        eleccion = respuesta.choices[0]

        return ModelResponse(
            provider=self.provider,
            model=self.config.model,
            content=eleccion.message.content or "",
            finish_reason=eleccion.finish_reason,
            usage=_usage_openai(respuesta.usage),
        )

    async def _pedir_stream(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> AsyncIterator[str]:
        params = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in mensajes],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
            "stream": True,
        }
        params.update(fusionar_extra(extra, params))

        # `create(stream=True)` ya devuelve el objeto stream; iteramos.
        stream = await self._cliente.chat.completions.create(**params)
        async for fragmento in stream:
            if not fragmento.choices:
                continue
            delta = fragmento.choices[0].delta
            if delta is not None and delta.content:
                yield delta.content

    def _normalizar_error(self, exc: Exception) -> ErrorNormalizado:
        import openai

        if isinstance(exc, openai.AuthenticationError):
            return ErrorNormalizado(
                "API key de OpenAI inválida o vencida.", "authentication", 401, False
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return ErrorNormalizado(
                "Sin permisos para usar este modelo o recurso.", "permission", 403, False
            )
        if isinstance(exc, openai.NotFoundError):
            return ErrorNormalizado(
                f"Modelo o recurso no encontrado: '{self.config.model}'.",
                "not_found",
                404,
                False,
            )
        if isinstance(exc, openai.BadRequestError):
            return ErrorNormalizado(
                f"Solicitud inválida: {exc}", "bad_request", 400, False
            )
        if isinstance(exc, openai.RateLimitError):
            return ErrorNormalizado(
                "Límite de tasa de OpenAI alcanzado (429).", "rate_limit", 429, True
            )
        if isinstance(exc, openai.APITimeoutError):
            return ErrorNormalizado(
                "Tiempo de espera agotado al contactar a OpenAI.",
                "timeout",
                None,
                True,
            )
        if isinstance(exc, openai.APIConnectionError):
            return ErrorNormalizado(
                f"Error de conexión con OpenAI: {exc}", "connection", None, True
            )
        if isinstance(exc, openai.APIStatusError):
            codigo = getattr(exc, "status_code", None)
            return ErrorNormalizado(
                f"OpenAI respondió con error HTTP {codigo}: {exc}",
                "api_status",
                codigo,
                codigo is not None and (codigo == 429 or codigo >= 500),
            )
        return ErrorNormalizado(
            f"Error inesperado de OpenAI: {exc}", "unknown", None, False
        )
