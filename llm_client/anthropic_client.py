"""Cliente asíncrono para Anthropic (usa `AsyncAnthropic`).

Particularidades de Anthropic que este cliente resuelve:

- `max_tokens` es OBLIGATORIO en la API (no hay default del lado del SDK).
- El rol "system" va como parámetro separado, NO dentro de `messages`
  (para eso está el helper `_separar_system` de la clase base).
- `messages.stream(...)` se usa como context manager asíncrono y expone
  `text_stream`, un iterador asíncrono de fragmentos de texto.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from .base import BaseLLMClient, ErrorNormalizado, fusionar_extra
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider


def _usage_anthropic(usage) -> dict[str, int] | None:
    """Normaliza el usage de Anthropic (input/output) a un dict simple."""
    if usage is None:
        return None
    entrada = getattr(usage, "input_tokens", 0)
    salida = getattr(usage, "output_tokens", 0)
    return {
        "prompt_tokens": entrada,
        "completion_tokens": salida,
        "total_tokens": entrada + salida,
    }


class AnthropicClient(BaseLLMClient):
    """Implementación concreta para Anthropic."""

    provider = Provider.ANTHROPIC

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        import anthropic

        self._cliente = anthropic.AsyncAnthropic(
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
        system, historial = self._separar_system(mensajes)

        params: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in historial],
        }
        if system:
            params["system"] = system
        params.update(fusionar_extra(extra, params))

        respuesta = await self._cliente.messages.create(**params)

        # `content` es una lista de bloques (text, tool_use, thinking...).
        texto = "".join(
            bloque.text
            for bloque in respuesta.content
            if getattr(bloque, "type", None) == "text"
        )
        return ModelResponse(
            provider=self.provider,
            model=self.config.model,
            content=texto,
            finish_reason=getattr(respuesta, "stop_reason", None),
            usage=_usage_anthropic(getattr(respuesta, "usage", None)),
        )

    async def _pedir_stream(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> AsyncIterator[str]:
        system, historial = self._separar_system(mensajes)

        params: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in historial],
        }
        if system:
            params["system"] = system
        params.update(fusionar_extra(extra, params))

        async with self._cliente.messages.stream(**params) as stream:
            async for texto in stream.text_stream:
                yield texto

    def _normalizar_error(self, exc: Exception) -> ErrorNormalizado:
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            return ErrorNormalizado(
                "API key de Anthropic inválida o vencida.", "authentication", 401, False
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return ErrorNormalizado(
                "Sin permisos para usar este modelo o recurso.", "permission", 403, False
            )
        if isinstance(exc, anthropic.NotFoundError):
            return ErrorNormalizado(
                f"Modelo o recurso no encontrado: '{self.config.model}'.",
                "not_found",
                404,
                False,
            )
        if isinstance(exc, anthropic.BadRequestError):
            return ErrorNormalizado(
                f"Solicitud inválida: {exc}", "bad_request", 400, False
            )
        if isinstance(exc, anthropic.RateLimitError):
            return ErrorNormalizado(
                "Límite de tasa de Anthropic alcanzado (429).", "rate_limit", 429, True
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return ErrorNormalizado(
                "Tiempo de espera agotado al contactar a Anthropic.",
                "timeout",
                None,
                True,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return ErrorNormalizado(
                f"Error de conexión con Anthropic: {exc}", "connection", None, True
            )
        if isinstance(exc, anthropic.APIStatusError):
            codigo = getattr(exc, "status_code", None)
            return ErrorNormalizado(
                f"Anthropic respondió con error HTTP {codigo}: {exc}",
                "api_status",
                codigo,
                codigo is not None and (codigo == 429 or codigo >= 500),
            )
        return ErrorNormalizado(
            f"Error inesperado de Anthropic: {exc}", "unknown", None, False
        )
