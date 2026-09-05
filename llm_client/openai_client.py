"""Cliente asíncrono para OpenAI (usa `AsyncOpenAI`).

Puntos clave de la implementación:

- Nada de esto bloquea el event loop: `AsyncOpenAI` + `await`.
- El SDK actual prefiere `max_completion_tokens` sobre el viejo
  `max_tokens` (que quedó deprecado en los modelos nuevos), así que el
  campo genérico `max_tokens` de la config se mapea a ese parámetro.
- `stream=True` + `async for` para el modo streaming.

Esta clase además sirve de base para proveedores *compatibles con la API
de OpenAI* (hoy: Kimi). Una subclase solo ajusta atributos de clase:
`provider`, `nombre_proveedor`, `base_url` y `envia_temperature`.
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
    """Implementación concreta para OpenAI.

    Atributos de clase que una subclase puede ajustar (p. ej. Kimi):
        provider           -> Provider que identifica a este cliente.
        nombre_proveedor   -> etiqueta usada en los mensajes de error.
        base_url           -> endpoint del SDK (None = el oficial de OpenAI).
        envia_temperature  -> False si el proveedor fija temperature solo
                              (kimi-k3 la fija en 1.0 y exige omitirla).
    parametros_extra   -> dict de params fijos a inyectar en cada request
                          vía extra_body (p. ej. kimi-k2.6 desactiva su
                          "thinking").
    """

    provider = Provider.OPENAI
    nombre_proveedor: str = "OpenAI"
    base_url: str | None = None
    envia_temperature: bool = True
    # Params fijos que una subclase puede inyectar en cada request
    # (p. ej. Kimi desactiva el "thinking" de kimi-k2.6). Se envían
    # como `extra_body` (los kwargs de create() están tipados).
    parametros_extra: dict[str, object] = {}

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        import openai

        parametros: dict[str, object] = {
            "api_key": self._api_key(),
            "max_retries": config.max_retries,
            "timeout": config.timeout,
        }
        if self.base_url is not None:
            parametros["base_url"] = self.base_url
        self._cliente = openai.AsyncOpenAI(**parametros)

    # ------------------------------------------------------------------
    # Implementación de la interfaz base
    # ------------------------------------------------------------------
    async def _pedir_completos(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> ModelResponse:
        params = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in mensajes],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.envia_temperature:
            params["temperature"] = self.config.temperature
        # Los params no estándar van por extra_body: `create()` solo
        # acepta kwargs tipados del SDK, y extra_body los mete al JSON.
        if self.parametros_extra:
            params["extra_body"] = self.parametros_extra
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
            "max_completion_tokens": self.config.max_tokens,
            "stream": True,
        }
        if self.envia_temperature:
            params["temperature"] = self.config.temperature
        # Los params no estándar van por extra_body (ver _pedir_completos).
        if self.parametros_extra:
            params["extra_body"] = self.parametros_extra
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
                f"API key de {self.nombre_proveedor} inválida o vencida.",
                "authentication",
                401,
                False,
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return ErrorNormalizado(
                "Sin permisos para usar este modelo o recurso.", "permission", 403, False
            )
        if isinstance(exc, openai.NotFoundError):
            return ErrorNormalizado(
                f"Modelo o recurso no encontrado en {self.nombre_proveedor}: "
                f"'{self.config.model}'.",
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
                f"Límite de tasa de {self.nombre_proveedor} alcanzado (429).",
                "rate_limit",
                429,
                True,
            )
        if isinstance(exc, openai.APITimeoutError):
            return ErrorNormalizado(
                f"Tiempo de espera agotado al contactar a {self.nombre_proveedor}.",
                "timeout",
                None,
                True,
            )
        if isinstance(exc, openai.APIConnectionError):
            return ErrorNormalizado(
                f"Error de conexión con {self.nombre_proveedor}: {exc}",
                "connection",
                None,
                True,
            )
        if isinstance(exc, openai.APIStatusError):
            codigo = getattr(exc, "status_code", None)
            return ErrorNormalizado(
                f"{self.nombre_proveedor} respondió con error HTTP {codigo}: {exc}",
                "api_status",
                codigo,
                codigo is not None and (codigo == 429 or codigo >= 500),
            )
        return ErrorNormalizado(
            f"Error inesperado de {self.nombre_proveedor}: {exc}",
            "unknown",
            None,
            False,
        )
