"""Cliente base abstracto: la interfaz común para todos los proveedores.

Acá vive la lógica transversal que NO depende del SDK:

- `generate()`  -> respuesta completa. Nunca lanza: ante cualquier error
                  devuelve un `ModelResponse` con el campo `error` poblado.
- `stream()`    -> generador asíncrono de fragmentos de texto. Como no se
                  puede "devolver" un error en medio de un flujo, lanza
                  `LLMClientError` (que el llamador captura con try/except).
- Reintentos con backoff exponencial para errores transitorios
  (rate limit 429, caídas de red, timeouts).

Las subclases solo implementan lo específico de cada SDK:
`_pedir_completos()`, `_pedir_stream()` y `_normalizar_error()`.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMClientError(Exception):
    """Error controlado del cliente (para que nada "crashee" el programa).

    Atributos:
        provider:     proveedor que falló.
        tipo:         categoría del error (rate_limit, authentication, ...).
        status_code:  código HTTP si la API alcanzó a responder.
    """

    def __init__(
        self,
        mensaje: str,
        *,
        provider: Provider | None = None,
        tipo: str = "unknown",
        status_code: int | None = None,
    ) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.provider = provider
        self.tipo = tipo
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code is not None:
            return f"{self.mensaje} (HTTP {self.status_code})"
        return self.mensaje


@dataclass
class ErrorNormalizado:
    """Versión uniforme de un error del SDK, para decidir si se reintenta."""

    mensaje: str
    tipo: str
    status_code: int | None = None
    recuperable: bool = False


def fusionar_extra(extra: dict, fijados: dict) -> dict:
    """Filtra overrides por llamada para no pisar parámetros ya fijados.

    Por ejemplo, `model`, `temperature` y `max_tokens` salen de la config:
    si el llamador los pasa por `**extra`, se ignoran silenciosamente.
    """
    return {k: v for k, v in extra.items() if k not in fijados}


class BaseLLMClient(ABC):
    """Interfaz asíncrona común para clientes de LLM.

    Todo el resto del código habla SOLO con esta abstracción: cambiar de
    proveedor no obliga a tocar ni una línea de quien consume el cliente.
    """

    provider: Provider

    def __init__(self, config: LLMConfig) -> None:
        if config.provider is not self.provider:
            raise ValueError(
                f"{type(self).__name__} solo admite provider "
                f"'{self.provider.value}', pero recibió '{config.provider.value}'."
            )
        self.config = config

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    async def generate(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> ModelResponse:
        """Genera una respuesta completa (sin streaming).

        Garantía: nunca lanza. Ante cualquier error devuelve un
        `ModelResponse` con `error` poblado y `content` vacío.
        """
        try:
            return await self._ejecutar_con_reintentos(
                lambda: self._pedir_completos(mensajes, **extra),
                descripcion=f"generate/{self.provider.value}",
            )
        except LLMClientError as exc:
            logger.error("generate/%s falló: %s", self.provider.value, exc)
            return ModelResponse(
                provider=self.provider,
                model=self.config.model,
                error=str(exc),
            )

    async def stream(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> AsyncIterator[str]:
        """Generador asíncrono de fragmentos de texto (token a token).

        Acá no se puede "devolver" un error a mitad de flujo, así que ante
        un fallo se lanza `LLMClientError`. Patrón de uso:

            async for trozo in cliente.stream(mensajes):
                print(trozo, end="")
        """
        try:
            async for fragmento in self._pedir_stream(mensajes, **extra):
                yield fragmento
        except LLMClientError:
            raise
        except Exception as exc:  # red que cae a mitad del stream
            e = self._normalizar_error(exc)
            logger.error("stream/%s falló: %s", self.provider.value, e.mensaje)
            raise self._a_error_controlado(e) from exc

    # ------------------------------------------------------------------
    # Reintentos y normalización de errores
    # ------------------------------------------------------------------
    async def _ejecutar_con_reintentos(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        *,
        descripcion: str,
    ) -> T:
        """Ejecuta `coro_factory()` reintentando errores transitorios.

        Estrategia: backoff exponencial (1s, 2s, 4s...) hasta
        `config.max_retries` reintentos. Los errores no recuperables
        (API key inválida, request mal formado, 4xx...) fallan al toque.
        """
        for intento in range(self.config.max_retries + 1):
            try:
                return await coro_factory()
            except Exception as exc:  # noqa: BLE001 - se normaliza acá, en el borde
                e = self._normalizar_error(exc)
                if not e.recuperable or intento >= self.config.max_retries:
                    raise self._a_error_controlado(e) from exc
                espera = 2.0**intento
                logger.warning(
                    "Error transitorio en %s (intento %d/%d): %s. Reintento en %.1fs",
                    descripcion,
                    intento + 1,
                    self.config.max_retries + 1,
                    e.mensaje,
                    espera,
                )
                await asyncio.sleep(espera)
        raise LLMClientError(
            "No se pudo completar la operación.", provider=self.provider
        )

    def _a_error_controlado(self, e: ErrorNormalizado) -> LLMClientError:
        return LLMClientError(
            e.mensaje, provider=self.provider, tipo=e.tipo, status_code=e.status_code
        )

    @abstractmethod
    def _normalizar_error(self, exc: Exception) -> ErrorNormalizado:
        """Traduce una excepción del SDK a un `ErrorNormalizado`."""

    # ------------------------------------------------------------------
    # Implementación específica del SDK (abstracta)
    # ------------------------------------------------------------------
    @abstractmethod
    async def _pedir_completos(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> ModelResponse:
        """Llamada de red: respuesta completa."""

    @abstractmethod
    def _pedir_stream(
        self, mensajes: Sequence[ChatMessage], **extra: object
    ) -> AsyncIterator[str]:
        """Llamada de red: stream de fragmentos de texto."""

    # ------------------------------------------------------------------
    # Helpers compartidos
    # ------------------------------------------------------------------
    def _api_key(self) -> str:
        """API key del proveedor actual (el schema ya la validó)."""
        campo = f"{self.provider.value}_api_key"
        key = getattr(self.config, campo)
        return key.get_secret_value() if key else ""

    @staticmethod
    def _separar_system(
        mensajes: Sequence[ChatMessage],
    ) -> tuple[str | None, list[ChatMessage]]:
        """Anthropic recibe el rol 'system' como parámetro aparte.

        Extraemos los mensajes system (y los concatenamos si hay varios)
        para poder mandarlos por separado en ese proveedor.
        """
        system = "\n".join(m.content for m in mensajes if m.role == "system") or None
        resto = [m for m in mensajes if m.role != "system"]
        return system, resto
