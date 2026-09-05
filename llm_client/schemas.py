"""Esquemas Pydantic del cliente LLM unificado.

Validar los datos ANTES de tocar la red (mensajes, temperatura,
max_tokens, API keys...) evita el clásico error de "diccionarios
anidados" cuando recién se parte con los SDKs: si algo está mal, el
error aparece acá, con un mensaje claro, y no en medio de una llamada.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

ROLES_VALIDOS = frozenset({"user", "assistant", "system"})


class Provider(str, Enum):
    """Proveedores de LLM soportados por el cliente unificado."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    KIMI = "kimi"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación: quién lo dice y qué dice."""

    role: str = Field(description="Rol: 'user', 'assistant' o 'system'")
    content: str = Field(min_length=1, description="Texto del mensaje")

    @field_validator("role")
    @classmethod
    def _rol_valido(cls, valor: str) -> str:
        if valor not in ROLES_VALIDOS:
            raise ValueError(
                f"role debe ser uno de {sorted(ROLES_VALIDOS)}; recibido: '{valor}'"
            )
        return valor

    def to_dict(self) -> dict[str, str]:
        """Serialización plana, lista para enviar a cualquier SDK."""
        return {"role": self.role, "content": self.content}


class LLMConfig(BaseModel):
    """Configuración validada de un cliente LLM."""

    provider: Provider = Field(description="Proveedor que se quiere usar")
    model: str = Field(min_length=1, description="Nombre del modelo (p. ej. gpt-4o-mini)")

    # API keys. Solo se exige la del proveedor activo (ver validador).
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    kimi_api_key: Optional[SecretStr] = None

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Creatividad: 0 (determinista) a 2 (más libre).",
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Máximo de tokens a generar.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Reintentos ante errores transitorios (429, red, timeout).",
    )
    timeout: float = Field(default=60.0, gt=0, description="Timeout de cada request (segundos).")

    @model_validator(mode="after")
    def _exigir_api_key_del_proveedor(self) -> "LLMConfig":
        """Cada proveedor exige SU propia API key (nunca la del otro)."""
        campo_key = f"{self.provider.value}_api_key"
        key = getattr(self, campo_key)
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"Falta '{campo_key}': es obligatoria para usar el proveedor "
                f"'{self.provider.value}'."
            )
        return self


class ModelResponse(BaseModel):
    """Respuesta normalizada de cualquier proveedor."""

    provider: Provider = Field(description="Proveedor que respondió")
    model: str = Field(description="Modelo que respondió")
    content: str = Field(default="", description="Texto generado (vacío si hubo error)")
    error: Optional[str] = Field(
        default=None,
        description="Mensaje de error controlado. None = todo bien.",
    )
    finish_reason: Optional[str] = Field(default=None, description="Por qué terminó (stop, length...)")
    usage: Optional[dict[str, int]] = Field(
        default=None,
        description="Tokens usados: prompt_tokens, completion_tokens, total_tokens.",
    )

    @property
    def ok(self) -> bool:
        """True si la llamada terminó sin error."""
        return self.error is None
