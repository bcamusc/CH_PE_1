"""Cliente asíncrono para Kimi (Moonshot AI).

Kimi expone una API *compatible con OpenAI*: mismo formato
`/chat/completions`, mismas respuestas (`choices[0].message.content`,
`usage`...) y las mismas excepciones del SDK de OpenAI.

Por eso `KimiClient` NO reimplementa nada: hereda de `OpenAIClient` y
solo cambia tres atributos de clase:

    provider           -> Provider.KIMI
    base_url           -> "https://api.moonshot.ai/v1" (endpoint de Kimi)
    envia_temperature  -> False (kimi-k3 fija temperature=1.0 y exige
                          omitirla del request)

Además, kimi-k2.6 trae el modo "thinking" ACTIVADO por defecto: razona
hasta agotar el presupuesto de tokens y puede dejar la respuesta
(`content`) vacía. Para que el cliente se comporte igual que con otros
proveedores, lo desactivamos con `thinking: {"type": "disabled"}` (solo
aplica a la familia k2.6; kimi-k3 y kimi-k2.7-code siempre piensan).

Este es el valor de la "compatibilidad de API": sumar un proveedor que
habla el mismo protocolo cuesta pocas líneas y cero lógica duplicada.
"""

from __future__ import annotations

from .openai_client import OpenAIClient
from .schemas import LLMConfig, Provider


class KimiClient(OpenAIClient):
    """Implementación concreta para Kimi (Moonshot AI).

    Todo el comportamiento (normal, streaming, reintentos y errores) lo
    hereda de `OpenAIClient`; aquí solo se ajusta lo que cambia entre
    proveedores compatibles con la API de OpenAI.
    """

    provider = Provider.KIMI
    nombre_proveedor: str = "Kimi"
    base_url: str = "https://api.moonshot.ai/v1"
    envia_temperature: bool = False

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # kimi-k2.6 viene con thinking activado: mejor respuesta directa.
        if config.model.startswith("kimi-k2.6"):
            self.parametros_extra = {"thinking": {"type": "disabled"}}
