"""Script de demostración del cliente LLM unificado.

Prueba las DOS formas de consumir el cliente con la misma pregunta
("¿Qué es la entropía?"):

    1) Modo normal   -> `manager.generate()`  imprime la respuesta completa.
    2) Modo streaming-> `manager.stream()`    imprime los fragmentos al llegar.

Ejecutar:

    python main.py

Antes, copia `.env.example` a `.env` y pega tu API key del proveedor
que elijas en `LLM_PROVIDER`.
"""

import asyncio
import sys

from dotenv import load_dotenv

from llm_client.base import LLMClientError
from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage

PREGUNTA = "¿Qué es la entropía? Explícalo en máximo 3 líneas."


def construir_manager() -> AsyncLLMManager:
    """Crea el manager leyendo el .env; con errores claros si falta algo."""
    try:
        return AsyncLLMManager.desde_env()
    except ValueError as exc:
        print("No se pudo crear el cliente:", exc)
        print()
        print("Revisa tu archivo .env:")
        print("  1) Copia .env.example a .env")
        print("  2) Define LLM_PROVIDER=openai, anthropic o kimi")
        print("  3) Pega la API key correspondiente (OPENAI_API_KEY,")
        print("     ANTHROPIC_API_KEY o KIMI_API_KEY)")
        sys.exit(1)


async def probar_modo_normal(manager: AsyncLLMManager, mensajes: list[ChatMessage]) -> None:
    print("=== 1) Modo normal: respuesta completa ===")
    respuesta = await manager.generate(mensajes)
    if respuesta.ok:
        print(f"[{respuesta.provider.value} · {respuesta.model}]")
        print(respuesta.content)
        print(
            f"\n-> finish_reason: {respuesta.finish_reason or 'n/a'} | "
            f"tokens: {respuesta.usage}"
        )
    else:
        # El error llegó como dato, no como crash (ver manejo en la base).
        print("Error controlado:", respuesta.error)


async def probar_streaming(manager: AsyncLLMManager, mensajes: list[ChatMessage]) -> None:
    print("\n=== 2) Modo streaming: token a token ===")
    try:
        async for fragmento in manager.stream(mensajes):
            print(fragmento, end="", flush=True)
        print("\n-> Streaming terminado sin errores.")
    except LLMClientError as exc:
        print("\nError controlado durante el streaming:", exc)


async def main() -> None:
    manager = construir_manager()
    print(f"Cliente activo: {manager.provider.value} | modelo: {manager.model}\n")

    mensajes = [ChatMessage(role="user", content=PREGUNTA)]

    await probar_modo_normal(manager, mensajes)
    await probar_streaming(manager, mensajes)


if __name__ == "__main__":
    # python-dotenv carga el .env (si existe) a las variables de entorno.
    load_dotenv()
    asyncio.run(main())
