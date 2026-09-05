"""Tests del cliente LLM unificado SIN red y SIN API keys reales.

Estrategia: reemplazamos `_cliente` (el objeto del SDK) por un doble con
`unittest.mock` que devuelve respuestas falsas, así validamos normal +
streaming + manejo de errores sin hacer ninguna llamada HTTP.

Correr:  python -m unittest discover -s tests -v
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import SecretStr, ValidationError

from llm_client.anthropic_client import AnthropicClient
from llm_client.base import LLMClientError
from llm_client.kimi_client import KimiClient
from llm_client.manager import AsyncLLMManager
from llm_client.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, Provider


# ----------------------------------------------------------------------
# Helpers de pruebas
# ----------------------------------------------------------------------
def _config(provider: Provider, **cambios) -> LLMConfig:
    datos: dict = {
        "provider": provider,
        "model": "modelo-de-prueba",
        "openai_api_key": SecretStr("sk-prueba") if provider is Provider.OPENAI else None,
        "anthropic_api_key": SecretStr("sk-ant-prueba") if provider is Provider.ANTHROPIC else None,
        "kimi_api_key": SecretStr("sk-kimi-prueba") if provider is Provider.KIMI else None,
    }
    datos.update(cambios)
    return LLMConfig(**datos)


def _respuesta_openai():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="La entropía mide el desorden."),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _respuesta_anthropic():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="La entropía mide el desorden.")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=8, output_tokens=4),
    )


def _fake_openai_cliente(respuesta_o_stream):
    """Cliente OpenAI falso: `.chat.completions.create` es un AsyncMock."""
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=respuesta_o_stream))
        )
    )


def _fake_anthropic_cliente(respuesta=None, stream_manager=None):
    messages = {}
    if respuesta is not None:
        messages["create"] = AsyncMock(return_value=respuesta)
    if stream_manager is not None:
        # OJO: messages.stream() NO es async en sí mismo; devuelve un
        # context manager que se usa con `async with`. Por eso lambda.
        messages["stream"] = lambda *args, **kwargs: stream_manager
    return SimpleNamespace(messages=SimpleNamespace(**messages))


def _respuesta_error(status: int):
    """Simula una httpx.Response para construir excepciones del SDK."""
    return SimpleNamespace(
        request=SimpleNamespace(url="https://api.prueba/v1/chat/completions"),
        status_code=status,
        headers={"x-request-id": "test-123"},
    )


class _FakeStreamAnthropic:
    """Simula el context manager asíncrono que devuelve messages.stream()."""

    def __init__(self, partes: list[str]):
        self._partes = partes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @property
    def text_stream(self):
        async def _gen():
            for p in self._partes:
                yield p

        return _gen()


def _gen_openai(fragmentos: list[str]):
    """Generador async de chunks estilo ChatCompletionChunk."""

    async def _gen():
        for f in fragmentos:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=f))]
            )

    return _gen()


# ----------------------------------------------------------------------
# Tests de schemas.py (validación Pydantic)
# ----------------------------------------------------------------------
class TestSchemas(unittest.TestCase):
    def test_rol_invalido(self):
        with self.assertRaises(ValidationError):
            ChatMessage(role="robot", content="hola")

    def test_rol_valido(self):
        m = ChatMessage(role="system", content="Sé breve.")
        self.assertEqual(m.to_dict(), {"role": "system", "content": "Sé breve."})

    def test_temperatura_fuera_de_rango(self):
        with self.assertRaises(ValidationError):
            _config(Provider.OPENAI, temperature=3.0)

    def test_max_tokens_negativo(self):
        with self.assertRaises(ValidationError):
            _config(Provider.OPENAI, max_tokens=0)

    def test_falta_api_key_del_proveedor(self):
        with self.assertRaises(ValidationError):
            LLMConfig(
                provider=Provider.OPENAI,
                model="gpt-4o-mini",
                openai_api_key=None,
                anthropic_api_key=SecretStr("sk-ant-prueba"),
            )

    def test_api_key_del_otro_proveedor_no_sirve(self):
        # Tener la key de Anthropic NO habilita OpenAI.
        with self.assertRaises(ValidationError):
            LLMConfig(
                provider=Provider.OPENAI,
                model="gpt-4o-mini",
                anthropic_api_key=SecretStr("sk-ant-prueba"),
            )

    def test_falta_api_key_de_kimi(self):
        with self.assertRaises(ValidationError):
            LLMConfig(
                provider=Provider.KIMI,
                model="kimi-k2.6",
                kimi_api_key=None,
            )

    def test_api_key_de_otro_proveedor_no_habilita_kimi(self):
        # Tener la key de OpenAI NO habilita Kimi.
        with self.assertRaises(ValidationError):
            LLMConfig(
                provider=Provider.KIMI,
                model="kimi-k2.6",
                openai_api_key=SecretStr("sk-prueba"),
            )


# ----------------------------------------------------------------------
# Tests del cliente OpenAI (modo normal + streaming + errores)
# ----------------------------------------------------------------------
class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    async def test_generate_modo_normal(self):
        cliente = OpenAIClient(_config(Provider.OPENAI))
        cliente._cliente = _fake_openai_cliente(_respuesta_openai())

        respuesta = await cliente.generate(
            [ChatMessage(role="user", content="¿Qué es la entropía?")]
        )

        self.assertTrue(respuesta.ok)
        self.assertIn("entropía", respuesta.content)
        self.assertEqual(respuesta.finish_reason, "stop")
        self.assertEqual(respuesta.usage["total_tokens"], 15)
        self.assertEqual(respuesta.provider, Provider.OPENAI)

        # Verifica que la llamada usó los parámetros de la config.
        llamada = cliente._cliente.chat.completions.create.await_args.kwargs
        self.assertEqual(llamada["model"], "modelo-de-prueba")
        self.assertEqual(llamada["max_completion_tokens"], 1024)
        self.assertEqual(llamada["messages"][0]["role"], "user")

    async def test_streaming(self):
        cliente = OpenAIClient(_config(Provider.OPENAI))
        cliente._cliente = _fake_openai_cliente(
            _gen_openai(["La ", "entropía ", "es ", "una ", "medida."])
        )

        partes = []
        async for trozo in cliente.stream(
            [ChatMessage(role="user", content="Hola")]
        ):
            partes.append(trozo)

        self.assertEqual("".join(partes), "La entropía es una medida.")

    async def test_rate_limit_devuelve_error_controlado_en_generate(self):
        from openai import RateLimitError

        # side_effect LANZA la excepción (return_value solo la devolvería).
        create = AsyncMock(
            side_effect=RateLimitError(
                "rate limit", response=_respuesta_error(429), body=None
            )
        )
        cliente = OpenAIClient(_config(Provider.OPENAI, max_retries=0))
        cliente._cliente = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        respuesta = await cliente.generate(
            [ChatMessage(role="user", content="Hola")]
        )

        self.assertIsInstance(respuesta, ModelResponse)
        self.assertFalse(respuesta.ok)
        self.assertIn("Límite de tasa", respuesta.error or "")

    async def test_autenticacion_no_se_reintenta(self):
        from openai import AuthenticationError

        cliente = OpenAIClient(_config(Provider.OPENAI, max_retries=3))
        create = AsyncMock(
            side_effect=AuthenticationError(
                "bad key", response=_respuesta_error(401), body=None
            )
        )
        cliente._cliente = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        respuesta = await cliente.generate(
            [ChatMessage(role="user", content="Hola")]
        )

        self.assertFalse(respuesta.ok)
        self.assertIn("API key", respuesta.error or "")
        # Un 401 NO se reintenta: solo hubo un intento.
        self.assertEqual(create.await_count, 1)


# ----------------------------------------------------------------------
# Tests del cliente Anthropic
# ----------------------------------------------------------------------
class TestAnthropicClient(unittest.IsolatedAsyncioTestCase):
    async def test_generate_con_system_separado(self):
        cliente = AnthropicClient(_config(Provider.ANTHROPIC))
        cliente._cliente = _fake_anthropic_cliente(respuesta=_respuesta_anthropic())

        respuesta = await cliente.generate(
            [
                ChatMessage(role="system", content="Sé muy breve."),
                ChatMessage(role="user", content="Hola"),
            ]
        )

        self.assertTrue(respuesta.ok)
        self.assertIn("entropía", respuesta.content)
        self.assertEqual(respuesta.usage["total_tokens"], 12)

        # El system debe ir como parámetro aparte y NO dentro de messages.
        llamada = cliente._cliente.messages.create.await_args.kwargs
        self.assertEqual(llamada["system"], "Sé muy breve.")
        self.assertEqual(len(llamada["messages"]), 1)
        self.assertEqual(llamada["messages"][0]["role"], "user")
        self.assertIn("max_tokens", llamada)  # Anthropic lo exige sí o sí.

    async def test_streaming_con_text_stream(self):
        cliente = AnthropicClient(_config(Provider.ANTHROPIC))
        cliente._cliente = _fake_anthropic_cliente(
            stream_manager=_FakeStreamAnthropic(["Hola ", "mundo ", "async."])
        )

        partes = []
        async for trozo in cliente.stream(
            [ChatMessage(role="user", content="Hola")]
        ):
            partes.append(trozo)

        self.assertEqual("".join(partes), "Hola mundo async.")

    async def test_rate_limit_en_stream_lanza_error_controlado(self):
        from anthropic import RateLimitError

        class _StreamQueFalla:
            async def __aenter__(self):
                raise RateLimitError(
                    "rate", response=_respuesta_error(429), body=None
                )

            async def __aexit__(self, *args):
                return False

        cliente = AnthropicClient(_config(Provider.ANTHROPIC))
        cliente._cliente = _fake_anthropic_cliente(stream_manager=_StreamQueFalla())

        with self.assertRaises(LLMClientError) as ctx:
            async for _ in cliente.stream(
                [ChatMessage(role="user", content="Hola")]
            ):
                pass

        self.assertIn("Límite de tasa", str(ctx.exception))


# ----------------------------------------------------------------------
# Tests del cliente Kimi (hereda de OpenAIClient: API compatible)
# ----------------------------------------------------------------------
class TestKimiClient(unittest.IsolatedAsyncioTestCase):
    """Kimi no duplica lógica: reutiliza OpenAIClient con otros atributos."""

    async def test_apunta_al_endpoint_de_moonshot(self):
        # Construir AsyncOpenAI no toca la red; verificamos el base_url.
        # OJO: el SDK normaliza la URL (objeto URL + barra final).
        cliente = KimiClient(_config(Provider.KIMI))
        self.assertEqual(
            str(cliente._cliente.base_url).rstrip("/"),
            "https://api.moonshot.ai/v1",
        )

    async def test_generate_modo_normal(self):
        cliente = KimiClient(_config(Provider.KIMI))
        cliente._cliente = _fake_openai_cliente(_respuesta_openai())

        respuesta = await cliente.generate(
            [ChatMessage(role="user", content="¿Qué es la entropía?")]
        )

        self.assertTrue(respuesta.ok)
        self.assertEqual(respuesta.provider, Provider.KIMI)
        self.assertIn("entropía", respuesta.content)
        self.assertEqual(respuesta.usage["total_tokens"], 15)

        # Reutiliza el formato de OpenAI, incluido max_completion_tokens.
        llamada = cliente._cliente.chat.completions.create.await_args.kwargs
        self.assertEqual(llamada["model"], "modelo-de-prueba")
        self.assertEqual(llamada["max_completion_tokens"], 1024)
        # Kimi (kimi-k3) fija temperature: no debe ir en el request.
        self.assertNotIn("temperature", llamada)

    async def test_streaming(self):
        cliente = KimiClient(_config(Provider.KIMI))
        cliente._cliente = _fake_openai_cliente(
            _gen_openai(["Hola ", "Kimi ", "async."])
        )

        partes = []
        async for trozo in cliente.stream(
            [ChatMessage(role="user", content="Hola")]
        ):
            partes.append(trozo)

        self.assertEqual("".join(partes), "Hola Kimi async.")

    async def test_desactiva_thinking_en_kimi_k26(self):
        # kimi-k2.6 piensa por defecto y puede dejar content vacío:
        # el cliente debe mandar thinking disabled (vía extra_body).
        cliente = KimiClient(_config(Provider.KIMI, model="kimi-k2.6"))
        cliente._cliente = _fake_openai_cliente(_respuesta_openai())

        await cliente.generate([ChatMessage(role="user", content="Hola")])

        llamada = cliente._cliente.chat.completions.create.await_args.kwargs
        self.assertEqual(llamada["extra_body"], {"thinking": {"type": "disabled"}})

    async def test_no_envia_thinking_en_kimi_k3(self):
        # kimi-k3 siempre piensa: no acepta el param thinking.
        cliente = KimiClient(_config(Provider.KIMI, model="kimi-k3"))
        cliente._cliente = _fake_openai_cliente(_respuesta_openai())

        await cliente.generate([ChatMessage(role="user", content="Hola")])

        llamada = cliente._cliente.chat.completions.create.await_args.kwargs
        self.assertNotIn("extra_body", llamada)

    async def test_error_404_habla_de_kimi_y_no_se_reintenta(self):
        from openai import NotFoundError

        cliente = KimiClient(_config(Provider.KIMI, max_retries=3))
        create = AsyncMock(
            side_effect=NotFoundError(
                "modelo no existe", response=_respuesta_error(404), body=None
            )
        )
        cliente._cliente = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        respuesta = await cliente.generate(
            [ChatMessage(role="user", content="Hola")]
        )

        self.assertFalse(respuesta.ok)
        self.assertIn("Kimi", respuesta.error or "")
        self.assertEqual(create.await_count, 1)  # un 404 no se reintenta


# ----------------------------------------------------------------------
# Tests del manager (selección por configuración / entorno)
# ----------------------------------------------------------------------
class TestManager(unittest.TestCase):
    def test_desde_env_openai(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-prueba"},
            clear=False,
        ):
            manager = AsyncLLMManager.desde_env()
        self.assertEqual(manager.provider, Provider.OPENAI)
        self.assertIsInstance(manager._cliente, OpenAIClient)

    def test_desde_env_anthropic(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "sk-ant-prueba",
                "LLM_MODEL": "claude-3-7-sonnet-latest",
            },
            clear=False,
        ):
            manager = AsyncLLMManager.desde_env()
        self.assertEqual(manager.provider, Provider.ANTHROPIC)
        self.assertEqual(manager.model, "claude-3-7-sonnet-latest")

    def test_desde_env_kimi(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "kimi", "KIMI_API_KEY": "sk-kimi-prueba"},
            clear=False,
        ):
            manager = AsyncLLMManager.desde_env()
        self.assertEqual(manager.provider, Provider.KIMI)
        self.assertIsInstance(manager._cliente, KimiClient)

    def test_desde_env_falla_sin_api_key(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                AsyncLLMManager.desde_env()

    def test_desde_env_provider_invalido(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=False):
            with self.assertRaises(ValueError):
                AsyncLLMManager.desde_env()


if __name__ == "__main__":
    unittest.main()
