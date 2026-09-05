# Unified Async LLM Client

Cliente de LLM **unificado**, **asíncrono** y con **streaming** para
OpenAI, Anthropic y Kimi (Moonshot AI), en Python 3.12. Implementa una
interfaz común (`BaseLLMClient`) con validación Pydantic, reintentos con
backoff y errores controlados (nada "crashea").

> Pre-entrega 1 — Curso de LLMs. Repositorio con el código de los
> clientes async, esquemas Pydantic, `.env.example`, `main.py` de prueba
> y este `README.md`.

---

## 1. Qué hay en el repositorio

```
.
├── llm_client/               # El paquete con toda la lógica
│   ├── __init__.py           #   exporta la API pública
│   ├── schemas.py            #   Pydantic: ChatMessage, LLMConfig, ModelResponse
│   ├── base.py               #   BaseLLMClient (ABC) + errores + reintentos
│   ├── openai_client.py      #   OpenAIClient  (AsyncOpenAI)
│   ├── kimi_client.py        #   KimiClient (hereda de OpenAIClient)
│   ├── anthropic_client.py   #   AnthropicClient (AsyncAnthropic)
│   └── manager.py            #   AsyncLLMManager: elige proveedor por config
├── main.py                   # Demo: pregunta normal + streaming
├── tests/
│   └── test_clients.py       # Tests sin red (mocks de los SDKs)
├── .env.example              # Plantilla de variables de entorno
├── requirements.txt
└── README.md
```

## 2. Requisitos

- Python **3.12**
- Una API key del proveedor que quieras usar (OpenAI, Anthropic o Kimi)

## 3. Instalación

```bash
# 1) Entorno virtual con Python 3.12
python3.12 -m venv .venv

# 2) Activar (Windows)
.venv\Scripts\activate
#    o (macOS/Linux)
source .venv/bin/activate

# 3) Dependencias
pip install -r requirements.txt
```

## 4. Variables de entorno

Copia la plantilla y completa tus keys:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| Variable             | Obligatoria | Descripción                                          |
|----------------------|-------------|------------------------------------------------------|
| `LLM_PROVIDER`       | Sí          | Proveedor activo: `openai`, `anthropic` o `kimi`     |
| `OPENAI_API_KEY`     | Según el proveedor | Key de OpenAI (`sk-...`)                     |
| `ANTHROPIC_API_KEY`  | Según el proveedor | Key de Anthropic (`sk-ant-...`)              |
| `KIMI_API_KEY`       | Según el proveedor | Key de Kimi de Moonshot AI (`sk-...`)       |
| `LLM_MODEL`          | No          | Modelo (vacío = sugerido: `gpt-4o-mini` / `claude-haiku-4-5` / `kimi-k2.6`) |
| `LLM_TEMPERATURE`    | No          | Creatividad, `0.0` a `2.0` (default `0.7`)           |
| `LLM_MAX_TOKENS`     | No          | Máx. tokens a generar (default `1024`)               |

> El archivo `.env` **no se sube** al repositorio (está en `.gitignore`).
> Solo se sube `.env.example`.

## 5. Cómo ejecutar la demo

```bash
python main.py
```

La demo pregunta *"¿Qué es la entropía?"* de dos formas:

1. **Modo normal** — `manager.generate(...)`: espera la respuesta completa
   e imprime el texto, el `finish_reason` y el uso de tokens.
2. **Modo streaming** — `manager.stream(...)`: imprime los fragmentos
   apenas van llegando de la API (`async for` sobre el stream).

### Cómo correr los tests (sin red y sin API keys)

```bash
python -m unittest discover -s tests -v
```

Los tests reemplazan a los SDKs con *mocks*, así que validan toda la
lógica (normal + streaming + errores) sin gastar tokens ni exponer keys.

## 6. Cómo usarlo en tu propio código

```python
import asyncio
from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage

load_dotenv()

async def ejemplo():
    manager = AsyncLLMManager.desde_env()   # respeta LLM_PROVIDER del .env

    mensajes = [ChatMessage(role="user", content="Hola, ¿cómo estás?")]

    # 1) Respuesta completa (nunca lanza; revisa .ok / .error)
    respuesta = await manager.generate(mensajes)
    if respuesta.ok:
        print(respuesta.content)

    # 2) Streaming
    async for trozo in manager.stream(mensajes):
        print(trozo, end="")

asyncio.run(ejemplo())
```

Para cambiar de proveedor basta cambiar `LLM_PROVIDER` en el `.env`:
**ninguna línea de tu código cambia** (intercambiabilidad).

También puedes construir el manager con configuración explícita:

```python
from llm_client import LLMConfig, Provider, AsyncLLMManager
from pydantic import SecretStr

config = LLMConfig(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    openai_api_key=SecretStr("sk-..."),
    temperature=0.3,
    max_tokens=512,
)
manager = AsyncLLMManager(config)
```

## 7. Decisiones de diseño (y errores comunes que evita)

### 7.1 Intercambiabilidad
`BaseLLMClient` es una clase abstracta (`ABC`) que define la interfaz:
`generate()` y `stream()`. `OpenAIClient`, `KimiClient` y
`AnthropicClient` la implementan y el `AsyncLLMManager` instancia la
clase correcta según `LLMConfig.provider`. Quien consume el cliente
**solo conoce la abstracción**.

### 7.2 Asincronía (no bloquees el event loop)
Todo usa los SDKs asíncronos: `AsyncOpenAI` y `AsyncAnthropic`, siempre
con `await`. Kimi usa el SDK de OpenAI apuntando a su propio endpoint
(API compatible), así que también corre sobre `AsyncOpenAI`. Un error
clásico es usar el cliente síncrono (`OpenAI` en vez de `AsyncOpenAI`)
dentro de una función `async`: eso bloquea todo el programa mientras el
modelo "piensa".

### 7.3 Streaming con `yield`
- **OpenAI y Kimi**: `create(..., stream=True)` devuelve un `AsyncStream`
  que se recorre con `async for chunk`; el texto está en
  `chunk.choices[0].delta.content`.
- **Anthropic**: `messages.stream(...)` se usa como *context manager*
  asíncrono y expone `text_stream`, un iterador asíncrono de fragmentos.

En todos los casos el método `_pedir_stream()` es un **generador
asíncrono** (`async def` + `yield`), así que los fragmentos se entregan
al llamador en cuanto llegan.

### 7.4 Validación con Pydantic (antes de tocar la red)
`schemas.py` valida todo lo que entra y lo que sale:
- `ChatMessage.role` solo acepta `user | assistant | system`.
- `LLMConfig.temperature` debe estar en `[0, 2]`, `max_tokens > 0`.
- `LLMConfig` exige la API key del proveedor activo (con `SecretStr`
  para no imprimir keys por accidente).
- `ModelResponse` normaliza la salida de todos los proveedores y agrega
  el campo `error` opcional.

Esto evita el típico "error de diccionarios anidados": si un mensaje
está mal formado, el error aparece **aquí**, con mensaje claro.

### 7.5 Errores controlados y reintentos
- Errores **no recuperables** (API key inválida `401`, request malo
  `400`, sin permisos `403`, modelo inexistente `404`) fallan de
  inmediato con un error claro.
- Errores **transitorios** (rate limit `429`, caídas de red, timeouts,
  `5xx`) se **reintentan** con backoff exponencial
  (1s, 2s, 4s...) hasta `max_retries` veces.
- `generate()` **nunca lanza**: devuelve un `ModelResponse` con el campo
  `error` poblado. En `stream()` no se puede "devolver" un error a mitad
  de flujo, así que se lanza `LLMClientError` (se captura alrededor del
  `async for`).

### 7.6 Diferencias de API resueltas internamente
| Aspecto                | OpenAI                          | Anthropic                          |
|------------------------|---------------------------------|------------------------------------|
| Rol `system`           | Va dentro de `messages`         | Parámetro `system` aparte          |
| Máx. tokens            | `max_completion_tokens`         | `max_tokens` (obligatorio)         |
| Texto de la respuesta  | `choices[0].message.content`    | Bloques `content` con `type="text"` |
| Streaming              | `stream=True` + `async for`     | `messages.stream(...)` + `text_stream` |

### 7.7 Kimi: un proveedor "compatible con OpenAI" sin duplicar código
Kimi (Moonshot AI) habla el **mismo protocolo** que OpenAI, así que
`KimiClient` **hereda de `OpenAIClient`** y solo cambia atributos de
clase: `provider`, `base_url` (`https://api.moonshot.ai/v1`),
`envia_temperature=False` (kimi-k3 fija `temperature=1.0` y exige
omitirla del request) y `parametros_extra` (kimi-k2.6 trae el modo
"thinking" activado, que puede agotar los tokens razonando y dejar la
respuesta vacía, así que se desactiva con `thinking: {"type":
"disabled"}`). Todo lo demás —parsing, streaming, reintentos y errores—
se reutiliza tal cual. Si aparece otro proveedor compatible (p. ej.
DeepSeek, Groq, Mistral), basta una subclase similar con pocas líneas.

## 8. Notas de versiones probadas

El código se probó con: `openai 3.7.0`, `anthropic 1.3.0`,
`pydantic 2.13.5`, Python 3.12.10. El `requirements.txt` pide versiones
mínimas compatibles con el código (p. ej. `openai>=2.0.0`, donde
`chat.completions` y `max_completion_tokens` ya existen).
