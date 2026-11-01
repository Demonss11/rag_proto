"""Лаунчер Chainlit в обход CLI:  python serve.py [порт]

Штатный `chainlit run` вызывает `nest_asyncio.apply()` (chainlit/cli/__init__.py),
который на Python 3.14 ломает `asyncio.Task.set_name()` -> `current_task()`
возвращает None -> `sniffio` не видит event loop -> `anyio.to_thread.run_sync`
падает с `NoEventLoopError`. Из-за этого `FileResponse` отдаёт 500 на все
/assets/*, /favicon. nest_asyncio нужен только во встраиваемых средах
(Jupyter), uvicorn в нём не нуждается, поэтому здесь он не применяется.
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

from chainlit.auth import ensure_jwt_secret
from chainlit.config import config, init_config, load_module
from chainlit.markdown import init_markdown


DEFAULT_PORT = 8501


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(
        os.environ.get("CHAINLIT_PORT", DEFAULT_PORT)
    )
    host = os.environ.get("CHAINLIT_HOST", "127.0.0.1")

    # init_config обязателен: читает .chainlit/config.toml относительно cwd
    init_config()

    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_chainlit.py")
    config.run.module_name = target
    load_module(target)

    ensure_jwt_secret()

    if not (config.code.on_chat_start or config.code.on_message):
        raise RuntimeError("Не заданы on_chat_start / on_message")

    init_markdown(config.root)

    async def start() -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                "chainlit.server:app",
                host=host,
                port=port,
                ws="auto",
                log_level="info",
            )
        )
        await server.serve()

    asyncio.run(start())


if __name__ == "__main__":
    main()
