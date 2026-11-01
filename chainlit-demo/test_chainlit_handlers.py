"""Смоук-тест хендлеров Chainlit-макета: поднимает HTTP-контекст
со стабом emitter'а и прогоняет полный сценарий ответа."""

import asyncio

import chainlit as cl
from chainlit.context import init_http_context

import app_chainlit


async def main() -> None:
    init_http_context()

    # 1. старт чата: welcome + task list + подсказки
    await app_chainlit.start()
    print("[ok] on_chat_start")

    # 2. полный цикл ответа: шаги -> стрим -> источники -> подсказки
    await app_chainlit.answer_question("Какие тарифы на логистику?")
    print("[ok] answer_question")

    # 3. fallback-ответ на нерелевантный запрос
    await app_chainlit.answer_question("котики")
    print("[ok] fallback answer")

    # 4. колбэк кнопки-подсказки
    action = cl.Action(
        name="suggest", payload={"q": "Сколько возвратов в марте?"}, label="x"
    )
    await app_chainlit.on_suggest(action)
    print("[ok] action_callback")

    print("ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
