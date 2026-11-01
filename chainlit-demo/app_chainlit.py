"""Chainlit-макет RAG-чата. Запуск:  chainlit run demo/app_chainlit.py"""

from __future__ import annotations

import asyncio

import chainlit as cl
import mock_rag


WELCOME = """\
# 📊 Ассистент по Excel-отчётности

Я отвечаю по данным из папки `data/normalized/`. Сейчас в индексе:

| Файл | Листы | Чанков |
|---|---|---|
| `Отчёт_продаж_2025.xlsx` | Сводка, Север, Возвраты | 84 |
| `Прайс_поставщиков.xlsx` | Прайс | 31 |
| `Остатки_склада.xlsx` | Склад-1, Склад-2 | 33 |

> **Это макет визуала.** Ответы и источники — заготовки, реального поиска нет.
"""


def _sources_block(answer: mock_rag.Answer) -> str:
    """Список источников в виде таблицы markdown."""
    lines = ["| # | Источник | Лист | Чанк | Score |", "|---|---|---|---|---|"]
    for i, c in enumerate(answer.sources, start=1):
        lines.append(
            f"| {i} | `{c.source_file}` | {c.sheet} | `{c.chunk_id}` | {c.score:.2f} |"
        )
    return "\n".join(lines)


def _actions_for(suggestions: list[str]) -> list[cl.Action]:
    return [
        cl.Action(
            name="suggest",
            payload={"q": s},
            label=s,
            tooltip="Задать этот вопрос",
        )
        for s in suggestions
    ]


@cl.action_callback("suggest")
async def on_suggest(action: cl.Action):
    """Клик по кнопке-подсказке подставляет вопрос в чат."""
    await cl.Message(content=action.payload["q"], author="Вы").send()
    await answer_question(action.payload["q"])


@cl.on_chat_start
async def start():
    await cl.Message(content=WELCOME).send()

    task_list = cl.TaskList(
        tasks=[
            cl.Task(title="convert · XLSX → Markdown", status=cl.TaskStatus.DONE),
            cl.Task(title="chunking · 148 чанков", status=cl.TaskStatus.DONE),
            cl.Task(title="embed + index · vectors.bin", status=cl.TaskStatus.DONE),
            cl.Task(title="search + rerank", status=cl.TaskStatus.READY),
        ]
    )
    await task_list.send()

    await cl.Message(
        content="Попробуйте спросить:",
        actions=_actions_for(
            [
                "Почему выросла выручка в регионе Север?",
                "Какие тарифы на логистику?",
                "Сколько возвратов в марте?",
            ]
        ),
    ).send()


async def _run_pipeline_steps(query: str) -> None:
    """Отрисовывает шаги пайплайна как раскрывающиеся шаги."""
    for title, status in mock_rag.pipeline_steps(query):
        kind = "retrieval" if "search" in title or "embed" in title else "run"
        if "rerank" in title:
            kind = "rerank"
        async with cl.Step(name=title, type=kind, default_open=False) as step:
            step.output = status
            await asyncio.sleep(0.45)


async def _stream_answer(text: str) -> None:
    """Имитирует потоковую генерацию токен за токеном."""
    msg = cl.Message(content="")
    for token in _tokenize(text):
        await msg.stream_token(token)
        await asyncio.sleep(0.02)
    await msg.send()


def _tokenize(text: str):
    """Грубая нарезка на «токены»: слова с сохранением пробелов и переносов."""
    buf = ""
    for ch in text:
        buf += ch
        if ch in " \n":
            yield buf
            buf = ""
    if buf:
        yield buf


async def answer_question(query: str) -> None:
    answer = mock_rag.answer(query)

    await _run_pipeline_steps(query)

    await _stream_answer(answer.text)

    await cl.Message(
        content="**Источники**\n\n" + _sources_block(answer),
        elements=[
            cl.Text(
                name=c.chunk_id,
                content=c.text,
                display="side",
                language="markdown",
            )
            for c in answer.sources
        ],
    ).send()

    await cl.Message(
        content="Уточнить:",
        actions=_actions_for(mock_rag.suggestions(query)),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    await answer_question(message.content)
