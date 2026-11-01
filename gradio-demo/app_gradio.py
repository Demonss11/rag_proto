"""Gradio-макет RAG-чата. Запуск:  python demo/app_gradio.py"""

from __future__ import annotations

import time

import gradio as gr

import mock_rag


WELCOME = """\
# 📊 Ассистент по Excel-отчётности

Отвечаю по данным из `data/normalized/`: `Отчёт_продаж_2025.xlsx`,
`Прайс_поставщиков.xlsx`, `Остатки_склада.xlsx` — **148 чанков**.

> **Это макет визуала.** Ответы и источники — заготовки, реального поиска нет.
"""


def _tokenize(text: str):
    buf = ""
    for ch in text:
        buf += ch
        if ch in " \n":
            yield buf
            buf = ""
    if buf:
        yield buf


def _sources_md(answer: mock_rag.Answer) -> str:
    lines = ["| # | Источник | Лист | Чанк | Score |", "|---|---|---|---|---|"]
    for i, c in enumerate(answer.sources, start=1):
        lines.append(
            f"| {i} | `{c.source_file}` | {c.sheet} | `{c.chunk_id}` | {c.score:.2f} |"
        )
    return "\n".join(lines)


def _pipeline_md(query: str) -> str:
    rows = []
    for title, status in mock_rag.pipeline_steps(query):
        rows.append(f"- ✅ **{title}** — {status}")
    return "\n".join(rows)


def _suggestion_buttons(query: str) -> list[str]:
    return mock_rag.suggestions(query)


def respond(message: str, history: list):
    """Основной цикл: шаги пайплайна -> потоковый ответ -> источники."""
    # Radio в outputs обновляем через gr.update(choices=...), иначе список
    # воспринимается как выбранное значение и падает preprocess.
    clear_suggestions = gr.update(choices=_suggestion_buttons(""), value=None)

    if not message or not message.strip():
        yield "", history, _pipeline_md(""), "", clear_suggestions
        return

    answer = mock_rag.answer(message)

    # 1. шаги пайплайна в боковой панели
    pipeline = _pipeline_md(message)
    yield "", history, pipeline, "", clear_suggestions

    # 2. потоковая генерация
    stream = ""
    for token in _tokenize(answer.text):
        stream += token
        yield "", history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": stream}], pipeline, "", clear_suggestions
        time.sleep(0.015)

    # 3. источники и подсказки
    sources = "**Источники**\n\n" + _sources_md(answer)
    yield "", history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer.text}], \
        pipeline, sources, gr.update(choices=_suggestion_buttons(message), value=None)


def pick_suggestion(suggestion: str | None, history: list):
    """Клик по подсказке сразу запускает ответ без ввода в поле."""
    # Сброс выбора (value=None) тоже шлёт change — его игнорируем,
    # но отдаём no-op апдейты, чтобы outputs остались валидными.
    if not suggestion:
        yield (gr.update(),) * 5
        return
    yield from respond(suggestion, history)


with gr.Blocks(title="RAG demo · Gradio") as demo:
    gr.Markdown(WELCOME)

    with gr.Row():
        # --- центр: сам чат ---
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Диалог", height=520)
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Спросите по отчётам…",
                    show_label=False,
                    scale=4,
                    lines=1,
                )
                send = gr.Button("Отправить", variant="primary", scale=1)

            suggestions = gr.Radio(
                choices=_suggestion_buttons(""),
                label="Подсказки",
                interactive=True,
            )

        # --- правая колонка: пайплайна и источники ---
        with gr.Column(scale=2):
            with gr.Accordion("🔧 Пайплайн", open=True):
                pipeline_md = gr.Markdown("_задай вопрос — увидишь шаги_")
            with gr.Accordion("📚 Источники", open=True):
                sources_md = gr.Markdown("_появятся после ответа_")

    gr.Examples(
        examples=[
            ["Почему выросла выручка в регионе Север?"],
            ["Какие тарифы на логистику?"],
            ["Сколько возвратов в марте?"],
            ["Свободный остаток на складе"],
        ],
        inputs=[txt],
        label="Примеры запросов",
    )

    stream_args = dict(
        inputs=[txt, chatbot],
        outputs=[txt, chatbot, pipeline_md, sources_md, suggestions],
    )

    send.click(respond, **stream_args)
    txt.submit(respond, **stream_args)
    suggestions.change(
        pick_suggestion,
        inputs=[suggestions, chatbot],
        outputs=[txt, chatbot, pipeline_md, sources_md, suggestions],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
