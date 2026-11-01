"""Streamlit-макет RAG-чата. Запуск:  python streamlit-demo/app_streamlit.py"""

from __future__ import annotations

import time

import streamlit as st

import mock_rag

# ---------------------------------------------------------------------------
# Конфигурация страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ассистент по Excel-отчётности",
    page_icon="📊",
    layout="wide",
)

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

STARTERS = [
    "Почему выросла выручка в регионе Север?",
    "Какие тарифы на логистику?",
    "Сколько возвратов в марте?",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)


# ---------------------------------------------------------------------------
# Хелперы отрисовки
# ---------------------------------------------------------------------------
def sources_table(sources: list[mock_rag.Chunk]) -> str:
    lines = ["| # | Источник | Лист | Чанк | Score |", "|---|---|---|---|---|"]
    for i, c in enumerate(sources, start=1):
        lines.append(
            f"| {i} | `{c.source_file}` | {c.sheet} | `{c.chunk_id}` | {c.score:.2f} |"
        )
    return "\n".join(lines)


def render_sources(sources: list[mock_rag.Chunk]) -> None:
    st.markdown(sources_table(sources))
    for c in sources:
        with st.expander(f"`{c.chunk_id}` · {c.source_file} · {c.sheet}"):
            st.markdown(c.text)


def render_steps(steps: list[tuple[str, str]]) -> None:
    for title, status in steps:
        st.markdown(f"- ✅ **{title}** — {status}")


def tokenize(text: str):
    buf = ""
    for ch in text:
        buf += ch
        if ch in " \n":
            yield buf
            buf = ""
    if buf:
        yield buf


def stream_tokens(text: str):
    for token in tokenize(text):
        yield token
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Сайдбар
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Источники")
    last_sources = next(
        (m["sources"] for m in reversed(st.session_state.messages) if m.get("sources")),
        [],
    )
    if last_sources:
        st.caption("Последний ответ")
        st.markdown(sources_table(last_sources))
        chosen = st.selectbox(
            "Текст чанка",
            options=[c.chunk_id for c in last_sources],
            format_func=lambda cid: f"{cid} · {[c.sheet for c in last_sources if c.chunk_id == cid][0]}",
        )
        chunk = next(c for c in last_sources if c.chunk_id == chosen)
        st.markdown(chunk.text)
        st.caption(f"`{chunk.source_file}` · score {chunk.score:.2f}")
    else:
        st.info("Источники появятся после первого ответа.")

    st.divider()
    st.header("🔧 Пайплайн")
    last_steps = next(
        (m["steps"] for m in reversed(st.session_state.messages) if m.get("steps")),
        [],
    )
    if last_steps:
        render_steps(last_steps)
    else:
        st.info("Задайте вопрос.")


# ---------------------------------------------------------------------------
# Приветствие
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(WELCOME)
    st.caption("Попробуйте спросить:")
    for col, question in zip(st.columns(3), STARTERS):
        if col.button(question, use_container_width=True, key=f"start_{question}"):
            st.session_state.pending = question
            st.rerun()


# ---------------------------------------------------------------------------
# История диалога
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Источники ({len(msg['sources'])})", expanded=False):
                render_sources(msg["sources"])
            st.feedback("thumbs", key=f"fb_{i}")


# ---------------------------------------------------------------------------
# Подсказки после последнего ответа
# ---------------------------------------------------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_idx = len(st.session_state.messages) - 1
    last = st.session_state.messages[-1]
    if last.get("content"):
        st.caption("Уточнить:")
        for col, question in zip(st.columns(3), mock_rag.suggestions(last["query"])):
            # Индекс сообщения в ключах: одни и те же подсказки могут
            # повторяться в разных ответах, а ключ обязан быть уникальным.
            if col.button(question, use_container_width=True, key=f"sug_{last_idx}_{question}"):
                st.session_state.pending = question
                st.rerun()


# ---------------------------------------------------------------------------
# Ввод и обработка запроса
# ---------------------------------------------------------------------------
# Кнопка-подсказка не умеет сама заполнить chat_input, поэтому запрос
# приходит либо из поля ввода, либо из pending, оставленного кнопкой на прошлом прогоне.
query = st.chat_input("Спросите по отчётам…") or st.session_state.pending or ""
st.session_state.pending = None
query = query.strip()

if not query:
    st.stop()

# Пользовательское сообщение
st.session_state.messages.append({"role": "user", "content": query})
with st.chat_message("user"):
    st.markdown(query)

# Пайплайн + стриминг
answer = mock_rag.answer(query)

with st.chat_message("assistant", avatar="🤖"):
    with st.status("Ищу по индексу…", expanded=True) as status:
        steps = mock_rag.pipeline_steps(query)
        for title, detail in steps:
            st.write(f"{title} — {detail}")
            time.sleep(0.35)
        status.update(
            label=f"Пайплайн: {answer.candidates} → {answer.kept}",
            state="complete",
            expanded=False,
        )

    text = st.write_stream(stream_tokens(answer.text))

    with st.expander(f"Источники ({len(answer.sources)})", expanded=False):
        render_sources(answer.sources)
    st.feedback("thumbs", key=f"fb_new_{len(st.session_state.messages)}")

st.session_state.messages.append(
    {
        "role": "assistant",
        "content": text,
        "sources": answer.sources,
        "steps": steps,
        "query": query,
    }
)
st.rerun()
