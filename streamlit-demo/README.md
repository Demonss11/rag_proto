# streamlit-demo — Streamlit-вариант визуала RAG-чата

## Обзор

Третий вариант визуального макета RAG-ассистента по Excel-отчётности — на **Streamlit 1.50.0**.
Повторяет сценарий Chainlit- и Gradio-версий: вопрос → шаги пайплайна → потоковый
ответ → таблица источников с текстом чанков.

**Это макет визуала.** Нет ни чтения Excel, ни векторного поиска, ни LLM.
Весь «движок» — `mock_rag.py`, копия того, что лежит в `chainlit-demo/` и `gradio-demo/`.

## Запуск

```powershell
cd streamlit-demo
.\venv\Scripts\Activate.ps1
streamlit run app_streamlit.py --server.port 8502
```

Откроется `http://localhost:8502`.

> Порт **8502**, а не дефолтный 8501: на 8501 работает Chainlit-вариант,
> иначе два демо не поднять одновременно.

## Структура

| Файл | Назначение |
|---|---|
| `app_streamlit.py` | UI на Streamlit: чат, `st.status` с шагами, `st.write_stream`, сайдбар |
| `mock_rag.py` | Заглушка RAG-движка (та же, что в chainlit-demo и gradio-demo) |
| `requirements.txt` | Только `streamlit==1.50.0` |

## Чем отличается визуал

| Элемент | Chainlit | Gradio | Streamlit |
|---|---|---|---|
| Шаги пайплайна | `cl.TaskList` + `cl.Step` (CoT) | аккордеон в правой колонке | `st.status` (раскрывающийся, прямо в сообщении) |
| Источники | боковая панель `display="side"` | аккордеон в правой колонке | `st.expander` под ответом + сайдбар с выбором чанка |
| Стриминг | `stream_token` | generator-yield | `st.write_stream` |
| Подсказки | `cl.Action` (кнопки) | `gr.Radio` | `st.columns` + `st.button` |
| Приветствие | welcome-экран + `chainlit.md` | `gr.Markdown` сверху | `st.markdown`, пока нет сообщений |
| Реакция на ответ | — | — | `st.feedback("thumbs")` |

### Известные особенности Streamlit

- Кнопка не может сама заполнить `st.chat_input`, поэтому нажатие подсказки
  кладёт запрос в `st.session_state.pending`, и следующий прогон скрипта
  обрабатывает его как обычный ввод.
- Скрипт выполняется целиком на каждое действие, поэтому диалог хранится
  в `st.session_state.messages`, а после генерации вызывается `st.rerun()`.

## Сравнение трёх вариантов

| Фреймворк | Порт | Сильные стороны | Слабые стороны |
|---|---|---|---|
| Chainlit 2.11.0 | 8501 (`python serve.py 8501`) | CoT-панель, side panel для файлов, экосистема для чатов | 500 на статике при запуске через `chainlit run` (nest_asyncio + Py 3.14) |
| Gradio 6.31.0 | 7860 (`python app_gradio.py`) | две колонки из коробки, `gr.Examples` | чат «не чат», подсказки через Radio |
| Streamlit 1.50.0 | 8502 (`streamlit run …`) | быстрый старт, `st.status`, `st.feedback` | нет нативного чат-виджета с кнопками, пересборка скрипта на каждый клик |