# RAG demo — Chainlit

Мини-макет псевдо-диалога. **Ни Excel, ни векторного поиска, ни LLM нет** — `mock_rag.py`
отдаёт готовые ответы и источники, чтобы посмотреть, как интерфейс выглядит в деле.

## Структура

```
chainlit-demo/
├── mock_rag.py                  # заглушка: фейковый ретривал, rerank, ответы, источники
├── app_chainlit.py              # UI на Chainlit
├── serve.py                     # лаунчер без nest_asyncio (см. «Запуск»)
├── test_chainlit_handlers.py    # смоук-тест хендлеров без браузера
├── chainlit.md                  # приветственный экран
├── .chainlit/config.toml        # конфигурация Chainlit
└── requirements.txt             # chainlit==2.11.0
```

## Запуск

```powershell
cd chainlit-demo
.\venv\Scripts\Activate.ps1
python serve.py 8501
```

Открыть http://localhost:8501

> **Не используйте `chainlit run`** — он вызывает `nest_asyncio.apply()`, который на
> Python 3.14 ломает `asyncio.current_task()`. Из-за этого `anyio.to_thread.run_sync`
> не находит event loop, и все `/assets/*`, `/favicon` отдают 500. `serve.py` делает
> то же самое, но без `nest_asyncio`.

## Смоук-тест

```powershell
cd chainlit-demo
.\venv\Scripts\Activate.ps1
python test_chainlit_handlers.py
```

Вывод должен закончиться строкой `ALL PASSED`.

## Что смотреть

1. приветствие + `TaskList` со статусами пайплайна;
2. вопрос → раскрывающиеся **шаги** `convert → chunking → embed+search → rerank`;
3. потоковый ответ (токен за токеном, имитация генерации);
4. таблица **источников** — файл, лист, `chunk_id`, score; исходники чанков
   открываются в боковой панели;
5. кнопки-**подсказки** следующих вопросов.

Запросы с ответами по умолчанию: `выручка / регион Север`, `логистика / тарифы`,
`возвраты`, `остатки / склад`. Всё прочее уходит в fallback-ответ.

## Как перейти к настоящему пайплайну

`mock_rag.py` повторяет будущий контракт `src/` — прикрутить реальный пайплайн
значит заменить три функции:

| Сейчас (мок)        | Заменится на                                  |
|---------------------|-----------------------------------------------|
| `retrieve(query)`   | поиск по `data/index/vectors.bin`             |
| `answer(query)`     | генерация через `src/llm/`                    |
| `pipeline_steps()`  | реальные метрики из логов convert/chunk/embed |

`Answer(text, sources, candidates, kept)` и `Chunk(chunk_id, source_file, sheet, score, text)`
— то, что UI уже умеет отрисовывать, поэтому его трогать не придётся.