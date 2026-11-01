# KODA.md — контекст проекта `rag_proto`

## Обзор проекта

Прототип (проверка концепции визуальной части) RAG-ассистента по Excel-отчётности:
пользователь задаёт вопрос на естественном языке, система показывает шаги пайплайна
(`convert → chunking → embed+search → rerank`), потоковый ответ и таблицу источников
с ссылками на конкретный файл, лист и чанк.

**Важно: это макет, а не рабочий RAG.** В проекте нет ни чтения Excel, ни векторного
поиска, ни LLM. Весь «движок» — файл `demo/mock_rag.py`, который по ключевым словам
подбирает заранее заготовленный ответ и набор чанков. Цель — оценить, как интерфейс
выглядит в деле, и зафиксировать контракт данных для будущей реальной реализации.

### Технологии

- **Python 3.14** (каждый подпроект имеет свой `venv/`).
- **Chainlit 2.11.0** — вариант чат-интерфейса (`chainlit-demo/app_chainlit.py`).
- **Gradio 6.31.0** — альтернативный вариант (`gradio-demo/app_gradio.py`).
- **Streamlit 1.50.0** — третий вариант визуала (`streamlit-demo/app_streamlit.py`).
- Зависимости зафиксированы в `requirements.txt` каждого подпроекта.

### Архитектура

```
chainlit-demo/mock_rag.py    ──(Answer, Chunk)──►  chainlit-demo/app_chainlit.py
gradio-demo/mock_rag.py      ──(Answer, Chunk)──►  gradio-demo/app_gradio.py
streamlit-demo/mock_rag.py   ──(Answer, Chunk)──►  streamlit-demo/app_streamlit.py
  движок-заглушка                              только отрисовка
```

Разделение сознательное: UI-слой не знает ничего о поиске, он умеет рисовать только
две структуры — `Answer(text, sources, candidates, kept)` и
`Chunk(chunk_id, source_file, sheet, score, text)`. Переход к настоящему пайплайну
означает замену трёх функций в `mock_rag.py` (`retrieve`, `answer`, `pipeline_steps`),
при этом все три фронтендные реализации трогать не придётся.

Три варианта намеренно держатся одного сценария и одного текста, чтобы их можно
было сравнивать по одному критерию — визуальному оформлению.

## Структура каталога

```
rag_proto/
├── chainlit-demo/       # Изолированный вариант на Chainlit
│   ├── venv/            # venv с chainlit==2.11.0, starlette>=0.47.2,<0.48
│   ├── app_chainlit.py  # UI на Chainlit
│   ├── mock_rag.py      # Движок-заглушка (копия)
│   ├── test_chainlit_handlers.py  # Смоук-тест
│   ├── README.md        # Инструкции по запуску
│   ├── chainlit.md      # Приветственный экран
│   └── .chainlit/       # Конфигурация Chainlit
├── gradio-demo/         # Изолированный вариант на Gradio
│   ├── venv/            # venv с gradio==6.31.0, starlette>=1.0.1
│   ├── app_gradio.py    # UI на Gradio
│   ├── mock_rag.py      # Движок-заглушка (копия)
│   └── README.md        # Инструкции по запуску
├── streamlit-demo/      # Изолированный вариант на Streamlit
│   ├── venv/            # venv со streamlit==1.50.0
│   ├── app_streamlit.py # UI на Streamlit
│   ├── mock_rag.py      # Движок-заглушка (копия)
│   └── README.md        # Инструкции по запуску + таблица отличий визуала
├── demo-old-backup/     # Старая структура (бэкап до разделения)
├── KODA.md              # Этот файл
└── chainlit.md          # Глобальный приветственный экран (не используется)
```

Каждый подпроект (`chainlit-demo/`, `gradio-demo/`) — полностью автономный:
свой `venv/`, свой `mock_rag.py`, нет `sys.path.insert` — модули импортируются
напрямую, так как лежат в одной папке.

## Сборка и запуск

### Chainlit-вариант

Запускать **только** через `serve.py` (штантный `chainlit run` даёт 500 на статике,
см. «Известные проблемы»):

```powershell
cd chainlit-demo
.\venv\Scripts\Activate.ps1
python serve.py 8501
```

### Gradio-вариант

```powershell
cd gradio-demo
.\venv\Scripts\Activate.ps1
python app_gradio.py
```

### Streamlit-вариант

```powershell
cd streamlit-demo
.\venv\Scripts\Activate.ps1
streamlit run app_streamlit.py --server.port 8502
```

Порт 8502 вместо дефолтного 8501 — на 8501 поднят Chainlit, иначе оба демо
одновременно не поднять.

### Смоук-тест хендлеров (Chainlit)

```powershell
cd chainlit-demo
.\venv\Scripts\Activate.ps1
python test_chainlit_handlers.py
```

Тестовый фреймворк не используется: `test_chainlit_handlers.py` — самостоятельный
скрипт на `asyncio`, печатает `[ok]` по этапам и `ALL PASSED` в конце. Линтер и
чекер типов не настроены.

## Известные проблемы

### 500-ошибки на `/assets/...` и `/favicon`

**Симптомы:** при запуске Chainlit браузер не загружает CSS/JS — все запросы
к `/assets/index-*.css`, `/assets/index-*.js`, `/favicon` возвращают 500.

**Истинная причина (подтверждена бисектом):** `nest_asyncio 1.6.0` на Python 3.14.
Штатный `chainlit run` вызывает `nest_asyncio.apply()` (в `chainlit/cli/__init__.py`).
На Python 3.14 это ломает цепочку:

```
nest_asyncio.apply()  →  asyncio.current_task() == None
                       →  sniffio не видит event loop
                       →  anyio.to_thread.run_sync() бросает NoEventLoopError
                       →  FileResponse в starlette падает на os.stat()
                       →  500 на всех /assets/*, /favicon
```

Что было исключено экспериментально: понижение `starlette` до 0.47.3 не помогает;
понижение `anyio` до 4.13.0 не помогает; чистый starlette `FileResponse` работает
200 до тех пор, пока не вызван `nest_asyncio.apply()`.

**Решение:** запускать через `serve.py` — он повторяет `run_chainlit()`, но
без `nest_asyncio.apply()` (нужен только в Jupyter, uvicorn в нём не нуждается):

```powershell
cd chainlit-demo
.\venv\Scripts\Activate.ps1
python serve.py 8501
```

После этого `/assets/*.css`, `/assets/*.js`, `/favicon` отдают 200.

### Зависимости

| Подпроект | Цепочка | Starlette |
|---|---|---|
| `chainlit-demo/` | chainlit 2.11.0 → starlette | `>=0.47.2,<0.48` |
| `gradio-demo/` | gradio 6.31.0 → starlette | `>=1.0.1,<2.0` |
| `streamlit-demo/` | streamlit 1.50.0 → tornado | не используется |

Раздельные venv позволяют избежать конфликта версий starlette. Streamlit к
конфликту не относится: его серверная часть — Tornado, starlette в venv вообще
не ставится.

## Правила разработки

### Стиль кода

- Заголовки модулей — русскоязычный docstring в одну-две строки, объясняющий
  назначение и способ запуска.
- `from __future__ import annotations` в начале модулей.
- Структуры данных — `@dataclass` (`Chunk`, `Answer`), а не словари.
- Публичные функции движка имеют type hints и короткий docstring с описанием
  возвращаемого контракта (`-> tuple[list[Chunk], int]`).
- Комментарии на русском, объясняют *зачем* (например, почему chosen именно такой
  псевдостемминг), а не *что* делает строка.
- Разделители крупных блоков — строки `# ---...---` с заголовком раздела.
- Импорты приложения из соседнего модуля делаются напрямую: `mock_rag.py` лежит
  в той же папке, что и UI, поэтому `sys.path.insert` не нужен.

### Тестирование

- Проверка UI идёт через вызов асинхронных хендлеров напрямую: поднимается
  `init_http_context()` со стабом эмиттера, затем по очереди дёргаются
  `start()`, `answer_question()`, колбэки действий.
- Новый хендлер или изменение сценария ответа — добавить шаг в
  `chainlit-demo/test_chainlit_handlers.py` и убедиться, что вывод заканчивается строкой
  `ALL PASSED`.
- Обязательные сценарии для проверки: релевантный запрос, нерелевантный запрос
  (fallback), клик по кнопке-подсказке.

### Контракт, который нельзя ломать

При переходе к реальному пайплайну сохраняется сигнатурная совместимость:

| Сейчас (мок) | Заменится на |
|---|---|
| `retrieve(query)` | поиск по `data/index/vectors.bin` |
| `answer(query)` | генерация через `src/llm/` |
| `pipeline_steps(query)` | реальные метрики из логов convert/chunk/embed |

Поля `Answer(text, sources, candidates, kept)` и
`Chunk(chunk_id, source_file, sheet, score, text)` уже используются всеми тремя
фронтендными реализациями — их переименование или удаление ломает отрисовку.

### Прочее

- Тестовые запросы, на которых заведены ответы: `выручка / регион Север`,
  `логистика / тарифы`, `возвраты`, `остатки / склад`. Всё остальное уходит в
  fallback-ответ — при добавлении новой темы ответ и чанки заводятся в `_KB`
  и `_SOURCES`.
- Не коммитить и не править вручную: `venv/`, `.files/`, `__pycache__/`,
  `.chainlit/translations/`.
- Секреты и API-ключи в макете не нужны; при подключении реального LLM ключи
  выносить в переменные окружения, а не в код (в `.chainlit/config.toml`
  `persist_user_env = false` — сохранять ключи пользователя в БД не следует).

## Открытые задачи (TODO)

- Реальный пайплайн: `convert` (XLSX → Markdown), chunking, эмбеддинги и
  векторный индекс вместо `mock_rag.py`.
- Каталог `data/normalized/` и `data/index/vectors.bin`, на которые ссылается
  приветственный текст, пока не существуют.
- `chainlit.md` содержит шаблонный текст Chainlit — заменить на описание ассистента
  либо очистить, если приветственный экран не нужен.
- `mock_rag.py` лежит в трёх подпроектах копией. Пока варианты сравниваются по
  визуалу это терпимо, но при переходе к реальному пайплайну копию стоит убрать
  в общий пакет (например `src/rag/`), иначе правки придётся дублировать трижды.
