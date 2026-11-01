"""Макет RAG-движка для теста визуала.

Ничего реально не считает: имитирует конвертацию Excel -> Markdown,
чанкинг, поиск, rerank и генерацию ответа. Ответы и источники — заготовки.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Фейковый корпус: будто бы это результат пайплайна convert/chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    sheet: str
    score: float
    text: str


@dataclass
class Answer:
    text: str
    sources: list[Chunk] = field(default_factory=list)
    candidates: int = 0
    kept: int = 0


_SOURCES = [
    Chunk(
        chunk_id="chunk_018",
        source_file="Отчёт_продаж_2025.xlsx",
        sheet="Сводка",
        score=0.0,
        text=(
            "| Показатель | Янв | Фев | Мар |\n"
            "|---|---|---|---|\n"
            "| Выручка, ₽ | 4 820 000 | 5 140 000 | 6 010 000 |\n"
            "| Маржа, % | 31.4 | 32.8 | 34.1 |\n"
        ),
    ),
    Chunk(
        chunk_id="chunk_031",
        source_file="Отчёт_продаж_2025.xlsx",
        sheet="Север",
        score=0.0,
        text=(
            "Регион «Север» показал рост выручки на 18,7% к прошлому кварталу. "
            "Основной вклад дали каналы оптовых продаж и два новых контракта "
            "с сетевыми покупателями."
        ),
    ),
    Chunk(
        chunk_id="chunk_047",
        source_file="Прайс_поставщиков.xlsx",
        sheet="Прайс",
        score=0.0,
        text=(
            "Стоимость логистики для Северного региона: 42 ₽/кг при объёме "
            "от 500 кг. При объёме свыше 2000 кг действует тариф 36 ₽/кг."
        ),
    ),
    Chunk(
        chunk_id="chunk_052",
        source_file="Отчёт_продаж_2025.xlsx",
        sheet="Возвраты",
        score=0.0,
        text=(
            "Доля возвратов за март составила 2,3%. Наиболее частая причина — "
            "повреждение упаковки при транспортировке (61% случаев)."
        ),
    ),
    Chunk(
        chunk_id="chunk_063",
        source_file="Остатки_склада.xlsx",
        sheet="Склад-1",
        score=0.0,
        text=(
            "Остаток на 31.03: 1 240 паллет. Резерв под подтверждённые "
            "заказы — 380 паллет, свободный остаток — 860 паллет."
        ),
    ),
]


# Псевдо-ответы: ключ -> (релевантные chunk_id, markdown-тело ответа)
_KB = [
    (
        {"выручк", "рост", "продаж", "квартал", "север", "регион"},
        ["chunk_031", "chunk_018"],
        (
            "**Выручка за квартал выросла**, основной вклад дал регион «Север»:\n\n"
            "- рост к прошлому кварталу — **18,7 %**;\n"
            "- вклад дали оптовые каналы и два новых контракта с сетевыми покупателями;\n"
            "- маржа по сводке поднялась с 31,4 % до 34,1 %.\n\n"
            "Данные собраны из листов `Сводка` и `Север` файла `Отчёт_продаж_2025.xlsx`."
        ),
    ),
    (
        {"логист", "доставк", "тариф", "стоимост", "перевоз"},
        ["chunk_047", "chunk_031"],
        (
            "Тарифы на логистику по Северному региону:\n\n"
            "| Объём | Тариф |\n"
            "|---|---|\n"
            "| от 500 кг | 42 ₽/кг |\n"
            "| свыше 2000 кг | 36 ₽/кг |\n\n"
            "Тариф указан в `Прайс_поставщиков.xlsx`, лист `Прайс`."
        ),
    ),
    (
        {"возврат", "брак", "качеств", "поврежд"},
        ["chunk_052"],
        (
            "Доля возвратов за март — **2,3 %**.\n\n"
            "Наиболее частая причина — повреждение упаковки при транспортировке, "
            "на неё приходится **61 %** всех случаев."
        ),
    ),
    (
        {"остатк", "склад", "паллет", "паллет", "резерв"},
        ["chunk_063"],
        (
            "Остатки на складе на 31.03:\n\n"
            "- всего — 1 240 паллет;\n"
            "- в резерве под подтверждённые заказы — 380;\n"
            "- свободный остаток — **860 паллет**."
        ),
    ),
]

_DEFAULT = ["chunk_018", "chunk_031"]
_DEFAULT_ANSWER = (
    "В индексе нет точного совпадения, поэтому показываю ближайшие фрагменты.\n\n"
    "Ближайшими оказались сводные показатели по продажам за квартал "
    "(`Отчёт_продаж_2025.xlsx`, листы `Сводка` и `Север`).\n\n"
    "Уточните запрос — например, добавьте название листа или показатель "
    "(`выручка`, `возвраты`, `остатки`, `логистика`)."
)


def _tokens(query: str) -> set[str]:
    """Грубый стемминг: отрезаем окончание и берём первые 6 символов."""
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", query.lower())
    stems = set()
    for w in words:
        stems.add(w[:6])
        if len(w) > 4:
            stems.add(w[:-1][:6])
    return stems


def _pick(query: str) -> tuple[list[str], str]:
    q = _tokens(query)
    best_key, best_hits = None, 0
    for key, ids, body in _KB:
        key_stems = {w[:6] for w in key}
        hits = len(q & key_stems)
        if hits > best_hits:
            best_key, best_hits = (ids, body), hits
    if best_key is None or best_hits == 0:
        return _DEFAULT, _DEFAULT_ANSWER
    return best_key


def retrieve(query: str) -> tuple[list[Chunk], int]:
    """Возвращает (топ-чанков, сколько кандидатов было до rerank)."""
    ids, _ = _pick(query)
    by_id = {c.chunk_id: c for c in _SOURCES}
    hits = [by_id[i] for i in ids if i in by_id]
    for rank, chunk in enumerate(hits, start=1):
        chunk.score = round(max(0.31, 0.94 - 0.11 * (rank - 1)), 2)
    return hits, 8


def answer(query: str) -> Answer:
    """Полный мок: ретривал -> rerank -> 'генерация'."""
    hits, candidates = retrieve(query)
    ids, body = _pick(query)
    by_id = {c.chunk_id: c for c in _SOURCES}
    return Answer(
        text=body,
        sources=[by_id[i] for i in ids if i in by_id],
        candidates=candidates,
        kept=len(hits),
    )


def pipeline_steps(query: str) -> list[tuple[str, str]]:
    """Шаги пайплайна для отображения в UI (название, статус)."""
    a = answer(query)
    return [
        ("convert · XLSX → Markdown", "3 файла → 41 страница"),
        ("chunking", "148 чанков, 512 токенов, overlap 64"),
        (f"embed + search", f"{a.candidates} кандидатов"),
        ("rerank", f"{a.kept} чанков после отсечения"),
    ]


def suggestions(query: str) -> list[str]:
    ids, _ = _pick(query)
    if "chunk_047" in ids:
        return ["Какой тариф при объёме от 2000 кг?", "Сколько стоила логистика в марте?"]
    if "chunk_052" in ids:
        return ["Почему растут возвраты?", "Доля брака по регионам"]
    if "chunk_063" in ids:
        return ["Свободный остаток на складе", "Что в резерве под заказы?"]
    return ["Покажи маржу по месяцам", "Какие регионы росли быстрее?", "Сколько возвратов в марте?"]


def sleep(seconds: float = 0.0) -> None:
    if seconds:
        time.sleep(seconds)
