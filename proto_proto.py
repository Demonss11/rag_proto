Изучи прототип: """
Финальный прототип RAG-пайплайна на BGE-M3 + JSON.

Секции:
  1. Клиент BGE-M3
  2. Схема чанка, хеши, чанкование (стабильные ID)
  3. Индексация (контекстный префикс в embedding_text)
  4. Хранение (save/load с версионированием)
  5. Поиск: гибрид (dense+sparse) + RRF/weighted + ColBERT + parent expansion
  6. Запрос к LLM
  7. Оценка (Recall@k, nDCG@k, MRR)
  8. Генерация пар (LLM) и remap при пересборке
  9. Калибровка весов (grid search + опциональный PyMC)
 10. Пример использования
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from openai import OpenAI

log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

KB_VERSION = 1  # bump, если меняется схема


# ============================================================
# 1. Клиент BGE-M3
# ============================================================

class LocalBGEM3Client:
    def __init__(self, base_url: str, api_key: str = "not-needed",
                 encoding_param: str = "encoding_type", timeout: float = 60.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.encoding_param = encoding_param

    def encode(self, texts: list[str], model: str = "bge-m3") -> list[dict]:
        results = [{"text": t} for t in texts]

        resp = self.client.embeddings.create(
            model=model, input=texts, encoding_format="float",
        )
        for it in resp.data:
            results[it.index]["dense"] = it.embedding

        for i, t in enumerate(texts):
            for vt in ("sparse", "colbert"):
                try:
                    r = self.client.embeddings.create(
                        model=model, input=[t],
                        extra_body={self.encoding_param: vt},
                    )
                    if r.data and r.data[0].embedding is not None:
                        results[i][vt] = r.data[0].embedding
                except Exception as e:
                    log.debug("skip %s: %s", vt, e)
        return results


# ============================================================
# 2. Схема чанка и утилиты
# ============================================================

def _sha(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


@dataclass
class Chunk:
    # идентичность (content-addressed, стабильна по тексту)
    chunk_id: str
    doc_id: str
    text_hash: str
    is_parent: bool = False
    parent_chunk_id: Optional[str] = None
    # источник
    file_name: str = ""
    file_content_hash: str = ""      # для инкрементальной переиндексации
    page: Optional[int] = None
    chunk_index: int = 0
    # структура
    heading: Optional[str] = None
    heading_path: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # содержимое
    text: str = ""
    embedding_text: str = ""
    # векторы (только у дочерних)
    dense: list[float] = field(default_factory=list)
    sparse: dict[str, float] = field(default_factory=dict)
    colbert: Optional[list[list[float]]] = None

    def to_json(self) -> dict:
        return asdict(self)


def normalize_sparse(sp) -> dict[str, float]:
    if sp is None:
        return {}
    if isinstance(sp, dict) and "indices" in sp and "values" in sp:
        return {str(i): float(v) for i, v in zip(sp["indices"], sp["values"])}
    if isinstance(sp, list):
        return {str(sp[i]): float(sp[i + 1]) for i in range(0, len(sp) - 1, 2)}
    if isinstance(sp, dict):
        return {str(k): float(v) for k, v in sp.items()}
    return {}


def make_embedding_text(chunk: Chunk) -> str:
    parts = [f"Документ: {chunk.file_name}."]
    if chunk.heading_path:
        parts.append("Раздел: " + " > ".join(chunk.heading_path) + ".")
    elif chunk.heading:
        parts.append(f"Раздел: {chunk.heading}.")
    if chunk.tags:
        parts.append("Теги: " + ", ".join(chunk.tags) + ".")
    return " ".join(parts) + "\n\n" + chunk.text


def split_into_chunks(
    text: str,
    doc_id: str,
    file_name: str,
    *,
    file_content_hash: str = "",
    parent_size: int = 1200,
    child_size: int = 300,
    heading: Optional[str] = None,
    heading_path: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    page: Optional[int] = None,
) -> list[Chunk]:
    """Чанкование с content-addressed ID."""
    heading_path = heading_path or []
    tags = tags or []

    def pack(paragraphs: list[str], target: int) -> list[str]:
        out, buf = [], ""
        for p in paragraphs:
            if len(buf) + len(p) + 1 > target and buf:
                out.append(buf.strip()); buf = p
            else:
                buf += ("\n" + p) if buf else p
        if buf.strip():
            out.append(buf.strip())
        return out

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parent_texts = pack(paragraphs, parent_size)

    chunks: list[Chunk] = []
    for ptext in parent_texts:
        p_hash = _sha(ptext)
        parent_id = f"{doc_id}_p_{p_hash}"

        chunks.append(Chunk(
            chunk_id=parent_id, doc_id=doc_id, text_hash=p_hash,
            is_parent=True, parent_chunk_id=None,
            file_name=file_name, file_content_hash=file_content_hash,
            page=page, heading=heading, heading_path=heading_path,
            tags=tags, text=ptext,
        ))

        child_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", ptext) if p.strip()]
        for ci, ctext in enumerate(pack(child_paragraphs, child_size)):
            c_hash = _sha(ctext)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c_{c_hash}", doc_id=doc_id, text_hash=c_hash,
                is_parent=False, parent_chunk_id=parent_id,
                file_name=file_name, file_content_hash=file_content_hash,
                page=page, chunk_index=ci, heading=heading,
                heading_path=heading_path, tags=tags, text=ctext,
            ))
    return chunks


# ============================================================
# 3. Индексация
# ============================================================

def embed_chunks(client: LocalBGEM3Client, chunks: list[Chunk]) -> list[Chunk]:
    children = [c for c in chunks if not c.is_parent]
    for c in children:
        c.embedding_text = make_embedding_text(c)
    if not children:
        return chunks

    vectors = client.encode([c.embedding_text for c in children])
    for c, v in zip(children, vectors):
        c.dense = [float(x) for x in v.get("dense", [])]
        c.sparse = normalize_sparse(v.get("sparse"))
        c.colbert = v.get("colbert")
    log.info("embedded %d child chunks", len(children))
    return chunks


# ============================================================
# 4. Хранение
# ============================================================

def save_kb(chunks: list[Chunk], path: str | Path) -> None:
    payload = {
        "version": KB_VERSION,
        "built_at": __import__("datetime").datetime.utcnow().isoformat(),
        "chunks": [c.to_json() for c in chunks],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("saved %d chunks → %s", len(chunks), path)


def load_kb(path: str | Path) -> list[Chunk]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "chunks" in raw:
        if raw.get("version") != KB_VERSION:
            log.warning("kb version %s != %s", raw.get("version"), KB_VERSION)
        raw = raw["chunks"]
    chunks = [Chunk(**r) for r in raw]
    for c in chunks:
        c.dense = np.asarray(c.dense, dtype=np.float32) if c.dense else np.zeros(0, np.float32)
        c.sparse = normalize_sparse(c.sparse)
        c.colbert = np.asarray(c.colbert, dtype=np.float32) if c.colbert else None
    return chunks


# ============================================================
# 5. Поиск
# ============================================================

def _dense_scores(q: np.ndarray, children: list[Chunk]) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-9)
    m = np.stack([c.dense / (np.linalg.norm(c.dense) + 1e-9) for c in children])
    return m @ q


def _sparse_scores(q_sparse: dict[str, float], children: list[Chunk]) -> np.ndarray:
    return np.array(
        [sum(w * c.sparse.get(t, 0.0) for t, w in q_sparse.items()) for c in children],
        dtype=np.float32,
    )


def _colbert_scores(q_mat: np.ndarray, children: list[Chunk]) -> np.ndarray:
    q = q_mat / (np.linalg.norm(q_mat, axis=1, keepdims=True) + 1e-9)
    out = np.zeros(len(children), dtype=np.float32)
    for i, c in enumerate(children):
        if c.colbert is None or not len(c.colbert):
            continue
        d = c.colbert / (np.linalg.norm(c.colbert, axis=1, keepdims=True) + 1e-9)
        out[i] = (q @ d.T).max(axis=1).sum()
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-9)


def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def _weighted(d: np.ndarray, s: np.ndarray, c: Optional[np.ndarray],
              w: dict[str, float]) -> np.ndarray:
    score = w.get("dense", 0.0) * _minmax(d) + w.get("sparse", 0.0) * _minmax(s)
    if c is not None and c.any():
        score = score + w.get("colbert", 0.0) * _minmax(c)
    return score


DEFAULT_WEIGHTS = {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}


def retrieve(
    question: str,
    client: LocalBGEMM3Client if False else "LocalBGEM3Client",  # type: ignore
    chunks: list[Chunk],
    *,
    mode: str = "rrf",                    # "rrf" | "weighted"
    weights: Optional[dict[str, float]] = None,
    top_k_parents: int = 3,
    top_k_children: int = 10,
    n_candidates: int = 30,
    filter_tags: Optional[Iterable[str]] = None,
    filter_doc_id: Optional[str] = None,
) -> list[Chunk]:
    """
    Гибридный поиск:
      1) dense + sparse по детям, слияние (RRF или weighted)
      2) ColBERT-реранк по кандидатам
      3) расширение child → parent
    """
    children = [c for c in chunks if not c.is_parent and c.dense.size]
    if filter_tags:
        ft = set(filter_tags)
        children = [c for c in children if ft.issubset(set(c.tags))]
    if filter_doc_id:
        children = [c for c in children if c.doc_id == filter_doc_id]
    if not children:
        return []

    qv = client.encode([question])[0]
    q_dense = np.asarray(qv["dense"], dtype=np.float32)
    q_sparse = normalize_sparse(qv.get("sparse"))
    q_colbert = qv.get("colbert")

    d = _dense_scores(q_dense, children)
    s = _sparse_scores(q_sparse, children)

    if mode == "weighted":
        weights = weights or DEFAULT_WEIGHTS
        c_all = _colbert_scores(np.asarray(q_colbert, np.float32), children) \
            if q_colbert is not None else None
        blended = _weighted(d, s, c_all, weights)
        cand_idx = list(np.argsort(-blended)[:n_candidates])
        pre_scores = blended
    else:  # rrf
        d_rank = list(np.argsort(-d)[:n_candidates])
        s_rank = list(np.argsort(-s)[:n_candidates])
        cand_idx = _rrf([d_rank, s_rank])[:n_candidates]
        pre_scores = None

    # ColBERT rerank
    if q_colbert is not None and len(q_colbert):
        sub = [children[i] for i in cand_idx]
        c_scores = _colbert_scores(np.asarray(q_colbert, np.float32), sub)
        order = np.argsort(-c_scores)
        final_children = [sub[j] for j in order[:top_k_children]]
    else:
        final_children = [children[i] for i in cand_idx[:top_k_children]]

    # parent expansion
    by_id = {c.chunk_id: c for c in chunks}
    parent_ids: list[str] = []
    for c in final_children:
        if c.parent_chunk_id and c.parent_chunk_id not in parent_ids:
            parent_ids.append(c.parent_chunk_id)
    return [by_id[pid] for pid in parent_ids if pid in by_id][:top_k_parents]


def retrieve_children(
    question: str,
    client: "LocalBGEM3Client",
    chunks: list[Chunk],
    *,
    top_k: int = 10,
    mode: str = "rrf",
    weights: Optional[dict[str, float]] = None,
) -> list[Chunk]:
    """Возвращает дочерние чанки в порядке релевантности (для оценки и разметки)."""
    children = [c for c in chunks if not c.is_parent and c.dense.size]
    if not children:
        return []

    qv = client.encode([question])[0]
    d = _dense_scores(np.asarray(qv["dense"], np.float32), children)
    s = _sparse_scores(normalize_sparse(qv.get("sparse")), children)
    q_colbert = qv.get("colbert")

    if mode == "weighted":
        weights = weights or DEFAULT_WEIGHTS
        c_all = _colbert_scores(np.asarray(q_colbert, np.float32), children) \
            if q_colbert is not None else None
        order = list(np.argsort(-_weighted(d, s, c_all, weights))[:top_k])
    else:
        order = _rrf([list(np.argsort(-d)[:top_k]),
                      list(np.argsort(-s)[:top_k])])[:top_k]
    return [children[i] for i in order]


# ============================================================
# 6. Запрос к LLM
# ============================================================

def ask_llm(
    question: str,
    context_chunks: list[Chunk],
    llm: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    if not context_chunks:
        return "Не нашёл релевантных фрагментов в базе знаний."
    blocks = []
    for i, c in enumerate(context_chunks, 1):
        src = c.file_name
        if c.page:
            src += f", стр. {c.page}"
        blocks.append(f"[{i}] ({src})\n{c.text}")
    context = "\n\n".join(blocks)
    prompt = (
        "Ты — ассистент, отвечающий только по предоставленному контексту.\n"
        "Если в контексте нет ответа — честно скажи, что не знаешь.\n"
        "Ссылайся на источники в формате [1], [2].\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}"
    )
    resp = llm.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ============================================================
# 7. Оценка
# ============================================================

def _rank_of(target_id: str, ranked: list[Chunk]) -> Optional[int]:
    for i, c in enumerate(ranked, 1):
        if c.chunk_id == target_id:
            return i
    return None


def evaluate(
    pairs: list[dict],
    client: "LocalBGEM3Client",
    chunks: list[Chunk],
    *,
    top_k_values: Iterable[int] = (5, 10),
    mode: str = "rrf",
    weights: Optional[dict[str, float]] = None,
) -> dict:
    """
    Возвращает {"recall@k": ..., "ndcg@k": ..., "mrr": ..., "n": ...}.
    Пара: {"question": str, "positive": chunk_id, "negatives": [chunk_id, ...]}
    """
    ks = sorted(set(top_k_values))
    kmax = max(ks)
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    rrs = []

    for p in pairs:
        ranked = retrieve_children(
            p["question"], client, chunks,
            top_k=kmax, mode=mode, weights=weights,
        )
        rank = _rank_of(p["positive"], ranked)
        for k in ks:
            recalls[k].append(1.0 if rank is not None and rank <= k else 0.0)
            if rank is not None and rank <= k:
                ndcgs[k].append(1.0 / np.log2(rank + 1))
            else:
                ndcgs[k].append(0.0)
        rrs.append(1.0 / rank if rank else 0.0)

    out = {"n": len(pairs), "mrr": float(np.mean(rrs)) if rrs else 0.0}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return out


# ============================================================
# 8. Генерация пар и remap
# ============================================================

def gen_questions(chunk_text: str, llm: OpenAI, n: int = 3,
                  model: str = "gpt-4o-mini") -> list[str]:
    prompt = (
        f"Прочитай фрагмент и сгенерируй {n} разных вопросов, "
        "на которые можно ответить ТОЛЬКО по этому фрагменту.\n"
        "Верни JSON: {\"questions\": [...]}\n\n"
        f"Фрагмент:\n{chunk_text}"
    )
    resp = llm.chat.completions.create(
        model=model, temperature=0.3,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content).get("questions", [])


def build_pairs(chunks: list[Chunk], llm: OpenAI,
                *, n_per_chunk: int = 2, n_neg: int = 4,
                seed: int = 0) -> list[dict]:
    import random
    rng = random.Random(seed)
    children = [c for c in chunks if not c.is_parent]
    ids = [c.chunk_id for c in children]
    pairs = []
    for c in children:
        for q in gen_questions(c.text, llm, n=n_per_chunk):
            pool = [x for x in ids if x != c.chunk_id]
            negatives = rng.sample(pool, k=min(n_neg, len(pool)))
            pairs.append({
                "question": q,
                "positive": c.chunk_id,
                "positive_snippet": c.text[:120],   # для восстановления
                "negatives": negatives,
            })
    log.info("generated %d pairs", len(pairs))
    return pairs


def remap_pairs(
    pairs: list[dict],
    old_chunks: list[Chunk],
    new_chunks: list[Chunk],
) -> list[dict]:
    """Пересобирает пары после переиндексации по text_hash."""
    old_by_hash = {c.text_hash: c.chunk_id for c in old_chunks if not c.is_parent}
    new_by_hash = {c.text_hash: c.chunk_id for c in new_chunks if not c.is_parent}
    common = set(old_by_hash) & set(new_by_hash)
    mapping = {old_by_hash[h]: new_by_hash[h] for h in common}

    remapped, dropped = [], 0
    for p in pairs:
        if p["positive"] not in mapping:
            dropped += 1
            continue
        remapped.append({
            "question": p["question"],
            "positive": mapping[p["positive"]],
            "positive_snippet": p.get("positive_snippet", ""),
            "negatives": [mapping.get(n, n) for n in p.get("negatives", [])],
        })
    log.info("remapped %d pairs, dropped %d", len(remapped), dropped)
    return remapped


# ============================================================
# 9. Калибровка весов
# ============================================================

def _phi(qv: dict, chunk: Chunk) -> np.ndarray:
    qd = np.asarray(qv["dense"], np.float32); qd /= (np.linalg.norm(qd) + 1e-9)
    cd = np.asarray(chunk.dense, np.float32); cd /= (np.linalg.norm(cd) + 1e-9)
    d = float(qd @ cd)
    qs = normalize_sparse(qv.get("sparse"))
    s = float(sum(w * chunk.sparse.get(t, 0.0) for t, w in qs.items()))
    c = 0.0
    if qv.get("colbert") is not None and chunk.colbert is not None:
        qc = np.asarray(qv["colbert"], np.float32)
        dc = np.asarray(chunk.colbert, np.float32)
        qc /= (np.linalg.norm(qc, axis=1, keepdims=True) + 1e-9)
        dc /= (np.linalg.norm(dc, axis=1, keepdims=True) + 1e-9)
        c = float((qc @ dc.T).max(axis=1).sum())
    return np.array([d, s, c], dtype=np.float64)


def _simplex_grid(step: float = 0.05) -> list[dict]:
    n = int(round(1 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append({
                "dense": i * step,
                "sparse": j * step,
                "colbert": k * step,
            })
    return out


def calibrate_weights_grid(
    pairs: list[dict],
    client: "LocalBGEM3Client",
    chunks: list[Chunk],
    *,
    metric: str = "mrr",
    step: float = 0.1,
    top_k: int = 10,
) -> dict:
    """Перебор весов на симплексе. Возвращает {"best": {...}, "score": ..., "all": [...]}."""
    children = [c for c in chunks if not c.is_parent]
    by_id = {c.chunk_id: c for c in children}

    # кэшируем qv и phi, чтобы не считать BGE на каждом шаге
    cache = []
    for p in pairs:
        qv = client.encode([p["question"]])[0]
        pos = by_id.get(p["positive"])
        if pos is None:
            continue
        phi_pos = _phi(qv, pos)
        phis_neg = [_phi(qv, by_id[n]) for n in p.get("negatives", []) if n in by_id]
        cache.append((phi_pos, phis_neg))

    results = []
    for w in _simplex_grid(step):
        wv = np.array([w["dense"], w["sparse"], w["colbert"]])
        rr_sum = 0.0
        n = 0
        for phi_pos, phis_neg in cache:
            if not phis_neg:
                continue
            pos_score = float(wv @ phi_pos)
            neg_scores = [float(wv @ ph) for ph in phis_neg]
            rank = 1 + sum(1 for s in neg_scores if s > pos_score)
            if rank <= top_k:
                rr_sum += 1.0 / rank
            n += 1
        score = rr_sum / n if n else 0.0
        results.append({**w, "score": score})

    results.sort(key=lambda x: -x["score"])
    return {"best": results[0], "metric": metric, "all": results}


def calibrate_weights_bayes(
    pairs: list[dict],
    client: "LocalBGEM3Client",
    chunks: list[Chunk],
    *,
    prior: tuple[float, float, float] = (4.0, 2.0, 4.0),
    n_samples: int = 2000,
) -> dict:
    """
    Bradley–Terry + Dirichlet prior. Требует `pip install pymc`.
    Возвращает posterior mean и 95% CI для каждого веса.
    """
    try:
        import pymc as pm
        import arviz as az  # noqa: F401
    except ImportError as e:
        raise ImportError("установите pymc: pip install pymc") from e

    children = [c for c in chunks if not c.is_parent]
    by_id = {c.chunk_id: c for c in children}

    dphi = []
    for p in pairs:
        pos = by_id.get(p["positive"])
        if pos is None:
            continue
        qv = client.encode([p["question"]])[0]
        phi_pos = _phi(qv, pos)
        for n_id in p.get("negatives", [])[:4]:
            neg = by_id.get(n_id)
            if neg is not None:
                dphi.append(phi_pos - _phi(qv, neg))
    dphi = np.asarray(dphi)
    if len(dphi) == 0:
        raise ValueError("нет пар для калибровки")
    # нормируем масштабы сигналов, чтобы не было плохой обусловленности
    dphi = dphi / (dphi.std(axis=0) + 1e-9)

    with pm.Model():
        w = pm.Dirichlet("w", a=np.asarray(prior, dtype=float))
        logits = pm.math.dot(dphi, w)
        pm.Bernoulli("obs", logit_p=logits, observed=np.ones(len(dphi)))
        trace = pm.sample(n_samples, tune=1000, chains=2, progressbar=False,
                          target_accept=0.9)

    post = trace.posterior["w"].stack(sample=("chain", "draw")).values
    return {
        "dense":  {"mean": float(post[0].mean()),
                   "ci95": [float(np.percentile(post[0], 2.5)),
                            float(np.percentile(post[0], 97.5))]},
        "sparse": {"mean": float(post[1].mean()),
                   "ci95": [float(np.percentile(post[1], 2.5)),
                            float(np.percentile(post[1], 97.5))]},
        "colbert": {"mean": float(post[2].mean()),
                    "ci95": [float(np.percentile(post[2], 2.5)),
                             float(np.percentile(post[2], 97.5))]},
    }


# ============================================================
# 10. Пример использования
# ============================================================

if __name__ == "__main__":
    BGE_URL = "http://127.0.0.1:17347/v1"
    KB_PATH = "kb.json"
    PAIRS_PATH = "eval_pairs.json"

    bge = LocalBGEM3Client(base_url=BGE_URL)
    llm = OpenAI(base_url="https://api.openai.com/v1", api_key="...")

    # ---------- 1. Построение базы ----------
    # docs = [Path("docs/report.md"), Path("docs/manual.md")]
    # all_chunks: list[Chunk] = []
    # for p in docs:
    #     text = p.read_text(encoding="utf-8")
    #     all_chunks += split_into_chunks(
    #         text, doc_id=p.stem, file_name=p.name,
    #         file_content_hash=_file_hash(p),
    #         tags=["отчёт"] if "report" in p.name else ["инструкция"],
    #     )
    # all_chunks = embed_chunks(bge, all_chunks)
    # save_kb(all_chunks, KB_PATH)

    # ---------- 2. Запрос ----------
    # chunks = load_kb(KB_PATH)
    # question = "Что такое BGE M3?"
    # ctx = retrieve(question, bge, chunks, mode="rrf", top_k_parents=3)
    # print(ask_llm(question, ctx, llm))

    # ---------- 3. Генерация пар ----------
    # pairs = build_pairs(chunks, llm, n_per_chunk=2, n_neg=4)
    # Path(PAIRS_PATH).write_text(
    #     json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 4. Оценка ----------
    # pairs = json.loads(Path(PAIRS_PATH).read_text(encoding="utf-8"))
    # print("rrf     :", evaluate(pairs, bge, chunks, mode="rrf"))
    # print("weighted:", evaluate(pairs, bge, chunks, mode="weighted",
    #                            weights=DEFAULT_WEIGHTS))

    # ---------- 5. Калибровка ----------
    # cal = calibrate_weights_grid(pairs, bge, chunks, metric="mrr", step=0.1)
    # print("best weights:", cal["best"])
    # # и опционально байес (нужен pymc):
    # # print(calibrate_weights_bayes(pairs, bge, chunks))

    # ---------- 6. Remap при пересборке ----------
    # old = load_kb("kb_old.json")
    # new = load_kb(KB_PATH)
    # pairs2 = remap_pairs(pairs, old, new)
    # Path("eval_pairs_remapped.json").write_text(
    #     json.dumps(pairs2, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("prototype ready; раскомментируйте нужную секцию в __main__")
