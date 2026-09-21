"""Оценка ответов: semantic similarity для всех ответов +
LLM-as-judge по осям factual_correctness / completeness / relevance (0–3).

Разделение стоимости: similarity дешёвая и гоняется на каждый ответ,
LLM-judge дорогой и включается по режиму DD_JUDGE_MODE
(all | low_similarity_only).
"""
import threading

from . import config
from .llm_client import LLMError, call_llm_json

_embedder = None
_embedder_lock = threading.Lock()


def _get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def semantic_similarity(a: str, b: str) -> float:
    model = _get_embedder()
    emb = model.encode([a, b], normalize_embeddings=True)
    return float(emb[0] @ emb[1])


JUDGE_SYSTEM = (
    "Ты — беспристрастный судья качества ответов QA-ассистента техподдержки. "
    "Сравниваешь ответ ассистента с эталонным ответом из базы знаний. "
    "Отвечай строго JSON без текста вне JSON."
)

JUDGE_USER_TMPL = """Вопрос пользователя:
{question}

Эталонный ответ (база знаний):
{golden}

Ответ ассистента:
{candidate}

Оцени ответ ассистента по трём осям, каждая — целое число 0–3:
- factual_correctness (фактология): 3 — полностью согласуется с эталоном; 2 — верно, но с мелкими неточностями; 1 — есть сомнительные или частично неверные утверждения; 0 — прямо противоречит эталону.
- completeness (полнота): 3 — покрывает весь вопрос так же, как эталон; 2 — отвечает на суть, упуская детали; 1 — отвечает лишь на малую часть; 0 — не отвечает на вопрос.
- relevance (релевантность): 3 — строго по теме вопроса; 2 — по теме с небольшими отступлениями; 1 — сильно уходит в сторону; 0 — нерелевантен.

Верни JSON: {{"factual_correctness": int, "completeness": int, "relevance": int, "comment": "краткое обоснование"}}"""


def _clamp_score(value) -> int:
    iv = int(round(float(value)))
    return max(0, min(3, iv))


def judge_answer(question: str, golden: str, candidate: str) -> dict:
    user = JUDGE_USER_TMPL.format(
        question=question[:2000], golden=golden[:2000], candidate=candidate[:2000]
    )
    data = call_llm_json(user, JUDGE_SYSTEM, temperature=0.0)
    try:
        return {
            "factual_correctness": _clamp_score(data["factual_correctness"]),
            "completeness": _clamp_score(data["completeness"]),
            "relevance": _clamp_score(data["relevance"]),
            "comment": str(data.get("comment", ""))[:600],
        }
    except (KeyError, TypeError, ValueError) as e:
        raise LLMError(f"Судья вернул неожиданный формат оценок: {e}")


def evaluate_answer(question: str, golden_answer: str, candidate_answer: str) -> dict:
    result = {
        "similarity": semantic_similarity(golden_answer, candidate_answer),
        "factual_correctness": None,
        "completeness": None,
        "relevance": None,
        "judge_comment": None,
    }
    need_judge = True
    if config.JUDGE_MODE == "low_similarity_only":
        need_judge = result["similarity"] < config.JUDGE_SIMILARITY_THRESHOLD
    if not need_judge:
        result["judge_comment"] = (
            f"judge пропущен: similarity >= {config.JUDGE_SIMILARITY_THRESHOLD}"
        )
        return result
    judged = judge_answer(question, golden_answer, candidate_answer)
    result.update(judged)
    return result
