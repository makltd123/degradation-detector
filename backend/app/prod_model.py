"""Эмуляция прод-QA-модели с управляемой деградацией.

Конфигурация прогона задаёт, какие параметры "испортили":
- context_truncation: float 0..1 — сколько релевантных статей БЗ реально
  попадает в контекст (кто-то обрезал retrieval / промпт);
- system_prompt == "vague" — системный промпт заменён на расплывчатый,
  без жёсткой привязки к базе знаний;
- use_weak_model: True — модель тихо понижена до более слабой.
"""
import re

from . import config
from .llm_client import call_llm

STOPWORDS = {
    "и", "в", "на", "с", "со", "как", "что", "для", "не", "ни", "я", "мы", "вы",
    "он", "она", "они", "оно", "у", "до", "по", "из", "за", "то", "же", "а",
    "но", "или", "к", "ко", "об", "от", "это", "есть", "быть", "мне", "вас",
    "там", "так", "уже", "ещё", "еще", "ли", "бы", "можно", "ли", "моя", "мой",
}

_SYSTEM_NORMAL_HEAD = (
    "Ты — ассистент техподдержки SaaS-продукта TaskFlow. "
    "Отвечай на вопрос пользователя, опираясь СТРОГО на приведённую ниже базу знаний. "
    "Используй факты из базы и не придумывай функции продукта. "
    "Если точного ответа в базе нет, ответь по ближайшей по смыслу статье "
    "и честно отметь, если каких-то данных в базе нет. "
    "Отвечай кратко (2–4 предложения), на русском языке.\n\n"
    "База знаний:\n\n"
)

_SYSTEM_VAGUE_HEAD = (
    "Ты — универсальный дружелюбный ассистент. Пользователь пишет в чат — "
    "старайся быть полезным в любом случае: рассуждай, предлагай варианты, "
    "опирайся на общий опыт, даже если не уверен в деталях. Ниже приложена "
    "какая-то справочная информация — используй её по своему усмотрению. "
    "Отвечай на русском языке.\n\n"
    "Справка:\n\n"
)


def _tokens(text: str) -> set:
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def retrieve_articles(question: str, articles: list, top_k: int) -> list:
    """Простой keyword-overlap ретривер: детерминированный, без внешних зависимостей."""
    qt = _tokens(question)

    def score(a):
        at = _tokens(a["question"]) | _tokens(a["answer"])
        return len(qt & at)

    ranked = sorted(articles, key=lambda a: score(a), reverse=True)
    picked = [a for a in ranked if score(a) > 0][:top_k]
    if not picked:
        picked = ranked[:top_k]
    return picked


def answer_question(question: str, run_config: dict) -> dict:
    """Отвечает как 'прод-модель' с данной конфигурацией деградации."""
    top_k = config.TOP_K_ARTICLES
    truncation = run_config.get("context_truncation")
    if truncation is not None:
        top_k = max(1, round(top_k * float(truncation)))

    articles = run_config.get("_kb_articles") or []
    picked = retrieve_articles(question, articles, top_k)
    context = "\n\n".join(
        f"[{i + 1}] Q: {a['question']}\nA: {a['answer']}" for i, a in enumerate(picked)
    )

    vague = run_config.get("system_prompt") == "vague"
    head = _SYSTEM_VAGUE_HEAD if vague else _SYSTEM_NORMAL_HEAD
    system = f"{head}{context}"

    model = config.WEAK_MODEL if run_config.get("use_weak_model") else config.MODEL
    answer = call_llm(question, system, model=model, temperature=0.3)
    return {
        "answer": answer.strip(),
        "model_used": model,
        "context_articles": len(picked),
    }
