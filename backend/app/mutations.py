"""Генерация adversarial тест-кейсов: LLM-мутации исторических запросов.

Набор генерируется заново при каждом прогоне — не статичный тест-сет.
"""
import random

from . import config
from .llm_client import LLMError, call_llm_json

TYPES = ["paraphrase", "hardened", "edge_case", "combined"]

TYPE_INSTRUCTIONS = {
    "paraphrase": "перефразируй вопрос другими словами, точно сохранив смысл",
    "hardened": (
        "усложни вопрос: добавь нерелевантные детали, разговорный стиль, "
        "1–2 опечатки — суть вопроса должна остаться той же"
    ),
    "edge_case": (
        "сформулируй вопрос на стыке темы исходного вопроса и ещё одной темы "
        "из списка тем базы знаний, чтобы вопрос частично подходил к двум статьям"
    ),
    "combined": (
        "объедини исходный вопрос и другой вопрос из списка тем в один "
        "составной вопрос пользователя"
    ),
}

SYSTEM = (
    "Ты — генератор adversarial тестовых вопросов для проверки QA-бота "
    "техподдержки SaaS-продукта. Отвечай строго JSON без текста вокруг."
)


def _chunk_prompt(chunk: list, kb_questions: list) -> str:
    topics = "\n".join(f"- {q}" for q in kb_questions)
    type_docs = "\n".join(f"- {t}: {TYPE_INSTRUCTIONS[t]}" for t in TYPES)
    tasks = "\n".join(
        f'{i}. [{", ".join(types)}] {row["question"]}'
        for i, (row, types) in enumerate(chunk, 1)
    )
    return f"""Темы базы знаний (вопросы статей FAQ):
{topics}

Описание типов мутаций:
{type_docs}

Исходные вопросы пользователей и требуемые типы мутаций (формат: номер. [типы] вопрос):
{tasks}

Сгенерируй для каждого исходного вопроса ровно по одной мутации каждого указанного для него типа.
Требования:
- Мутация — это один вопрос пользователя, звучащий естественно для чата поддержки.
- paraphrase сохраняет точный смысл; hardened сохраняет суть, но усложняет формулировку; edge_case затрагивает тему исходного вопроса и ещё одну тему из списка; combined объединяет исходный вопрос с другим вопросом из списка в один.
- Не придумывай функции продукта, которых нет в темах базы знаний.

Верни строго JSON:
{{"mutations": [{{"source_index": 1, "type": "paraphrase", "question": "..."}}]}}"""


def generate_mutations(source_rows: list, kb_questions: list) -> list:
    """source_rows: строки historical_queries (question + article_id).

    Возвращает list[dict]: article_id, source_query, mutation_type, question.
    """
    plans = []
    for row in source_rows:
        types = TYPES[:]
        random.shuffle(types)
        plans.append((dict(row), types[: config.MUTATIONS_PER_QUERY]))

    chunks = [plans[i : i + 5] for i in range(0, len(plans), 5)]

    out = []
    seen = set()
    for chunk in chunks:
        try:
            data = call_llm_json(_chunk_prompt(chunk, kb_questions), SYSTEM, temperature=0.8)
        except LLMError:
            continue  # неудачный чанк пропускаем, остальные живут
        if isinstance(data, list):
            data = {"mutations": data}
        mutations = data.get("mutations", []) if isinstance(data, dict) else []
        for m in mutations:
            try:
                idx = int(m.get("source_index", -1)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(chunk)):
                continue
            row, allowed_types = chunk[idx]
            mtype = m.get("type")
            question = str(m.get("question", "")).strip()
            if mtype not in allowed_types or len(question) < 10:
                continue
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "article_id": row["article_id"],
                    "source_query": row["question"],
                    "mutation_type": mtype,
                    "question": question,
                }
            )
    return out
