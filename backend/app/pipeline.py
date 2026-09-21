"""Оркестрация прогона: мутации → прод-модель → оценка → детекция."""
import json
import random
import threading
from datetime import datetime, timezone

from . import config, db
from .detection import detect_degradation
from .evaluation import evaluate_answer
from .llm_client import LLMError
from .mutations import generate_mutations
from .prod_model import answer_question

_run_lock = threading.Lock()


class RunBusy(RuntimeError):
    pass


class SetupNeeded(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_config(raw: dict):
    """Приводит конфиг из запроса к каноничному виду + человекочитаемое описание."""
    raw = raw or {}
    preset = raw.get("preset")
    if preset is None:
        if "context_truncation" in raw:
            preset = "context_truncation"
        elif raw.get("system_prompt"):
            preset = "vague_prompt"
        elif raw.get("use_weak_model"):
            preset = "weak_model"
        else:
            preset = "baseline"

    if preset == "baseline":
        cfg = {}
        desc = "Baseline: полный контекст БЗ, стандартный промпт, основная модель"
    elif preset == "context_truncation":
        value = float(raw.get("value", raw.get("context_truncation", 0.5)))
        value = min(max(value, 0.1), 0.9)
        cfg = {"context_truncation": value}
        desc = f"Деградация: контекст обрезан до {int(value * 100)}% релевантных статей БЗ"
    elif preset == "vague_prompt":
        cfg = {"system_prompt": "vague"}
        desc = "Деградация: системный промпт заменён на расплывчатый (без привязки к БЗ)"
    elif preset == "weak_model":
        cfg = {"use_weak_model": True}
        desc = f"Деградация: модель тихо понижена до {config.WEAK_MODEL}"
    else:
        raise ValueError(f"Неизвестный preset деградации: {preset}")
    return preset, cfg, desc


def run_setup() -> dict:
    from .synth_data import KNOWLEDGE_BASE, generate_historical_queries

    with db.get_conn() as c:
        c.execute("DELETE FROM test_cases")
        c.execute("DELETE FROM runs")
        c.execute("DELETE FROM historical_queries")
        c.execute("DELETE FROM kb_articles")

    with db.get_conn() as c:
        for a in KNOWLEDGE_BASE:
            c.execute(
                "INSERT INTO kb_articles(question, answer) VALUES(?, ?)",
                (a["question"], a["answer"]),
            )
        articles = c.execute(
            "SELECT id FROM kb_articles ORDER BY id"
        ).fetchall()
        kb_id_by_index = [r["id"] for r in articles]

        queries = generate_historical_queries()
        for q in queries:
            c.execute(
                "INSERT INTO historical_queries(question, article_id, kind) VALUES(?, ?, ?)",
                (q["question"], kb_id_by_index[q["kb_index"]], q["kind"]),
            )
    return {
        "kb_articles": len(KNOWLEDGE_BASE),
        "historical_queries": len(queries),
    }


def start_run(day_label, raw_config) -> dict:
    if not _run_lock.acquire(blocking=False):
        raise RunBusy("Уже выполняется другой прогон — дождитесь его завершения.")
    try:
        preset, cfg, desc = normalize_config(raw_config)
        if db.count("kb_articles") == 0 or db.count("historical_queries") == 0:
            raise SetupNeeded("Данные не сгенерированы — сначала выполните POST /setup.")
        label = (day_label or "").strip() or f"День {db.count('runs') + 1}"
        run_id = db.execute(
            "INSERT INTO runs(created_at, day_label, config_preset, config_json,"
            " config_description, status) VALUES(?, ?, ?, ?, ?, 'running')",
            (_now_iso(), label, preset, json.dumps(cfg, ensure_ascii=False), desc),
        )
    except Exception:
        _run_lock.release()
        raise

    threading.Thread(target=_worker, args=(run_id, cfg), daemon=True).start()
    return get_run(run_id)


def _worker(run_id: int, cfg: dict):
    try:
        _execute(run_id, cfg)
    except Exception as e:  # noqa: BLE001 — фиксируем любую ошибку прогона в БД
        db.execute(
            "UPDATE runs SET status='error', error=? WHERE id=?",
            (str(e)[:1000], run_id),
        )
    finally:
        _run_lock.release()


def _execute(run_id: int, cfg: dict):
    articles = [
        {"id": r["id"], "question": r["question"], "answer": r["answer"]}
        for r in db.fetchall("SELECT * FROM kb_articles")
    ]
    hist = [dict(r) for r in db.fetchall("SELECT * FROM historical_queries")]
    if not hist:
        raise SetupNeeded("Исторический лог пуст — выполните setup.")

    art_by_id = {a["id"]: a for a in articles}
    sample = random.sample(hist, min(config.SOURCE_QUERY_SAMPLE, len(hist)))

    # Тест-кейсы генерируются заново при каждом прогоне
    mutations = generate_mutations(sample, [a["question"] for a in articles])
    if not mutations:
        raise LLMError("Генератор мутаций не вернул ни одного тест-кейса.")

    run_cfg = dict(cfg)
    run_cfg["_kb_articles"] = articles

    for m in mutations:
        golden = art_by_id[m["article_id"]]
        record = {
            "run_id": run_id,
            "mutation_type": m["mutation_type"],
            "source_query": m["source_query"],
            "question": m["question"],
            "golden_question": golden["question"],
            "golden_answer": golden["answer"],
            "answer": None,
            "model_used": None,
            "factual_correctness": None,
            "completeness": None,
            "relevance": None,
            "similarity": None,
            "judge_comment": None,
        }
        try:
            pm = answer_question(m["question"], run_cfg)
            record["answer"] = pm["answer"]
            record["model_used"] = pm["model_used"]
        except Exception as e:  # noqa: BLE001 — кейс падает, прогон живёт
            record["judge_comment"] = f"ошибка вызова прод-модели: {e}"[:600]

        if record["answer"]:
            try:
                ev = evaluate_answer(m["question"], golden["answer"], record["answer"])
                record.update(ev)
            except Exception as e:  # noqa: BLE001
                record["similarity"] = None
                record["judge_comment"] = f"ошибка оценки: {e}"[:600]

        db.execute(
            "INSERT INTO test_cases(run_id, mutation_type, source_query, question,"
            " golden_question, golden_answer, answer, model_used,"
            " factual_correctness, completeness, relevance, similarity, judge_comment)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record["run_id"], record["mutation_type"], record["source_query"],
                record["question"], record["golden_question"], record["golden_answer"],
                record["answer"], record["model_used"],
                record["factual_correctness"], record["completeness"],
                record["relevance"], record["similarity"], record["judge_comment"],
            ),
        )

    cases = [dict(r) for r in db.fetchall(
        "SELECT * FROM test_cases WHERE run_id=?", (run_id,)
    )]
    agg = _aggregate(cases)
    baseline_pool = _baseline_pool(exclude_run=run_id)
    detection = detect_degradation(baseline_pool, cases)

    db.execute(
        "UPDATE runs SET status='done', avg_judge_score=?, avg_similarity=?,"
        " pct_low_factual=?, degradation_detected=?, degraded_axes=?, detection_json=?"
        " WHERE id=?",
        (
            agg["avg_judge_score"], agg["avg_similarity"], agg["pct_low_factual"],
            int(detection["degradation_detected"]),
            json.dumps(detection["degraded_axes"], ensure_ascii=False),
            json.dumps(detection, ensure_ascii=False),
            run_id,
        ),
    )


def _aggregate(cases: list) -> dict:
    judged = [c for c in cases if c.get("factual_correctness") is not None]
    sims = [c["similarity"] for c in cases if c.get("similarity") is not None]

    avg_judge = pct_low = avg_sim = None
    if judged:
        avg_judge = sum(
            (c["factual_correctness"] + c["completeness"] + c["relevance"]) / 3
            for c in judged
        ) / len(judged)
        pct_low = sum(1 for c in judged if c["factual_correctness"] < 2) / len(judged)
    if sims:
        avg_sim = sum(sims) / len(sims)
    return {
        "avg_judge_score": round(avg_judge, 4) if avg_judge is not None else None,
        "avg_similarity": round(avg_sim, 4) if avg_sim is not None else None,
        "pct_low_factual": round(pct_low, 4) if pct_low is not None else None,
    }


def _baseline_pool(exclude_run: int | None = None) -> list:
    rows = db.fetchall(
        "SELECT id FROM runs WHERE config_preset='baseline' AND status='done'"
        " AND (? IS NULL OR id != ?)"
        " ORDER BY id DESC LIMIT ?",
        (exclude_run, exclude_run, config.BASELINE_RUNS_LIMIT),
    )
    ids = [r["id"] for r in rows]
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    return [dict(r) for r in db.fetchall(
        f"SELECT * FROM test_cases WHERE run_id IN ({marks})", ids
    )]


# ---------- сериализация ----------

def run_summary(row) -> dict:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "day_label": row["day_label"],
        "config_preset": row["config_preset"],
        "config_description": row["config_description"],
        "status": row["status"],
        "error": row["error"],
        "avg_judge_score": row["avg_judge_score"],
        "avg_similarity": row["avg_similarity"],
        "pct_low_factual": row["pct_low_factual"],
        "degradation_detected": (
            bool(row["degradation_detected"])
            if row["degradation_detected"] is not None
            else None
        ),
        "degraded_axes": (
            json.loads(row["degraded_axes"]) if row["degraded_axes"] else []
        ),
    }


def _total(c: dict):
    vals = [c.get(a) for a in ("factual_correctness", "completeness", "relevance")]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _scores(c: dict) -> dict:
    return {
        a: c.get(a) for a in ("factual_correctness", "completeness", "relevance")
    }


def _case_sort_key(c: dict):
    sim = c.get("similarity")
    tot = _total(c)
    return (
        0 if sim is not None else 1,
        sim or 0,
        0 if tot is not None else 1,
        tot or 0,
    )


def get_run(run_id: int):
    row = db.fetchone("SELECT * FROM runs WHERE id=?", (run_id,))
    if row is None:
        return None
    summary = run_summary(row)
    cases = [dict(r) for r in db.fetchall(
        "SELECT * FROM test_cases WHERE run_id=?", (run_id,)
    )]
    cases.sort(key=_case_sort_key)
    detail = dict(summary)
    detail["cases"] = cases
    return detail


def list_runs() -> list:
    rows = db.fetchall("SELECT * FROM runs ORDER BY id ASC")
    return [run_summary(r) for r in rows]


def diff_with_baseline(run_id: int):
    row = db.fetchone("SELECT * FROM runs WHERE id=?", (run_id,))
    if row is None:
        return None
    result = {"run": run_summary(row), "baseline_run_ids": [], "examples": []}

    if row["config_preset"] == "baseline":
        result["note"] = "Это baseline-прогон — с ним сравниваются остальные прогоны."
        return result

    baseline_rows = db.fetchall(
        "SELECT id FROM runs WHERE config_preset='baseline' AND status='done'"
        " ORDER BY id DESC LIMIT ?",
        (config.BASELINE_RUNS_LIMIT,),
    )
    baseline_ids = [r["id"] for r in baseline_rows]
    result["baseline_run_ids"] = baseline_ids
    if not baseline_ids:
        result["note"] = "Нет завершённых baseline-прогонов для сравнения."
        return result

    marks = ",".join("?" * len(baseline_ids))
    baseline_cases = [dict(r) for r in db.fetchall(
        f"SELECT * FROM test_cases WHERE run_id IN ({marks})", baseline_ids
    )]
    current_cases = [dict(r) for r in db.fetchall(
        "SELECT * FROM test_cases WHERE run_id=?", (run_id,)
    )]

    if row["detection_json"]:
        result["detection"] = json.loads(row["detection_json"])
    else:
        result["detection"] = detect_degradation(baseline_cases, current_cases)

    result["examples"] = _worst_pairs(current_cases, baseline_cases)
    return result


def _worst_pairs(current_cases: list, baseline_cases: list) -> list:
    """Пары 'было / стало': матч по (source_query, mutation_type), сортировка по просадке.

    Тест-кейсы перегенерируются каждый прогон, поэтому точного попарного
    соответствия нет — матчинг по происхождению кейса.
    """
    best_base = {}
    for c in baseline_cases:
        tot = _total(c)
        if tot is None:
            continue
        key = (c["source_query"].strip().lower(), c["mutation_type"])
        if key not in best_base or tot > best_base[key][0]:
            best_base[key] = (tot, c)

    pairs = []
    for c in current_cases:
        tot = _total(c)
        if tot is None:
            continue
        key = (c["source_query"].strip().lower(), c["mutation_type"])
        if key not in best_base:
            continue
        base_tot, bc = best_base[key]
        pairs.append({
            "drop": round(base_tot - tot, 2),
            "mutation_type": c["mutation_type"],
            "question": c["question"],
            "source_query": c["source_query"],
            "golden_question": c["golden_question"],
            "golden_answer": c["golden_answer"],
            "baseline_answer": bc.get("answer"),
            "current_answer": c.get("answer"),
            "baseline_total": base_tot,
            "current_total": tot,
            "baseline_scores": _scores(bc),
            "current_scores": _scores(c),
        })

    pairs.sort(key=lambda p: p["drop"], reverse=True)
    if pairs:
        return pairs[:8]

    # фолбэк: просто худшие кейсы текущего прогона
    worst = sorted(
        current_cases,
        key=lambda c: (0, _total(c)) if _total(c) is not None else (1, 0),
    )[:8]
    return [
        {
            "drop": None,
            "mutation_type": c["mutation_type"],
            "question": c["question"],
            "source_query": c["source_query"],
            "golden_question": c["golden_question"],
            "golden_answer": c["golden_answer"],
            "baseline_answer": None,
            "current_answer": c.get("answer"),
            "baseline_total": None,
            "current_total": _total(c),
            "baseline_scores": None,
            "current_scores": _scores(c),
        }
        for c in worst
    ]
