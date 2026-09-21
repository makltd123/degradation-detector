"""Статистический слой детекции деградации.

Сравнивает метрики прогона с пулом baseline-прогонов ("день 1–3"):
относительное падение > порога И (при достаточной выборке) значимый
Welch t-test → помечаем ось как просевшую.
"""
import numpy as np
from scipy import stats

AXES = ["factual_correctness", "completeness", "relevance"]

ALPHA = 0.05
MIN_GROUP_FOR_TTEST = 5
REL_DROP_FLAG = 0.10   # практическая значимость: падение на 10%+
REL_DROP_STRONG = 0.25 # сильная просадка — флагуем даже без t-test
LOW_FACT_RISE_ABS = 0.15


def _clean(values):
    return [v for v in values if v is not None]


def _mean(values):
    return float(np.mean(values)) if values else None


def _welch_p(base_values, cur_values):
    if len(base_values) < MIN_GROUP_FOR_TTEST or len(cur_values) < MIN_GROUP_FOR_TTEST:
        return None
    b = np.asarray(base_values, dtype=float)
    c = np.asarray(cur_values, dtype=float)
    if np.std(b) == 0 and np.std(c) == 0:
        return None
    return float(stats.ttest_ind(c, b, equal_var=False).pvalue)


def _compare(metric: str, base_values, cur_values) -> dict:
    res = {
        "metric": metric,
        "baseline_mean": None,
        "current_mean": None,
        "delta": None,
        "rel_drop": None,
        "p_value": None,
        "degraded": False,
        "strong": False,
    }
    bm, cm = _mean(base_values), _mean(cur_values)
    if bm is None or cm is None:
        return res
    res["baseline_mean"] = round(bm, 4)
    res["current_mean"] = round(cm, 4)
    delta = bm - cm  # положительный = стало хуже
    res["delta"] = round(delta, 4)
    if abs(bm) > 1e-9:
        res["rel_drop"] = round(delta / abs(bm), 4)
    p = _welch_p(base_values, cur_values)
    res["p_value"] = round(p, 4) if p is not None else None

    rd = res["rel_drop"]
    significant = p is None or p < ALPHA
    if rd is not None:
        res["degraded"] = bool((rd >= REL_DROP_FLAG and significant) or rd >= REL_DROP_STRONG)
        res["strong"] = bool(rd >= REL_DROP_STRONG)
    return res


def detect_degradation(baseline_cases: list, current_cases: list) -> dict:
    """baseline_cases / current_cases — списки dict с оценками по осям и similarity."""
    result = {
        "axes": {},
        "semantic_similarity": None,
        "low_factual_rate": None,
        "degradation_detected": False,
        "degraded_axes": [],
        "note": None,
    }
    if not baseline_cases:
        result["note"] = "Нет baseline-прогонов (день 1–3) — сравнение невозможно."
        return result

    for axis in AXES:
        cmp_res = _compare(
            axis,
            _clean([c.get(axis) for c in baseline_cases]),
            _clean([c.get(axis) for c in current_cases]),
        )
        result["axes"][axis] = cmp_res
        if cmp_res["degraded"]:
            result["degraded_axes"].append(axis)

    sim = _compare(
        "semantic_similarity",
        _clean([c.get("similarity") for c in baseline_cases]),
        _clean([c.get("similarity") for c in current_cases]),
    )
    result["semantic_similarity"] = sim
    if sim["degraded"]:
        result["degraded_axes"].append("semantic_similarity")

    def low_share(cases):
        vals = [
            c.get("factual_correctness")
            for c in cases
            if c.get("factual_correctness") is not None
        ]
        if not vals:
            return None
        return sum(1 for v in vals if v < 2) / len(vals)

    lb, lc = low_share(baseline_cases), low_share(current_cases)
    lf = {"baseline": None, "current": None, "rise": None, "flagged": False}
    if lb is not None and lc is not None:
        lf = {
            "baseline": round(lb, 4),
            "current": round(lc, 4),
            "rise": round(lc - lb, 4),
            "flagged": bool((lc - lb) >= LOW_FACT_RISE_ABS),
        }
    result["low_factual_rate"] = lf
    if lf["flagged"]:
        result["degraded_axes"].append("low_factual_rate")

    result["degradation_detected"] = bool(result["degraded_axes"])
    return result
