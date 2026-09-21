"""Единый клиент LLM для всех ролей (генерация мутаций / прод-модель / судья).

Поддерживает провайдеров openai (любой OpenAI-compatible endpoint через
LLM_BASE_URL) и anthropic.
"""
import json
import re
import threading

import httpx

from . import config


class LLMError(RuntimeError):
    pass


_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(timeout=httpx.Timeout(180.0, connect=20.0))
    return _client


def call_llm(
    user: str,
    system: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    if not config.API_KEY:
        raise LLMError(
            "Ключ LLM API не задан. Заполните backend/.env "
            "(LLM_PROVIDER, LLM_API_KEY, LLM_MODEL)."
        )
    model = model or config.MODEL
    temperature = config.TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or config.MAX_TOKENS
    if config.PROVIDER == "anthropic":
        return _call_anthropic(system, user, model, temperature, max_tokens)
    return _call_openai(system, user, model, temperature, max_tokens)


def _call_openai(system, user, model, temperature, max_tokens) -> str:
    url = config.BASE_URL.rstrip("/") + "/chat/completions"
    resp = _get_client().post(
        url,
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        headers={"Authorization": f"Bearer {config.API_KEY}"},
    )
    if resp.status_code != 200:
        raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(system, user, model, temperature, max_tokens) -> str:
    base = config.BASE_URL.rstrip("/")
    url = base if base.endswith("/v1/messages") else base + "/v1/messages"
    resp = _get_client().post(
        url,
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        headers={
            "x-api-key": config.API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    if resp.status_code != 200:
        raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
    parts = resp.json().get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def extract_json(text: str):
    """Достаёт первый сбалансированный JSON-объект/массив из ответа модели."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    for opener in ("{", "["):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        try:
            return json.loads(cleaned[start:])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Не удалось извлечь JSON из ответа модели: {(text or '')[:200]}")


def call_llm_json(user: str, system: str, retries: int = 1, **kwargs):
    prompt = user
    last_err = None
    for _ in range(retries + 1):
        raw = call_llm(prompt, system, **kwargs)
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            prompt = (
                user
                + "\n\nВАЖНО: предыдущий ответ не был валидным JSON. "
                "Верни ТОЛЬКО JSON без какого-либо текста вокруг."
            )
    raise LLMError(f"Модель не вернула валидный JSON: {last_err}")
