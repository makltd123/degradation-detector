# Silent Degradation Detector

**English** | [Русский](README.ru.md)

A prototype service for early detection of **silent LLM quality degradation** in production. It does not wait for manual labeling: it generates adversarial test cases from the historical query log on its own, runs them through the current "production model" configuration every "day", and catches quality drops with LLM-as-judge and semantic similarity — even when the service is formally up and answering.

The knowledge base, test cases and UI are in Russian.

## How it works

```
Setup: FAQ knowledge base (~36 articles) + historical query log (~100)
  ↓
Every run (a "day"):
  1. An LLM generates fresh mutations of historical questions
     (paraphrase / harder / edge case / two-in-one) — the test set is NOT static
  2. Test cases go through an emulated "production model" with the chosen
     degradation config (baseline / truncated context / vague prompt /
     weak model)
  3. Answers are scored: semantic similarity (all answers) + LLM judge
     on factual_correctness / completeness / relevance (0–3)
  4. Comparison with baseline runs: relative drop + Welch's t-test →
     degradation_detected + list of axes that dropped
```

## Structure

```
backend/            FastAPI + SQLite
  app/
    config.py       settings from the environment / .env
    db.py           SQLite schema
    llm_client.py   single LLM client (openai-compatible | anthropic)
    synth_data.py   knowledge base + historical log generator
    prod_model.py   production model emulation with controlled degradation
    mutations.py    adversarial test case generation
    evaluation.py   semantic similarity + LLM judge
    detection.py    statistical detection (t-test, thresholds)
    pipeline.py     run orchestration
    main.py         REST API
frontend/           React + recharts (one monitoring screen)
```

## Running it

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # fill in LLM_API_KEY and the provider
uvicorn app.main:app --reload --port 8000
```

Requires Python 3.10+. The embedding model is downloaded automatically on first run.

### Frontend

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

## Demo walkthrough

1. **Setup** — generate the knowledge base and the query log.
2. Run 1–2 **baseline** runs (healthy model).
3. Run with degradation (for example "Vague system prompt" or "Truncated context").
4. The chart shows judge score and similarity dropping; the row is marked **DEGRADED**.
5. Click the degraded run → which axes dropped (factual correctness / completeness / relevance), the p-value, and concrete "before → after" examples: the question, the reference from the knowledge base, the baseline answer and the degraded model's answer, sorted by the size of the drop.

## Environment variables (backend/.env)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (or any OpenAI-compatible) / `anthropic` |
| `LLM_API_KEY` | — | API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Main model (all roles: mutations, production model, judge) |
| `LLM_WEAK_MODEL` | `gpt-3.5-turbo` | "Weak" model for the weak_model preset |
| `DD_JUDGE_MODE` | `all` | `all` — judge every answer; `low_similarity_only` — only where similarity is below the threshold (cheaper) |
| `DD_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embeddings (multilingual, for Russian content) |

## Important limitation

LLM-as-judge has its own bias and noise: this is an **early warning** tool, not a replacement for manual QA. In production such a service should be validated periodically with human review of a small sample of answers.

## License

[MIT](LICENSE)

---

Author: [Fedor Makovskiy](https://fedor.alqira.ru), AI developer. Services and projects with quality metrics: [fedor.alqira.ru](https://fedor.alqira.ru).
