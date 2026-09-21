# Silent Degradation Detector

[English](README.md) | **Русский**

Прототип сервиса раннего обнаружения **тихой деградации качества LLM** в проде. Сервис не ждёт ручной разметки: он сам генерирует adversarial тест-кейсы на основе исторического лога запросов, каждый «день» прогоняет их через текущую конфигурацию «прод-модели» и ловит просадку качества через LLM-as-judge и semantic similarity — даже когда сервис формально жив и отвечает.

## Как это работает

```
Setup: база знаний FAQ (~36 статей) + исторический лог запросов (~100)
  ↓
Каждый прогон («день»):
  1. LLM генерирует мутации исторических вопросов заново
     (перефраз / усложнённый / edge case / два-в-одном) — тест-сет НЕ статичный
  2. Тест-кейсы прогоняются через эмулируемую «прод-модель» с выбранной
     конфигурацией деградации (baseline / обрезанный контекст / размытый промпт /
     слабая модель)
  3. Ответы оцениваются: semantic similarity (все ответы) + LLM-judge
     по осям factual_correctness / completeness / relevance (0–3)
  4. Сравнение с baseline-прогонами: относительное падение + Welch t-test →
     degradation_detected + список просевших осей
```

## Структура

```
backend/            FastAPI + SQLite
  app/
    config.py       настройки из окружения / .env
    db.py           схема SQLite
    llm_client.py   единый клиент LLM (openai-compatible | anthropic)
    synth_data.py   база знаний + генератор исторического лога
    prod_model.py   эмуляция прод-модели с управляемой деградацией
    mutations.py    генерация adversarial тест-кейсов
    evaluation.py   semantic similarity + LLM-judge
    detection.py    статистическая детекция (t-test, пороги)
    pipeline.py     оркестрация прогона
    main.py         REST API
frontend/           React + recharts (один экран мониторинга)
```

## Запуск

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # заполнить LLM_API_KEY и провайдера
uvicorn app.main:app --reload --port 8000
```

Требуется Python 3.10+. Модель эмбеддингов при первом запуске скачается автоматически.

### Frontend

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

## Демо-сценарий

1. **Setup** — сгенерировать базу знаний и лог запросов.
2. Запустить 1–2 прогона **baseline** (здоровая модель).
3. Запустить прогон с деградацией (например, «Размытый системный промпт» или «Обрезанный контекст»).
4. На графике видно падение judge-score и similarity; строка помечается **DEGRADED**.
5. Клик по деградировавшему прогону → по каким осям просело (фактология / полнота / релевантность), p-value и конкретные примеры «было → стало»: вопрос, эталон из базы знаний, ответ baseline и ответ деградировавшей модели, отсортированные по величине просадки.

## Переменные окружения (backend/.env)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (или любой OpenAI-compatible) / `anthropic` |
| `LLM_API_KEY` | — | Ключ API |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Основная модель (все роли: мутации, прод-модель, судья) |
| `LLM_WEAK_MODEL` | `gpt-3.5-turbo` | «Слабая» модель для пресета weak_model |
| `DD_JUDGE_MODE` | `all` | `all` — судья на каждый ответ; `low_similarity_only` — только где similarity < порога (экономия) |
| `DD_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Эмбеддинги (мультиязычные, под русский контент) |

## Важное ограничение

LLM-as-judge сам по себе имеет bias и шум: это инструмент **ранней тревоги**, а не замена ручного QA. В проде такой сервис нужно периодически валидировать человеческой проверкой небольшой выборки ответов.

## Лицензия

[MIT](LICENSE)
