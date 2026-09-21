import { useCallback, useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { createRun, fetchDiff, fetchRun, fetchRuns, setupProject } from "./api";

const PRESETS = [
  { value: "baseline", label: "Baseline — здоровая модель" },
  { value: "context_truncation", label: "Обрезанный контекст (50% базы знаний)" },
  { value: "vague_prompt", label: "Размытый системный промпт" },
  { value: "weak_model", label: "Пониженная (слабая) модель" },
];

const AXIS_NAMES = {
  factual_correctness: "Фактология",
  completeness: "Полнота",
  relevance: "Релевантность",
  semantic_similarity: "Semantic similarity",
  low_factual_rate: "Доля ответов с фактологией < 2",
};

const MUTATION_NAMES = {
  paraphrase: "перефраз",
  hardened: "усложнённый",
  edge_case: "edge case",
  combined: "два-в-одном",
};

const errText = (e) =>
  e?.response?.data?.detail || e?.message || "Что-то пошло не так";

function fmt(v, d = 2) {
  return v == null ? "—" : Number(v).toFixed(d);
}

function DegradedDot({ cx, cy, payload, color }) {
  if (cx == null || cy == null) return null;
  if (payload.degraded)
    return (
      <circle cx={cx} cy={cy} r={7} fill="#e11d48" stroke="#fff" strokeWidth={2} />
    );
  return <circle cx={cx} cy={cy} r={3.5} fill={color} />;
}

function ScoreBadge({ value }) {
  const tone =
    value == null
      ? "badge-gray"
      : value >= 3
        ? "badge-green"
        : value === 2
          ? "badge-amber"
          : "badge-red";
  return <span className={`badge ${tone}`}>{value == null ? "—" : `${value}/3`}</span>;
}

function SimBadge({ value }) {
  if (value == null) return <span className="badge badge-gray">—</span>;
  const tone = value >= 0.8 ? "badge-green" : value >= 0.65 ? "badge-amber" : "badge-red";
  return <span className={`badge ${tone}`}>{Math.round(value * 100)}%</span>;
}

function MetricCell({ m }) {
  if (!m) return null;
  return (
    <div className={`metric-cell ${m.degraded ? "metric-degraded" : ""}`}>
      <div className="metric-name">
        {AXIS_NAMES[m.metric] || m.metric} {m.degraded ? "⚠" : ""}
      </div>
      <div className="metric-values">
        {fmt(m.baseline_mean)} → {fmt(m.current_mean)}{" "}
        {m.rel_drop != null && (
          <span className={m.rel_drop >= 0.1 ? "drop-red" : "drop-ok"}>
            ({(m.rel_drop * 100).toFixed(1)}%)
          </span>
        )}
      </div>
      <div className="metric-sub">
        p-value: {m.p_value ?? "н/д"} · деградация: {m.degraded ? "да" : "нет"}
      </div>
    </div>
  );
}

function CaseCard({ c }) {
  return (
    <div className="case-card">
      <div className="case-head">
        <span className={`tag tag-${c.mutation_type}`}>
          {MUTATION_NAMES[c.mutation_type] || c.mutation_type}
        </span>
        <span className="case-q">{c.question}</span>
      </div>
      <div className="case-badges">
        <span className="badge-label">судья:</span>
        <ScoreBadge value={c.factual_correctness} />
        <ScoreBadge value={c.completeness} />
        <ScoreBadge value={c.relevance} />
        <span className="badge-label">sim:</span>
        <SimBadge value={c.similarity} />
        {c.model_used && <span className="model-tag">{c.model_used}</span>}
      </div>
      {c.judge_comment && <div className="case-comment">{c.judge_comment}</div>}
      <details>
        <summary>Эталон и ответ модели</summary>
        <div className="case-block">
          <b>Эталон (база знаний):</b> {c.golden_answer}
        </div>
        <div className="case-block">
          <b>Ответ модели:</b> {c.answer || <i>нет ответа</i>}
        </div>
        <div className="case-block muted">Из лога: {c.source_query}</div>
      </details>
    </div>
  );
}

function ExampleCard({ ex }) {
  return (
    <div className="example-card">
      <div className="case-head">
        <span className={`tag tag-${ex.mutation_type}`}>
          {MUTATION_NAMES[ex.mutation_type] || ex.mutation_type}
        </span>
        <span className="case-q">{ex.question}</span>
      </div>
      <div className="pair-grid">
        <div className="pair-col">
          <div className="pair-title">Было (baseline): {fmt(ex.baseline_total)} / 9</div>
          <div>{ex.baseline_answer || <i>нет данных</i>}</div>
        </div>
        <div className="pair-col pair-degraded">
          <div className="pair-title">Стало: {fmt(ex.current_total)} / 9</div>
          <div>{ex.current_answer || <i>нет ответа</i>}</div>
        </div>
      </div>
      <div className="case-block muted">
        <b>Эталон:</b> {ex.golden_answer}
      </div>
    </div>
  );
}

export default function App() {
  const [setupInfo, setSetupInfo] = useState(null);
  const [runs, setRuns] = useState([]);
  const [label, setLabel] = useState("");
  const [preset, setPreset] = useState("baseline");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [details, setDetails] = useState(null);
  const [diff, setDiff] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setRuns(await fetchRuns());
    } catch {
      /* backend ещё не поднят */
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    setLabel((prev) => prev || `День ${runs.length + 1}`);
  }, [runs.length]);

  const doneRuns = runs.filter((r) => r.status === "done");
  const anyRunning = runs.some((r) => r.status === "running");
  const chartData = doneRuns.map((r) => ({
    name: r.day_label || `#${r.id}`,
    judge: r.avg_judge_score,
    sim: r.avg_similarity,
    degraded: !!r.degradation_detected,
  }));

  async function handleSetup() {
    if (
      setupInfo &&
      !window.confirm("Пересоздать данные? Все прогоны и история будут удалены.")
    )
      return;
    setBusy(true);
    setError("");
    try {
      setSetupInfo(await setupProject());
      setSelectedId(null);
      setDetails(null);
      setDiff(null);
      await refresh();
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleStart() {
    setBusy(true);
    setError("");
    try {
      await createRun({ day_label: label, config: { preset } });
      setLabel(`День ${runs.length + 2}`);
      setSelectedId(null);
      setDetails(null);
      setDiff(null);
      await refresh();
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  }

  function handleSelect(r) {
    if (selectedId === r.id) {
      setSelectedId(null);
      setDetails(null);
      setDiff(null);
      return;
    }
    setSelectedId(r.id);
    setDetails(null);
    setDiff(null);
    fetchRun(r.id)
      .then(setDetails)
      .catch((e) => setError(errText(e)));
    if (r.status === "done" && r.degradation_detected)
      fetchDiff(r.id)
        .then(setDiff)
        .catch(() => {});
  }

  return (
    <div className="container">
      <header>
        <h1>Silent Degradation Detector</h1>
        <p className="subtitle">
          Ловим тихую деградацию качества LLM в «проде» — без ручной разметки
        </p>
      </header>

      {error && <div className="alert">{error}</div>}

      <section className="grid-2">
        <div className="card">
          <h2>1 · Данные</h2>
          <p className="hint">
            Генерирует базу знаний FAQ (~36 статей техподдержки TaskFlow) и
            исторический лог пользовательских запросов (~100 штук).
          </p>
          <button className="btn" onClick={handleSetup} disabled={busy}>
            {setupInfo ? "Пересоздать данные" : "Setup"}
          </button>
          {setupInfo && (
            <div className="setup-info">
              База знаний: <b>{setupInfo.kb_articles}</b> статей · исторических
              запросов: <b>{setupInfo.historical_queries}</b>
            </div>
          )}
        </div>

        <div className="card">
          <h2>2 · Новый прогон («день»)</h2>
          <div className="form-row">
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Метка дня"
            />
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              {PRESETS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
            <button
              className="btn btn-primary"
              onClick={handleStart}
              disabled={busy || anyRunning}
            >
              {anyRunning ? "Прогон выполняется…" : "Запустить прогон"}
            </button>
          </div>
          <p className="hint">
            Каждый прогон заново генерирует adversarial тест-кейсы из
            исторического лога, прогоняет их через «прод-модель» с выбранной
            конфигурацией и оценивает ответы (LLM-judge + semantic similarity).
            Обычно занимает 1–3 минуты.
          </p>
        </div>
      </section>

      <section className="card">
        <h2>3 · Метрики по прогонам во времени</h2>
        {chartData.length === 0 ? (
          <p className="hint">
            Пока нет завершённых прогонов. Нажмите «Setup», затем запустите
            baseline-прогон.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis yAxisId="j" domain={[0, 3]} tickCount={4} />
              <YAxis yAxisId="s" orientation="right" domain={[0, 1]} tickCount={3} />
              <Tooltip />
              <Legend />
              <Line
                yAxisId="j"
                type="monotone"
                dataKey="judge"
                name="Judge score (0–3)"
                stroke="#4f46e5"
                dot={<DegradedDot color="#4f46e5" />}
              />
              <Line
                yAxisId="s"
                type="monotone"
                dataKey="sim"
                name="Semantic similarity (0–1)"
                stroke="#0ea5e9"
                dot={<DegradedDot color="#0ea5e9" />}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
        <p className="hint">Красная точка — прогон с обнаруженной деградацией относительно baseline.</p>
      </section>

      <section className="card">
        <h2>4 · Прогоны</h2>
        <table className="runs-table">
          <thead>
            <tr>
              <th>#</th>
              <th>День</th>
              <th>Конфигурация</th>
              <th>Статус</th>
              <th>Judge ↓</th>
              <th>Similarity ↓</th>
              <th>% факт.&lt;2</th>
              <th>Детекция</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr
                key={r.id}
                className={selectedId === r.id ? "selected" : ""}
                onClick={() => handleSelect(r)}
              >
                <td>{r.id}</td>
                <td>{r.day_label}</td>
                <td className="conf-cell">{r.config_description}</td>
                <td>
                  <span
                    className={`badge ${
                      r.status === "done"
                        ? "badge-green"
                        : r.status === "running"
                          ? "badge-amber"
                          : "badge-red"
                    }`}
                  >
                    {r.status}
                  </span>
                </td>
                <td>{fmt(r.avg_judge_score)}</td>
                <td>{fmt(r.avg_similarity)}</td>
                <td>
                  {r.pct_low_factual == null ? "—" : `${Math.round(r.pct_low_factual * 100)}%`}
                </td>
                <td>
                  {r.degradation_detected == null ? (
                    "—"
                  ) : r.degradation_detected ? (
                    <span className="badge badge-red">⚠ DEGRADED</span>
                  ) : (
                    <span className="badge badge-green">ok</span>
                  )}
                </td>
              </tr>
            ))}
            {runs.length === 0 && (
              <tr>
                <td colSpan={8} className="hint">
                  Нет прогонов.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <p className="hint">
          Кликните по строке — развернутся детали: тест-кейсы, ответы, оценки,
          а для деградировавших прогонов — сравнение с baseline «было → стало».
        </p>
      </section>

      {selectedId != null && (
        <section className="card">
          <h2>Детали прогона #{selectedId}</h2>
          {!details ? (
            <p className="hint">Загрузка…</p>
          ) : (
            <>
              {details.run.status === "error" && (
                <div className="alert">Ошибка прогона: {details.run.error}</div>
              )}
              {details.run.status === "running" && (
                <p className="hint">
                  Прогон ещё выполняется — тест-кейсы появляются по мере обработки…
                </p>
              )}

              {diff && (
                <div className="diff-block">
                  <h3>Сравнение с baseline</h3>
                  {diff.note && <p className="hint">{diff.note}</p>}
                  {diff.detection && !diff.note && (
                    <div className="metric-grid">
                      {Object.values(diff.detection.axes || {}).map((m) => (
                        <MetricCell key={m.metric} m={m} />
                      ))}
                      <MetricCell m={diff.detection.semantic_similarity} />
                      {diff.detection.low_factual_rate && (
                        <div className="metric-cell">
                          <div className="metric-name">{AXIS_NAMES.low_factual_rate}</div>
                          <div className="metric-values">
                            {fmt(diff.detection.low_factual_rate.baseline)} →{" "}
                            {fmt(diff.detection.low_factual_rate.current)}
                          </div>
                          <div className="metric-sub">
                            флаг: {diff.detection.low_factual_rate.flagged ? "да" : "нет"}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  {diff.examples?.length > 0 && diff.detection?.degradation_detected && (
                    <>
                      <h3>Самые сильные просадки «было → стало»</h3>
                      <div className="examples-grid">
                        {diff.examples.map((ex, i) => (
                          <ExampleCard key={i} ex={ex} />
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              <h3>Тест-кейсы (худшие сверху)</h3>
              <div className="cases-list">
                {(details.cases || []).map((c) => (
                  <CaseCard key={c.id} c={c} />
                ))}
                {details.cases?.length === 0 && (
                  <p className="hint">Кейсов пока нет.</p>
                )}
              </div>
            </>
          )}
        </section>
      )}

      <footer className="footer hint">
        LLM-as-judge — инструмент ранней тревоги, а не замена ручному QA: в проде его нужно
        периодически валидировать небольшой human review выборкой.
      </footer>
    </div>
  );
}
