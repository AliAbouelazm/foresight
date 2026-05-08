import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { getTrainingHistory, getMetrics } from "../api/client.js";
import s from "../styles/metrics.module.css";

const MODELS = [
  { key: "cnn", label: "TemporalCNN" },
  { key: "transformer", label: "Seq2Seq Transformer" },
];

function MetricCard({ label, value, desc }) {
  return (
    <div className={s.metric}>
      <span className={s.metricLabel}>{label}</span>
      <span className={s.metricValue}>{value ?? "--"}</span>
      {desc && <span className={s.metricDesc}>{desc}</span>}
    </div>
  );
}

export default function Metrics() {
  const [activeModel, setActiveModel] = useState("cnn");
  const [history, setHistory] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      setHistory(null);
      setMetrics(null);
      try {
        const [hist, met] = await Promise.all([
          getTrainingHistory(activeModel).catch(() => null),
          getMetrics(activeModel).catch(() => null),
        ]);
        setHistory(hist);
        setMetrics(met);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [activeModel]);

  const chartData = history
    ? history.epochs.map((epoch, i) => ({
        epoch,
        train: history.train_loss[i],
        val: history.val_loss[i],
      }))
    : [];

  return (
    <div className={s.page}>
      <div>
        <h1 className={s.title}>Model Performance</h1>
        <p className={s.subtitle}>
          Test set evaluation and training history. 50 input frames, 50 predicted frames.
        </p>
      </div>

      <div className={s.tabRow}>
        {MODELS.map(m => (
          <button
            key={m.key}
            className={`${s.tab} ${activeModel === m.key ? s.tabActive : ""}`}
            onClick={() => setActiveModel(m.key)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {error && <div className={s.error}>{error}</div>}

      {loading ? (
        <p className={s.loading}>Loading...</p>
      ) : (
        <>
          {metrics && (
            <div className={s.section}>
              <div className={s.sectionTitle}>Test Set Metrics</div>
              <div className={s.card}>
                <div className={s.metricGrid}>
                  <MetricCard
                    label="ADE"
                    value={metrics.ade.toFixed(5)}
                    desc="Avg displacement error (normalized)"
                  />
                  <MetricCard
                    label="FDE"
                    value={metrics.fde.toFixed(5)}
                    desc="Final frame displacement error"
                  />
                  <MetricCard
                    label="Best Val Loss"
                    value={metrics.best_val_loss?.toFixed(5)}
                    desc="Huber loss on validation set"
                  />
                  <MetricCard
                    label="Test Samples"
                    value={metrics.n_test?.toLocaleString()}
                    desc="15% held-out split"
                  />
                </div>
              </div>
            </div>
          )}

          {history ? (
            <div className={s.section}>
              <div className={s.sectionTitle}>Training History</div>
              <div className={s.card}>
                <div className={s.infoRow}>
                  <span>Epochs trained: <span>{history.epochs.length}</span></span>
                  <span>Best val loss: <span>{history.best_val_loss?.toFixed(5)}</span></span>
                </div>
                <div className={s.chartWrap} style={{ marginTop: 16 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 4, right: 12, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="epoch"
                        stroke="#555"
                        tick={{ fill: "#737373", fontSize: 11 }}
                        label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: "#555", fontSize: 11 }}
                      />
                      <YAxis
                        stroke="#555"
                        tick={{ fill: "#737373", fontSize: 11 }}
                        tickFormatter={v => v.toFixed(4)}
                      />
                      <Tooltip
                        contentStyle={{ background: "#161616", border: "1px solid #252525", borderRadius: 8, fontSize: 12 }}
                        formatter={v => v.toFixed(6)}
                        labelFormatter={l => `Epoch ${l}`}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                        formatter={v => v === "train" ? "Train loss" : "Val loss"}
                      />
                      <Line type="monotone" dataKey="train" stroke="#22d3ee" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                      <Line type="monotone" dataKey="val" stroke="#f97316" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            <div className={s.card}>
              <p className={s.loading}>
                Training history not available for this model. Retrain to generate it.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
