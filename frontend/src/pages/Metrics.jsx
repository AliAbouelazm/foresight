import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { getTrainingHistory, getMetrics } from "../api/client.js";
import s from "../styles/metrics.module.css";

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
  const [history, setHistory] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [tab, setTab] = useState("loss");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [hist, met] = await Promise.all([
          getTrainingHistory(),
          getMetrics("transformer").catch(() => null),
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
  }, []);

  const chartData = history
    ? history.epochs.map((epoch, i) => ({
        epoch,
        train: history.train_loss[i],
        val: history.val_loss[i],
      }))
    : [];

  if (loading) return <div className={s.page}><p className={s.loading}>Loading metrics...</p></div>;
  if (error) return <div className={s.page}><div className={s.error}>{error}</div></div>;

  return (
    <div className={s.page}>
      <div>
        <h1 className={s.title}>Model Performance</h1>
        <p className={s.subtitle}>Training history and test set evaluation for the Transformer model.</p>
      </div>

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
                desc="Final displacement error"
              />
              <MetricCard
                label="Best Val Loss"
                value={metrics.best_val_loss?.toFixed(5)}
                desc="Huber loss on validation set"
              />
              <MetricCard
                label="Test samples"
                value={metrics.n_test?.toLocaleString()}
                desc="15% held-out split"
              />
            </div>
          </div>
        </div>
      )}

      {history && (
        <div className={s.section}>
          <div className={s.sectionTitle}>Training History</div>
          <div className={s.card}>
            <div className={s.tabRow}>
              <button
                className={`${s.tab} ${tab === "loss" ? s.tabActive : ""}`}
                onClick={() => setTab("loss")}
              >
                Loss curves
              </button>
            </div>

            <div className={s.infoRow}>
              <span>Model: <span>{history.model}</span></span>
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
                    tickFormatter={(v) => v.toFixed(4)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#161616",
                      border: "1px solid #252525",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v) => v.toFixed(6)}
                    labelFormatter={(l) => `Epoch ${l}`}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                    formatter={(v) => (v === "train" ? "Train loss" : "Val loss")}
                  />
                  <Line
                    type="monotone"
                    dataKey="train"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="val"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {!history && !metrics && (
        <div className={s.card}>
          <p className={s.loading}>
            No training data available. Run the training pipeline first.
          </p>
        </div>
      )}
    </div>
  );
}
