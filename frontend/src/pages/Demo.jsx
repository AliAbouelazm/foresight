import { useState, useEffect, useRef } from "react";
import TrajectoryCanvas from "../components/TrajectoryCanvas.jsx";
import s from "../styles/demo.module.css";
import { getRandomTrajectory, predict } from "../api/client.js";

const FPS = 8;

function buildInput(trajectory) {
  const pts = trajectory.input;
  return pts.map((p, i) => {
    const prev = i > 0 ? pts[i - 1] : p;
    return [p.x, p.y, p.x - prev.x, p.y - prev.y];
  });
}

const PHASES = [
  {
    key: "input",
    color: "#22d3ee",
    label: "Observing",
    desc: "Watching the last 20 frames of a player's movement. This becomes the model's input.",
  },
  {
    key: "predict",
    color: "#f97316",
    label: "Predicting",
    desc: "Model predicts the next 20 frames. Orange = prediction, dashed green = where they actually went.",
  },
];

export default function Demo() {
  const [modelName, setModelName] = useState("transformer");
  const [trajectory, setTrajectory] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef(null);
  const lastRef = useRef(null);

  const inputLen = trajectory?.input?.length ?? 20;
  const totalFrames = trajectory ? inputLen + trajectory.target.length : 0;
  const phase = !trajectory ? null : frame < inputLen ? 0 : 1;

  useEffect(() => {
    if (!playing || !trajectory) return;
    function tick(ts) {
      if (!lastRef.current) lastRef.current = ts;
      if (ts - lastRef.current >= 1000 / FPS) {
        lastRef.current = ts;
        setFrame(f => {
          if (f + 1 >= totalFrames) { setPlaying(false); return f; }
          return f + 1;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, totalFrames, trajectory]);

  async function loadSample() {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setPlaying(false);
    setFrame(0);
    lastRef.current = null;
    try {
      const traj = await getRandomTrajectory();
      setTrajectory(traj);
      const pred = await predict(buildInput(traj), modelName);
      setPrediction(pred);
      setPlaying(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function replay() {
    setFrame(0);
    lastRef.current = null;
    setPlaying(true);
  }

  const maxUncertainty = prediction
    ? Math.max(...prediction.std.map(([sx, sy]) => (sx + sy) / 2))
    : null;
  const confidence = maxUncertainty != null
    ? Math.round(Math.max(0, Math.min(1, 1 - maxUncertainty * 20)) * 100)
    : null;

  const currentPhase = phase != null ? PHASES[phase] : null;

  return (
    <div className={s.page}>
      <div className={s.header}>
        <div>
          <h1 className={s.title}>Player Trajectory Prediction</h1>
          <p className={s.subtitle}>
            Given a player's last 20 frames of movement, predict where they go next.
          </p>
        </div>
        <div className={s.controls}>
          <select className={s.select} value={modelName} onChange={e => setModelName(e.target.value)}>
            <option value="transformer">Transformer</option>
            <option value="cnn">TemporalCNN</option>
          </select>
          <button className={s.btnPrimary} onClick={loadSample} disabled={loading}>
            {loading ? "Loading..." : "Load Sample"}
          </button>
          {trajectory && !loading && (
            <button className={s.btnSecondary} onClick={replay}>Replay</button>
          )}
        </div>
      </div>

      {error && <div className={s.error}>{error}</div>}

      <div className={s.main}>
        <div className={s.canvasCard}>
          <div className={s.canvasWrap}>
            <TrajectoryCanvas
              inputTrack={trajectory?.input}
              targetTrack={trajectory?.target}
              prediction={prediction}
              frameIdx={frame}
            />
            {!trajectory && !loading && (
              <div className={s.emptyOverlay}>
                <span>Press "Load Sample" to begin</span>
              </div>
            )}
          </div>

          <div className={s.legendRow}>
            <div className={s.legendItem}>
              <div className={s.dot} style={{ background: "#22d3ee" }} />
              <span>Past movement <em>(input)</em></span>
            </div>
            <div className={s.legendItem}>
              <div className={s.dot} style={{ background: "#f97316" }} />
              <span>Predicted path</span>
            </div>
            <div className={s.legendItem}>
              <div className={s.dot} style={{ background: "#a3e635" }} />
              <span>Actual path <em>(ground truth)</em></span>
            </div>
          </div>

          {trajectory && (
            <input
              type="range"
              min={0}
              max={totalFrames - 1}
              value={frame}
              onChange={e => { setPlaying(false); setFrame(Number(e.target.value)); }}
              style={{ width: "100%", accentColor: "var(--accent)" }}
            />
          )}
        </div>

        <div className={s.sidebar}>
          <div className={s.phaseCard} style={{ borderColor: currentPhase?.color ?? "var(--border)" }}>
            <div className={s.phaseLabel} style={{ color: currentPhase?.color ?? "var(--text-secondary)" }}>
              {currentPhase ? `Phase ${phase + 1}/2: ${currentPhase.label}` : "Ready"}
            </div>
            <p className={s.phaseDesc}>
              {currentPhase
                ? currentPhase.desc
                : "Load a sample to see how the model predicts a player's next 20 frames of movement based on their history."}
            </p>
          </div>

          {trajectory && (
            <div className={s.card}>
              <div className={s.cardTitle}>Playback</div>
              <div className={s.statGrid}>
                <div className={s.stat}>
                  <span className={s.statLabel}>Frame</span>
                  <span className={s.statValue}>{frame}<span className={s.statUnit}>/{totalFrames - 1}</span></span>
                </div>
                <div className={s.stat}>
                  <span className={s.statLabel}>Phase</span>
                  <span className={s.statValue} style={{ fontSize: 13 }}>
                    {frame < inputLen ? "Input" : "Predict"}
                  </span>
                </div>
              </div>
            </div>
          )}

          {prediction && confidence != null && (
            <div className={s.card}>
              <div className={s.cardTitle}>Model Confidence</div>
              <div className={s.confidenceValue} style={{
                color: confidence > 66 ? "#22d3ee" : confidence > 33 ? "#eab308" : "#f97316"
              }}>
                {confidence}%
              </div>
              <div className={s.confidenceTrack}>
                <div
                  className={s.confidenceFill}
                  style={{
                    width: `${confidence}%`,
                    background: confidence > 66 ? "#22d3ee" : confidence > 33 ? "#eab308" : "#f97316",
                  }}
                />
              </div>
              <p className={s.confidenceNote}>
                Based on {prediction.samples.length} MC dropout passes.<br />
                Lower uncertainty = higher confidence.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
