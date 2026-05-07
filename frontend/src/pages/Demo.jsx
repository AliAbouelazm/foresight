import { useState, useEffect, useRef, useCallback } from "react";
import TrajectoryCanvas from "../components/TrajectoryCanvas.jsx";
import ConfidenceBar from "../components/ConfidenceBar.jsx";
import s from "../styles/demo.module.css";
import cs from "../styles/components.module.css";
import { getRandomTrajectory, predict } from "../api/client.js";

const FPS = 10;

function buildInput(trajectory) {
  // trajectory.input is [{x, y}, ...] length 20
  // compute dx, dy
  const pts = trajectory.input;
  return pts.map((p, i) => {
    const prev = i > 0 ? pts[i - 1] : p;
    return [p.x, p.y, p.x - prev.x, p.y - prev.y];
  });
}

function meanStd(prediction) {
  if (!prediction) return { ade: null, maxUncertainty: null };
  const { mean, std, samples } = prediction;
  // ADE against mean
  let totalDist = 0;
  for (const [x, y] of mean) totalDist += Math.sqrt(x * x + y * y);
  const maxUncertainty = Math.max(...std.map(([sx, sy]) => (sx + sy) / 2));
  return { maxUncertainty };
}

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

  const totalFrames = trajectory ? trajectory.input.length + trajectory.target.length : 0;

  // animation loop
  useEffect(() => {
    if (!playing || !trajectory) return;

    function tick(ts) {
      if (!lastRef.current) lastRef.current = ts;
      if (ts - lastRef.current >= 1000 / FPS) {
        lastRef.current = ts;
        setFrame((f) => {
          if (f + 1 >= totalFrames) {
            setPlaying(false);
            return f;
          }
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

      const inputSeq = buildInput(traj);
      const pred = await predict(inputSeq, modelName);
      setPrediction(pred);
      setPlaying(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleScrub(e) {
    setPlaying(false);
    setFrame(Number(e.target.value));
  }

  function replay() {
    setFrame(0);
    lastRef.current = null;
    setPlaying(true);
  }

  const { maxUncertainty } = meanStd(prediction);
  const confidence = prediction
    ? Math.max(0, 1 - (maxUncertainty ?? 0) * 20)
    : null;

  const inputTrack = trajectory?.input;
  const targetTrack = trajectory?.target;

  return (
    <div className={s.page}>
      <div className={s.header}>
        <div>
          <h1 className={s.title}>Trajectory Prediction</h1>
          <p className={s.subtitle}>
            Load a test sample, watch the model predict the next 20 frames in real time.
          </p>
        </div>
        <div className={s.controls}>
          <select
            className={s.select}
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
          >
            <option value="transformer">Transformer</option>
            <option value="cnn">TemporalCNN</option>
          </select>
          <button className={s.btnPrimary} onClick={loadSample} disabled={loading}>
            {loading ? "Loading..." : "Load Sample"}
          </button>
          {trajectory && !loading && (
            <button className={s.btnSecondary} onClick={replay}>
              Replay
            </button>
          )}
        </div>
      </div>

      {error && <div className={s.error}>{error}</div>}

      <div className={s.main}>
        <div className={s.canvasCard}>
          <div className={s.canvasWrap}>
            <TrajectoryCanvas
              inputTrack={inputTrack}
              targetTrack={targetTrack}
              prediction={prediction}
              frameIdx={frame}
            />
            {playing && <span className={s.playingBadge}>LIVE</span>}
          </div>

          {trajectory && (
            <input
              type="range"
              min={0}
              max={totalFrames - 1}
              value={frame}
              onChange={handleScrub}
              style={{ width: "100%", accentColor: "var(--accent)" }}
            />
          )}

          <div className={cs.legend}>
            <div className={cs.legendItem}>
              <div className={cs.legendDot} style={{ background: "#22d3ee" }} />
              Input history
            </div>
            <div className={cs.legendItem}>
              <div className={cs.legendDot} style={{ background: "#a3e635" }} />
              Ground truth
            </div>
            <div className={cs.legendItem}>
              <div className={cs.legendDot} style={{ background: "#f97316" }} />
              Prediction
            </div>
          </div>
        </div>

        <div className={s.sidebar}>
          <div className={s.card}>
            <div className={s.cardTitle}>Stats</div>
            {trajectory ? (
              <div className={s.statGrid}>
                <div className={s.stat}>
                  <span className={s.statLabel}>Frame</span>
                  <span className={s.statValue}>
                    {frame}<span className={s.statUnit}>/ {totalFrames - 1}</span>
                  </span>
                </div>
                <div className={s.stat}>
                  <span className={s.statLabel}>Model</span>
                  <span className={s.statValue} style={{ fontSize: 13 }}>
                    {modelName === "transformer" ? "Transformer" : "CNN"}
                  </span>
                </div>
                <div className={s.stat}>
                  <span className={s.statLabel}>Input frames</span>
                  <span className={s.statValue}>20</span>
                </div>
                <div className={s.stat}>
                  <span className={s.statLabel}>Predicted</span>
                  <span className={s.statValue}>20<span className={s.statUnit}>frames</span></span>
                </div>
              </div>
            ) : (
              <p className={s.empty}>Load a sample to begin</p>
            )}
          </div>

          {prediction && (
            <div className={s.card}>
              <div className={s.cardTitle}>Uncertainty</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <ConfidenceBar
                  label="Confidence"
                  value={confidence ?? 0}
                  max={1}
                />
                <ConfidenceBar
                  label="Max std"
                  value={maxUncertainty ?? 0}
                  max={0.05}
                />
              </div>
              <p style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 10 }}>
                Estimated via {prediction.samples.length} MC dropout passes
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
