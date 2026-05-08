import { useState, useRef, useCallback } from "react";
import TrajectoryCanvas from "../components/TrajectoryCanvas.jsx";
import s from "../styles/demo.module.css";
import { getRandomTrajectory, predict } from "../api/client.js";

function buildInput(trajectory) {
  const pts = trajectory.input;
  return pts.map((p, i) => {
    const prev = i > 0 ? pts[i - 1] : p;
    return [p.x, p.y, p.x - prev.x, p.y - prev.y];
  });
}

export default function Demo() {
  const [modelName, setModelName] = useState("transformer");
  const [trajectory, setTrajectory] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(false);

  // frame display only — updated from canvas callback, not setState per frame
  const [displayFrame, setDisplayFrame] = useState(0);
  const [scrubTo, setScrubTo] = useState(null);

  const totalFrames = trajectory
    ? trajectory.input.length + trajectory.target.length
    : 0;
  const inputLen = trajectory?.input?.length ?? 20;

  const onFrame = useCallback((n) => {
    // Update the scrubber every frame (lightweight — just a number update)
    setDisplayFrame(n);
  }, []);

  const onEnd = useCallback(() => setPlaying(false), []);

  async function loadSample() {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setPlaying(false);
    setScrubTo(null);
    setDisplayFrame(0);
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

  function handleScrub(e) {
    const n = Number(e.target.value);
    setPlaying(false);
    setScrubTo(n);
    setDisplayFrame(n);
  }

  function replay() {
    setScrubTo(0);
    setDisplayFrame(0);
    // small timeout so canvas resets before playing starts
    setTimeout(() => { setScrubTo(null); setPlaying(true); }, 50);
  }

  const phase = trajectory
    ? displayFrame < inputLen ? "input" : "predict"
    : null;

  const confidence = prediction
    ? Math.round(
        Math.max(0, Math.min(1, 1 - Math.max(...prediction.std.map(([a, b]) => (a + b) / 2)) * 20)) * 100
      )
    : null;

  return (
    <div className={s.page}>
      <div className={s.header}>
        <div>
          <h1 className={s.title}>Player Trajectory Prediction</h1>
          <p className={s.subtitle}>
            The model watches 20 frames of a player moving, then predicts the next 20.
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
              playing={playing}
              scrubTo={scrubTo}
              onFrame={onFrame}
              onEnd={onEnd}
            />
            {!trajectory && !loading && (
              <div className={s.emptyOverlay}>Press "Load Sample" to begin</div>
            )}
          </div>

          {trajectory && (
            <input
              type="range"
              min={0}
              max={totalFrames - 1}
              value={displayFrame}
              onChange={handleScrub}
              style={{ width: "100%", accentColor: "var(--accent)" }}
            />
          )}

          <div className={s.legendRow}>
            <div className={s.legendItem}>
              <div className={s.dot} style={{ background: "#22d3ee" }} />
              Input (both panels)
            </div>
            <div className={s.legendItem}>
              <div className={s.dot} style={{ background: "#a3e635" }} />
              Actual (left panel)
            </div>
            <div className={s.legendItem}>
              <div className={s.dot} style={{ background: "#f97316" }} />
              Predicted (right panel)
            </div>
          </div>
        </div>

        <div className={s.sidebar}>
          <div className={s.phaseCard} style={{
            borderLeftColor: phase === "input" ? "#22d3ee" : phase === "predict" ? "#f97316" : "var(--border)"
          }}>
            <div className={s.phaseLabel} style={{
              color: phase === "input" ? "#22d3ee" : phase === "predict" ? "#f97316" : "var(--text-secondary)"
            }}>
              {phase === "input" && "Phase 1/2: Observing"}
              {phase === "predict" && "Phase 2/2: Predicting"}
              {!phase && "How it works"}
            </div>
            <p className={s.phaseDesc}>
              {phase === "input" && "Both panels show the same cyan trail — the player's last 20 frames of movement that the model uses as input."}
              {phase === "predict" && "Left panel shows where the player actually went (green). Right panel shows what the model predicted (orange). Compare them directly."}
              {!phase && "Load a sample to see the split-panel comparison: actual path on the left vs model prediction on the right."}
            </p>
          </div>

          {trajectory && (
            <div className={s.card}>
              <div className={s.cardTitle}>Playback</div>
              <div className={s.statGrid}>
                <div className={s.stat}>
                  <span className={s.statLabel}>Frame</span>
                  <span className={s.statValue}>
                    {displayFrame}<span className={s.statUnit}>/{totalFrames - 1}</span>
                  </span>
                </div>
                <div className={s.stat}>
                  <span className={s.statLabel}>Phase</span>
                  <span className={s.statValue} style={{ fontSize: 13 }}>
                    {phase === "input" ? "Input" : "Predict"}
                  </span>
                </div>
              </div>
            </div>
          )}

          {confidence != null && (
            <div className={s.card}>
              <div className={s.cardTitle}>Confidence</div>
              <div className={s.confidenceValue} style={{
                color: confidence > 66 ? "#22d3ee" : confidence > 33 ? "#eab308" : "#f97316"
              }}>
                {confidence}%
              </div>
              <div className={s.confidenceTrack}>
                <div className={s.confidenceFill} style={{
                  width: `${confidence}%`,
                  background: confidence > 66 ? "#22d3ee" : confidence > 33 ? "#eab308" : "#f97316",
                }} />
              </div>
              <p className={s.confidenceNote}>
                Estimated via {prediction.samples.length} MC dropout passes
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
