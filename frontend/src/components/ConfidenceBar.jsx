import s from "../styles/components.module.css";

export default function ConfidenceBar({ label, value, max = 1 }) {
  const pct = Math.min(100, (value / max) * 100);
  const level = pct > 66 ? "High" : pct > 33 ? "Medium" : "Low";
  const cls = `${s.confidenceBar} ${
    level === "High" ? s.confidenceHigh : level === "Medium" ? s.confidenceMedium : s.confidenceLow
  }`;

  return (
    <div className={cls}>
      <div className={s.confidenceLabel}>
        <span>{label}</span>
        <span>{value.toFixed(4)}</span>
      </div>
      <div className={s.confidenceTrack}>
        <div className={s.confidenceFill} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
