import { useRef, useEffect, useCallback } from "react";

const FPS = 30;
const TRAIL = 16;
const GAP = 3;

function court(ctx, w, h) {
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, w, h);
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  const m = 0.04;
  const x0 = w * m, y0 = h * m, x1 = w * (1 - m), y1 = h * (1 - m);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.beginPath(); ctx.moveTo(w / 2, y0); ctx.lineTo(w / 2, y1); ctx.stroke();
  ctx.beginPath(); ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.09, 0, Math.PI * 2); ctx.stroke();
  const r = Math.min(w, h) * 0.27;
  ctx.beginPath(); ctx.arc(x0, h / 2, r, -Math.PI / 2.2, Math.PI / 2.2); ctx.stroke();
  ctx.beginPath(); ctx.arc(x1, h / 2, r, Math.PI - Math.PI / 2.2, Math.PI + Math.PI / 2.2); ctx.stroke();
  const pw = w * 0.13, ph = h * 0.38;
  ctx.strokeRect(x0, h / 2 - ph / 2, pw, ph);
  ctx.strokeRect(x1 - pw, h / 2 - ph / 2, pw, ph);
  ctx.restore();
}

function tc(p, w, h) {
  return [p.x * w, p.y * h];
}

function trail(ctx, pts, color) {
  if (pts.length < 2) return;
  for (let i = 1; i < pts.length; i++) {
    const t = i / (pts.length - 1);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 + t * 2.5;
    ctx.globalAlpha = 0.12 + t * 0.82;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pts[i - 1][0], pts[i - 1][1]);
    ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.restore();
  }
}

function dot(ctx, p, color, r = 5) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function panelLabel(ctx, text, color, w) {
  ctx.save();
  ctx.font = "bold 11px -apple-system, sans-serif";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.85;
  const tw = ctx.measureText(text).width;
  ctx.fillText(text, (w - tw) / 2, 20);
  ctx.restore();
}

function drawPanel(ctx, w, h, inputTrack, futureTrack, frame, futureColor, label) {
  court(ctx, w, h);
  panelLabel(ctx, label, futureColor, w);

  const inputLen = inputTrack.length;
  const inPred = frame >= inputLen;
  const predFrame = frame - inputLen;

  // faded full input when in prediction phase
  if (inPred) {
    ctx.save();
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.14;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    const pts = inputTrack.map(p => tc(p, w, h));
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.restore();
  }

  // input trail
  const iEnd = Math.min(frame + 1, inputLen);
  const iStart = Math.max(0, iEnd - TRAIL);
  const iPts = inputTrack.slice(iStart, iEnd).map(p => tc(p, w, h));
  trail(ctx, iPts, "#22d3ee");
  if (iPts.length) dot(ctx, iPts[iPts.length - 1], "#22d3ee", inPred ? 3 : 5);

  if (!inPred || !futureTrack) return;

  // future trail — starts exactly from last input point
  const anchor = tc(inputTrack[inputLen - 1], w, h);
  const futurePts = futureTrack.slice(0, predFrame + 1).map(p => tc(p, w, h));
  const combined = [anchor, ...futurePts];
  const visible = combined.slice(Math.max(0, combined.length - TRAIL));
  trail(ctx, visible, futureColor);
  if (visible.length > 1) dot(ctx, visible[visible.length - 1], futureColor, 5);
}

function renderFrame(ctx, W, H, inputTrack, targetTrack, prediction, frame) {
  const pw = (W - GAP) / 2;

  // Left: actual
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, pw, H); ctx.clip();
  drawPanel(ctx, pw, H, inputTrack, targetTrack, frame, "#a3e635", "WHAT ACTUALLY HAPPENED");
  ctx.restore();

  // Divider
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(pw, 0, GAP, H);

  // Right: predicted
  ctx.save();
  ctx.translate(pw + GAP, 0);
  ctx.beginPath(); ctx.rect(0, 0, pw, H); ctx.clip();
  const predTrack = prediction?.mean.map(([x, y]) => ({ x, y }));
  drawPanel(ctx, pw, H, inputTrack, predTrack ?? null, frame, "#f97316", "WHAT THE MODEL PREDICTED");
  ctx.restore();

  // Phase indicator strip at bottom
  const inputLen = inputTrack.length;
  const inPred = frame >= inputLen;
  ctx.save();
  ctx.fillStyle = inPred ? "rgba(249,115,22,0.12)" : "rgba(34,211,238,0.1)";
  ctx.fillRect(0, H - 3, W, 3);
  ctx.restore();
}

export default function TrajectoryCanvas({
  inputTrack, targetTrack, prediction,
  playing, scrubTo,
  onFrame, onEnd,
}) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const lastRef = useRef(null);
  const frameRef = useRef(0);
  const dataRef = useRef({ inputTrack, targetTrack, prediction });

  useEffect(() => {
    dataRef.current = { inputTrack, targetTrack, prediction };
  }, [inputTrack, targetTrack, prediction]);

  const redraw = useCallback((frame) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width * dpr)) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.scale(dpr, dpr);
    }
    const { inputTrack, targetTrack, prediction } = dataRef.current;
    if (!inputTrack) {
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(0, 0, rect.width, rect.height);
      return;
    }
    renderFrame(ctx, rect.width, rect.height, inputTrack, targetTrack, prediction, frame);
  }, []);

  useEffect(() => {
    frameRef.current = 0;
    lastRef.current = null;
    redraw(0);
    onFrame?.(0);
  }, [inputTrack, targetTrack, prediction]);

  useEffect(() => {
    if (scrubTo != null) {
      frameRef.current = scrubTo;
      redraw(scrubTo);
    }
  }, [scrubTo]);

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(rafRef.current);
      lastRef.current = null;
      return;
    }
    function tick(ts) {
      if (!lastRef.current) lastRef.current = ts;
      const elapsed = ts - lastRef.current;
      if (elapsed >= 1000 / FPS) {
        lastRef.current = ts - (elapsed % (1000 / FPS));
        const { inputTrack, targetTrack } = dataRef.current;
        const total = inputTrack ? inputTrack.length + (targetTrack?.length ?? 0) : 0;
        if (frameRef.current < total - 1) {
          frameRef.current += 1;
          redraw(frameRef.current);
          onFrame?.(frameRef.current);
        } else {
          onEnd?.();
          return;
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, redraw, onFrame, onEnd]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}
