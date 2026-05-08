import { useRef, useEffect, useCallback } from "react";

const FPS = 30;
const TRAIL_FRAMES = 14;

function drawCourt(ctx, w, h) {
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

function toC(p, w, h) {
  return [p.x * w, p.y * h];
}

function drawSmoothTrail(ctx, pts, color) {
  if (pts.length < 2) return;
  for (let i = 1; i < pts.length; i++) {
    const t = i / (pts.length - 1);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 + t * 2.5;
    ctx.globalAlpha = 0.15 + t * 0.75;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pts[i - 1][0], pts[i - 1][1]);
    ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.restore();
  }
}

function drawDot(ctx, p, color, r = 5) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawLabel(ctx, text, color, x, y) {
  ctx.save();
  ctx.font = "bold 11px -apple-system, sans-serif";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.85;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function render(ctx, w, h, inputTrack, targetTrack, prediction, frame) {
  drawCourt(ctx, w, h);
  if (!inputTrack) return;

  const inputLen = inputTrack.length;
  const inPred = frame >= inputLen;
  const predFrame = frame - inputLen;

  // Full faded input (always visible after input phase)
  if (inPred) {
    const fadedPts = inputTrack.map(p => toC(p, w, h));
    ctx.save();
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = 0.18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(fadedPts[0][0], fadedPts[0][1]);
    for (let i = 1; i < fadedPts.length; i++) ctx.lineTo(fadedPts[i][0], fadedPts[i][1]);
    ctx.stroke();
    ctx.restore();
  }

  // Input trail (last TRAIL_FRAMES frames)
  const inputEnd = Math.min(frame + 1, inputLen);
  const trailStart = Math.max(0, inputEnd - TRAIL_FRAMES);
  const visibleInput = inputTrack.slice(trailStart, inputEnd).map(p => toC(p, w, h));
  drawSmoothTrail(ctx, visibleInput, "#22d3ee");
  if (visibleInput.length) drawDot(ctx, visibleInput[visibleInput.length - 1], "#22d3ee", 5);

  if (!inPred) {
    drawLabel(ctx, "OBSERVING PLAYER MOVEMENT", "#22d3ee", 12, 20);
    return;
  }

  drawLabel(ctx, "PREDICTING NEXT MOVEMENT", "#f97316", 12, 20);

  // Ground truth trail
  if (targetTrack) {
    const gtEnd = Math.min(predFrame + 1, targetTrack.length);
    const gtTrail = targetTrack.slice(Math.max(0, gtEnd - TRAIL_FRAMES), gtEnd).map(p => toC(p, w, h));
    drawSmoothTrail(ctx, gtTrail, "#a3e635");
    if (gtTrail.length) drawDot(ctx, gtTrail[gtTrail.length - 1], "#a3e635", 4);
    if (gtEnd >= 2) {
      drawLabel(ctx, "actual", "#a3e635", gtTrail[gtTrail.length - 1][0] + 8, gtTrail[gtTrail.length - 1][1] - 4);
    }
  }

  // Prediction trail
  if (prediction) {
    const predEnd = Math.min(predFrame + 1, prediction.mean.length);
    const rawPts = [
      toC(inputTrack[inputTrack.length - 1], w, h),
      ...prediction.mean.slice(0, predEnd).map(([x, y]) => toC({ x, y }, w, h)),
    ];
    const predTrail = rawPts.slice(Math.max(0, rawPts.length - TRAIL_FRAMES));
    drawSmoothTrail(ctx, predTrail, "#f97316");
    if (predTrail.length) drawDot(ctx, predTrail[predTrail.length - 1], "#f97316", 5);
    if (predEnd >= 2) {
      drawLabel(ctx, "predicted", "#f97316", predTrail[predTrail.length - 1][0] + 8, predTrail[predTrail.length - 1][1] - 4);
    }
  }
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

  // keep data ref current
  useEffect(() => {
    dataRef.current = { inputTrack, targetTrack, prediction };
  }, [inputTrack, targetTrack, prediction]);

  const redraw = useCallback((frame) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }
    const { inputTrack, targetTrack, prediction } = dataRef.current;
    render(ctx, rect.width, rect.height, inputTrack, targetTrack, prediction, frame);
  }, []);

  // reset + draw when data changes
  useEffect(() => {
    frameRef.current = 0;
    lastRef.current = null;
    redraw(0);
    onFrame?.(0);
  }, [inputTrack, targetTrack, prediction]);

  // scrub
  useEffect(() => {
    if (scrubTo != null) {
      frameRef.current = scrubTo;
      redraw(scrubTo);
    }
  }, [scrubTo]);

  // RAF loop — no React setState per tick
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

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
