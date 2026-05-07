import { useRef, useEffect } from "react";

const TRAIL_LEN = 10;

function drawCourt(ctx, w, h) {
  ctx.fillStyle = "#0b1220";
  ctx.fillRect(0, 0, w, h);

  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1.5;

  const m = 0.04;
  const x0 = w * m, y0 = h * m, x1 = w * (1 - m), y1 = h * (1 - m);

  // boundary
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

  // half-court line
  ctx.beginPath(); ctx.moveTo(w / 2, y0); ctx.lineTo(w / 2, y1); ctx.stroke();

  // center circle
  ctx.beginPath(); ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.09, 0, Math.PI * 2); ctx.stroke();

  // three-point arcs
  const r = Math.min(w, h) * 0.27;
  ctx.beginPath(); ctx.arc(x0, h / 2, r, -Math.PI / 2.2, Math.PI / 2.2); ctx.stroke();
  ctx.beginPath(); ctx.arc(x1, h / 2, r, Math.PI - Math.PI / 2.2, Math.PI + Math.PI / 2.2); ctx.stroke();

  // paint areas
  const pw = w * 0.13, ph = h * 0.38;
  ctx.strokeRect(x0, h / 2 - ph / 2, pw, ph);
  ctx.strokeRect(x1 - pw, h / 2 - ph / 2, pw, ph);

  // free-throw circles
  ctx.beginPath(); ctx.arc(x0 + pw, h / 2, ph / 4, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(x1 - pw, h / 2, ph / 4, 0, Math.PI * 2); ctx.stroke();

  ctx.restore();
}

function toC(p, w, h) {
  return { x: p.x * w, y: p.y * h };
}

function drawTrail(ctx, pts, color, maxAlpha = 0.9) {
  if (pts.length < 2) return;
  for (let i = 1; i < pts.length; i++) {
    const alpha = (i / pts.length) * maxAlpha;
    const width = 1 + (i / pts.length) * 3;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pts[i - 1].x, pts[i - 1].y);
    ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
    ctx.restore();
  }
}

function drawGlowDot(ctx, p, color, r = 6) {
  // outer glow
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 16;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.4;
  ctx.beginPath(); ctx.arc(p.x, p.y, r * 1.8, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
  // solid dot
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawDashedLine(ctx, pts, color) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 5]);
  ctx.globalAlpha = 0.45;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();
  ctx.restore();
}

function phaseLabel(ctx, w, phase) {
  const labels = {
    input: { text: "● OBSERVING", color: "#22d3ee" },
    predict: { text: "◆ PREDICTING", color: "#f97316" },
  };
  const l = labels[phase];
  if (!l) return;
  ctx.save();
  ctx.font = "bold 11px -apple-system, sans-serif";
  ctx.fillStyle = l.color;
  ctx.globalAlpha = 0.9;
  ctx.fillText(l.text, 14, 22);
  ctx.restore();
}

export default function TrajectoryCanvas({ inputTrack, targetTrack, prediction, frameIdx }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const dpr = devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;

    drawCourt(ctx, w, h);

    if (!inputTrack) return;

    const fi = frameIdx ?? inputTrack.length - 1;
    const inputLen = inputTrack.length;
    const inPredPhase = fi >= inputLen;

    // -- Input trail --
    const inputVisible = inputTrack.slice(0, Math.min(fi + 1, inputLen)).map(p => toC(p, w, h));
    const trailStart = Math.max(0, inputVisible.length - TRAIL_LEN);
    drawTrail(ctx, inputVisible.slice(trailStart), "#22d3ee");

    if (inputVisible.length > 0) {
      drawGlowDot(ctx, inputVisible[inputVisible.length - 1], "#22d3ee", 5);
    }

    if (!inPredPhase) {
      phaseLabel(ctx, w, "input");
      return;
    }

    phaseLabel(ctx, w, "predict");

    const predFrame = fi - inputLen;

    // -- Ground truth (dashed) --
    if (targetTrack) {
      const targetPts = targetTrack.slice(0, predFrame + 1).map(p => toC(p, w, h));
      // full ghost path
      drawDashedLine(ctx, targetTrack.map(p => toC(p, w, h)), "#a3e635");
      // drawn so far
      if (targetPts.length > 1) {
        ctx.save(); ctx.setLineDash([]); ctx.restore();
        drawTrail(ctx, targetPts, "#a3e635", 0.7);
      }
      if (targetPts.length > 0) drawGlowDot(ctx, targetPts[targetPts.length - 1], "#a3e635", 4);
    }

    // -- Prediction --
    if (prediction) {
      const allPredPts = prediction.mean.map(([x, y]) => toC({ x, y }, w, h));
      const { std } = prediction;
      drawDashedLine(ctx, allPredPts, "#f97316");

      const predPts = allPredPts.slice(0, predFrame + 1);
      drawTrail(ctx, predPts, "#f97316");

      // uncertainty halos
      for (let i = 0; i < predPts.length; i++) {
        const sx = (std[i][0] + std[i][1]) / 2;
        const radius = Math.max(6, sx * w * 14);
        ctx.save();
        ctx.fillStyle = "rgba(249,115,22,0.07)";
        ctx.beginPath();
        ctx.arc(predPts[i].x, predPts[i].y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      if (predPts.length > 0) drawGlowDot(ctx, predPts[predPts.length - 1], "#f97316", 5);
    }
  }, [inputTrack, targetTrack, prediction, frameIdx]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}
