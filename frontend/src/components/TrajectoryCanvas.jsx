import { useRef, useEffect } from "react";

const COLORS = {
  input: "#22d3ee",
  target: "#a3e635",
  prediction: "#f97316",
  uncertainty: "rgba(249, 115, 22, 0.18)",
};

function toCanvas(point, w, h) {
  return { x: point.x * w, y: point.y * h };
}

function drawPath(ctx, points, color, lineWidth = 2, alpha = 1) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawDot(ctx, p, color, r = 4) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

export default function TrajectoryCanvas({ inputTrack, targetTrack, prediction, frameIdx }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width: w, height: h } = canvas.getBoundingClientRect();
    canvas.width = w * devicePixelRatio;
    canvas.height = h * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    ctx.clearRect(0, 0, w, h);

    // faint court grid lines
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo((w / 4) * i, 0);
      ctx.lineTo((w / 4) * i, h);
      ctx.stroke();
    }
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (h / 3) * i);
      ctx.lineTo(w, (h / 3) * i);
      ctx.stroke();
    }
    ctx.restore();

    const toC = (p) => toCanvas(p, w, h);
    const fi = frameIdx ?? (inputTrack ? inputTrack.length - 1 : 0);

    // input history (full)
    if (inputTrack) {
      const pts = inputTrack.map(toC);
      drawPath(ctx, pts, COLORS.input, 1.5, 0.5);
      if (fi < pts.length) {
        drawPath(ctx, pts.slice(0, fi + 1), COLORS.input, 2);
        drawDot(ctx, pts[fi], COLORS.input, 5);
      } else {
        drawPath(ctx, pts, COLORS.input, 2);
        drawDot(ctx, pts[pts.length - 1], COLORS.input, 5);
      }
    }

    const predOffset = inputTrack ? inputTrack.length : 0;

    // ground truth future (target)
    if (targetTrack) {
      const pts = targetTrack.map(toC);
      const showCount = Math.max(0, fi - predOffset + 1);
      drawPath(ctx, pts, COLORS.target, 1.5, 0.35);
      if (showCount > 0) drawPath(ctx, pts.slice(0, showCount), COLORS.target, 2);
      if (showCount > 0) drawDot(ctx, pts[showCount - 1], COLORS.target, 4);
    }

    // prediction
    if (prediction) {
      const { mean, std } = prediction;
      const pts = mean.map(([x, y]) => toC({ x, y }));
      const showCount = Math.max(0, fi - predOffset + 1);

      // uncertainty envelope using std
      if (showCount > 1) {
        ctx.save();
        ctx.fillStyle = COLORS.uncertainty;
        ctx.beginPath();
        for (let i = 0; i < showCount; i++) {
          const sx = (std[i][0] + std[i][1]) / 2;
          const radius = Math.max(4, sx * w * 12);
          ctx.moveTo(pts[i].x + radius, pts[i].y);
          ctx.arc(pts[i].x, pts[i].y, radius, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.restore();
      }

      drawPath(ctx, pts, COLORS.prediction, 1.5, 0.35);
      if (showCount > 0) drawPath(ctx, pts.slice(0, showCount), COLORS.prediction, 2.5);
      if (showCount > 0) drawDot(ctx, pts[showCount - 1], COLORS.prediction, 5);
    }
  }, [inputTrack, targetTrack, prediction, frameIdx]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}
