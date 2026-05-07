import { useEffect, useRef, useState } from "react";

const FPS = 10;

export default function SequencePlayer({ totalFrames, onFrame, autoPlay = false }) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(autoPlay);
  const rafRef = useRef(null);
  const lastRef = useRef(null);

  useEffect(() => {
    onFrame(frame);
  }, [frame]);

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(rafRef.current);
      return;
    }

    function tick(ts) {
      if (!lastRef.current) lastRef.current = ts;
      const elapsed = ts - lastRef.current;
      if (elapsed >= 1000 / FPS) {
        lastRef.current = ts;
        setFrame((f) => {
          const next = f + 1;
          if (next >= totalFrames) {
            setPlaying(false);
            return f;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, totalFrames]);

  function handleScrub(e) {
    setPlaying(false);
    setFrame(Number(e.target.value));
  }

  function reset() {
    setPlaying(false);
    setFrame(0);
    lastRef.current = null;
  }

  return { frame, playing, setPlaying, reset, handleScrub };
}
