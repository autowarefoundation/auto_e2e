"use client";

// use-playback: monotonic media clock for frame-accurate 10Hz playback.
//
// A requestAnimationFrame loop advances mediaTime += dt * speed * direction;
// the displayed frame is round(mediaTime * fps). The clock never blocks on
// slow frame fetches — late frames are simply dropped (the renderer draws
// whatever bitmap is ready for the current frame).

import { useCallback, useEffect, useRef, useState } from "react";

export const MIN_SPEED = 0.1;
export const MAX_SPEED = 16;

export interface PlaybackState {
  frame: number;
  playing: boolean;
  speed: number;
  direction: 1 | -1;
}

export interface PlaybackControls extends PlaybackState {
  setFrame: (frame: number) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  step: (delta: number) => void;
  setSpeed: (speed: number) => void;
  setDirection: (dir: 1 | -1) => void;
}

function clampSpeed(s: number): number {
  return Math.min(MAX_SPEED, Math.max(MIN_SPEED, s));
}

export function usePlayback(
  frameCount: number,
  fps: number,
  initialFrame = 0,
): PlaybackControls {
  const lastFrame = Math.max(0, frameCount - 1);
  const clampFrame = useCallback(
    (f: number) => Math.min(lastFrame, Math.max(0, Math.round(f))),
    [lastFrame],
  );

  const [frame, setFrameState] = useState(() =>
    Math.min(lastFrame, Math.max(0, initialFrame)),
  );
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [direction, setDirectionState] = useState<1 | -1>(1);

  // Mutable clock state read by the rAF loop (no re-render per tick).
  const mediaTimeRef = useRef(
    (Math.min(lastFrame, Math.max(0, initialFrame)) / (fps || 10)) as number,
  );
  const playingRef = useRef(false);
  const speedRef = useRef(1);
  const directionRef = useRef<1 | -1>(1);
  const rafRef = useRef(0);
  const lastTsRef = useRef<number | null>(null);

  playingRef.current = playing;
  speedRef.current = speed;
  directionRef.current = direction;

  useEffect(() => {
    if (!playing) {
      lastTsRef.current = null;
      return;
    }
    const effFps = fps || 10;

    const tick = (ts: number) => {
      const last = lastTsRef.current;
      lastTsRef.current = ts;
      if (last !== null) {
        const dt = Math.min((ts - last) / 1000, 0.5); // guard tab-suspend jumps
        mediaTimeRef.current +=
          dt * speedRef.current * directionRef.current;
        const maxT = lastFrame / effFps;
        if (mediaTimeRef.current >= maxT) {
          mediaTimeRef.current = maxT;
          setPlaying(false);
        } else if (mediaTimeRef.current <= 0) {
          mediaTimeRef.current = 0;
          setPlaying(false);
        }
        const f = Math.min(
          lastFrame,
          Math.max(0, Math.round(mediaTimeRef.current * effFps)),
        );
        setFrameState((prev) => (prev === f ? prev : f));
      }
      if (playingRef.current) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, fps, lastFrame]);

  const setFrame = useCallback(
    (f: number) => {
      const c = clampFrame(f);
      mediaTimeRef.current = c / (fps || 10);
      setFrameState(c);
    },
    [clampFrame, fps],
  );

  const play = useCallback(() => {
    // Restart from the top if the clip already ran off either end.
    const effFps = fps || 10;
    if (
      directionRef.current === 1 &&
      mediaTimeRef.current >= lastFrame / effFps &&
      lastFrame > 0
    ) {
      mediaTimeRef.current = 0;
      setFrameState(0);
    } else if (directionRef.current === -1 && mediaTimeRef.current <= 0) {
      mediaTimeRef.current = lastFrame / effFps;
      setFrameState(lastFrame);
    }
    setPlaying(true);
  }, [fps, lastFrame]);

  const pause = useCallback(() => setPlaying(false), []);

  const toggle = useCallback(() => {
    if (playingRef.current) setPlaying(false);
    else play();
  }, [play]);

  const step = useCallback(
    (delta: number) => {
      setPlaying(false);
      setFrameState((prev) => {
        const c = clampFrame(prev + delta);
        mediaTimeRef.current = c / (fps || 10);
        return c;
      });
    },
    [clampFrame, fps],
  );

  const setSpeed = useCallback((s: number) => {
    setSpeedState(clampSpeed(s));
  }, []);

  const setDirection = useCallback((dir: 1 | -1) => {
    setDirectionState(dir);
  }, []);

  return {
    frame,
    playing,
    speed,
    direction,
    setFrame,
    play,
    pause,
    toggle,
    step,
    setSpeed,
    setDirection,
  };
}
