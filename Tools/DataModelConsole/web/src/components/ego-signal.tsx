// Ego signal (ego.npy) small-multiples visualization.
//
// History channels (t = -6.4s..0): speed, accel, yaw_rate, curvature.
// Future channels (t = 0..+6.4s): accel, curvature.
// Each channel is its own lane with a shared t=0 marker.

import { EGO_DT, decodeEgo } from "@/lib/ego";

const LANE_W = 800;
const LANE_H = 72;
const PAD = 6;

interface Lane {
  label: string;
  unit: string;
  values: number[];
  side: "history" | "future";
  color: string;
}

function LaneChart({ lane }: { lane: Lane }) {
  const { values, side } = lane;
  const n = values.length;
  const span = n * EGO_DT; // seconds covered
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const range = max - min || 1;

  const xOf = (i: number) => (i / Math.max(n - 1, 1)) * LANE_W;
  const yOf = (v: number) =>
    LANE_H - PAD - ((v - min) / range) * (LANE_H - 2 * PAD);

  const points = values
    .map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(" ");

  // t=0 sits at the right edge of history lanes and the left edge of future.
  const zeroX = side === "history" ? LANE_W : 0;
  const zeroY = yOf(0);

  const tStart = side === "history" ? -span : 0;
  const tEnd = side === "history" ? 0 : span;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between font-mono text-[10px] text-slate-400">
        <span>
          {lane.label} <span className="text-slate-600">({lane.unit})</span>
        </span>
        <span className="text-slate-600">
          {tStart.toFixed(1)}s .. {tEnd >= 0 ? "+" : ""}
          {tEnd.toFixed(1)}s | min {Math.min(...values).toFixed(3)} | max{" "}
          {Math.max(...values).toFixed(3)}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${LANE_W} ${LANE_H}`}
        className="w-full rounded-md border border-slate-800 bg-slate-900/50"
        preserveAspectRatio="none"
        role="img"
        aria-label={`${lane.label} lane`}
      >
        {/* zero-value baseline */}
        <line
          x1={0}
          y1={zeroY}
          x2={LANE_W}
          y2={zeroY}
          stroke="#334155"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
        {/* t=0 marker */}
        <line
          x1={zeroX}
          y1={0}
          x2={zeroX}
          y2={LANE_H}
          stroke="#f59e0b"
          strokeWidth="1.5"
        />
        <polyline
          points={points}
          fill="none"
          stroke={lane.color}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
}

export function EgoSignal({
  history,
  future,
}: {
  history: number[];
  future: number[];
}) {
  if (history.length === 0 && future.length === 0) {
    return <p className="text-sm text-slate-500">No ego signal available.</p>;
  }

  const ego = decodeEgo(history, future);

  const historyLanes: Lane[] = [
    {
      label: "speed",
      unit: "m/s",
      values: ego.history.speed,
      side: "history",
      color: "#3b82f6",
    },
    {
      label: "accel",
      unit: "m/s^2",
      values: ego.history.accel,
      side: "history",
      color: "#22c55e",
    },
    {
      label: "yaw_rate",
      unit: "rad/s",
      values: ego.history.yawRate,
      side: "history",
      color: "#a855f7",
    },
    {
      label: "curvature",
      unit: "1/m",
      values: ego.history.curvature,
      side: "history",
      color: "#ef4444",
    },
  ];

  const futureLanes: Lane[] = [
    {
      label: "accel",
      unit: "m/s^2",
      values: ego.future.accel,
      side: "future",
      color: "#22c55e",
    },
    {
      label: "curvature",
      unit: "1/m",
      values: ego.future.curvature,
      side: "future",
      color: "#ef4444",
    },
  ];

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          History (t = -{(ego.history.speed.length * EGO_DT).toFixed(1)}s .. 0,
          10Hz)
        </p>
        {historyLanes.map(
          (lane) =>
            lane.values.length > 0 && <LaneChart key={lane.label} lane={lane} />,
        )}
      </div>
      <div className="space-y-2">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">
          Future (t = 0 .. +{(ego.future.accel.length * EGO_DT).toFixed(1)}s,
          10Hz)
        </p>
        {futureLanes.map(
          (lane) =>
            lane.values.length > 0 && (
              <LaneChart key={`f-${lane.label}`} lane={lane} />
            ),
        )}
      </div>
      <details className="text-xs">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          Raw values
        </summary>
        <pre className="mt-2 max-h-48 overflow-auto rounded-md border border-slate-800 bg-slate-900/50 p-3 font-mono text-[10px] leading-relaxed text-slate-400">
          history[{history.length}]:{" "}
          {history.map((v) => v.toFixed(4)).join(", ")}
          {"\n\n"}
          future[{future.length}]: {future.map((v) => v.toFixed(4)).join(", ")}
        </pre>
      </details>
    </div>
  );
}
