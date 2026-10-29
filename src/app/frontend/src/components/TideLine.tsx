import { memo } from "react";
import type { TidePoint } from "../lib/api";
import { clock } from "../lib/format";

const W = 320;
const H = 92;
const TOP = 26; // room above the curve for the high-tide label

/** The predicted tide around now (6 h back, 18 h ahead): a line, a "now" mark and the next high water. */
function TideLineInner({ curve, now }: { curve: TidePoint[]; now: TidePoint | null }) {
  if (curve.length < 2) return null;
  const t0 = new Date(curve[0].ts).getTime();
  const t1 = new Date(curve[curve.length - 1].ts).getTime();
  const lo = Math.min(...curve.map((p) => p.predicted_m));
  const hi = Math.max(...curve.map((p) => p.predicted_m));
  const x = (iso: string) => ((new Date(iso).getTime() - t0) / (t1 - t0)) * W;
  const y = (m: number) => H - 6 - ((Math.min(m, hi) - lo) / (hi - lo || 1)) * (H - 6 - TOP);
  const line = curve.map((p, i) => `${i ? "L" : "M"}${x(p.ts).toFixed(1)},${y(p.predicted_m).toFixed(1)}`).join("");
  const nowX = now ? x(now.ts) : null;
  const highX = now?.next_high_ts ? x(now.next_high_ts) : null;
  const highLabel = now?.next_high_ts ? `High tide ${clock(now.next_high_ts)}` : "";
  return (
    <svg className="tide" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={now?.next_high_ts ? `Tide chart. Next high tide at ${clock(now.next_high_ts)}.` : "Tide chart"}>
      <path d={`${line}L${W},${H}L0,${H}Z`} className="tide-water" />
      <path d={line} className="tide-line" />
      {nowX != null ? (
        <>
          <line x1={nowX} x2={nowX} y1={TOP} y2={H} className="tide-now" />
          <text x={nowX + 4} y={H - 8} className="tide-label">
            Now
          </text>
        </>
      ) : null}
      {highX != null && highX >= 0 && highX <= W && now?.next_high_m != null ? (
        <>
          <circle cx={highX} cy={y(now.next_high_m)} r="4" className="tide-high" />
          <text
            x={Math.min(Math.max(highX, 44), W - 44)}
            y={y(now.next_high_m) - 9}
            className="tide-label"
            textAnchor="middle"
          >
            {highLabel}
          </text>
        </>
      ) : null}
    </svg>
  );
}

export const TideLine = memo(TideLineInner);
