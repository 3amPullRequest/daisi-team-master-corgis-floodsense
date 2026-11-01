import type { Alert, Replay, Tier } from "../lib/api";
import { TIER_LABEL } from "../lib/content";
import { clock, duration } from "../lib/format";

const CODE: Record<string, Tier> = { M: "Medium", H: "High" };
const TICK_MS = 5 * 60_000;

/** One area PUB alerted that day: its first flash flood, if it had one, and since when FloodSense's warning had been in
 * force when it struck, if one was. */
export type LeadRow = { area: string; flood: string | null; warnedFrom: string | null; leadMin: number | null };

type Props = {
  replay: Replay;
  warnings: Record<string, string>; // the warning in force per area, one letter per tick (lib/warnings.ts)
  rows: LeadRow[];
  at: number; // the replay's current tick
  from: number; // the window shown, as tick indexes
  to: number;
  onSeek: (tick: number) => void;
};

/** Runs of Moderate or Elevated warnings in force in [from, to], one per change of tier. */
function runs(codes: string, from: number, to: number) {
  const out: { start: number; end: number; tier: Tier }[] = [];
  for (let k = from; k <= to; k++) {
    const tier = CODE[codes[k]];
    const last = out[out.length - 1];
    if (tier && last && last.tier === tier && last.end === k) last.end = k + 1;
    else if (tier) out.push({ start: k, end: k + 1, tier });
  }
  return out;
}

/** Each alerted area as a row in time: FloodSense's warnings in force as bars (each rating warns for the hour after
 * it), PUB's alerts as dots, and the replay's clock as a line. The gap between a bar's start and a flash flood's dot is
 * the warning FloodSense gave before it. */
export function StormTimeline({ replay, warnings, rows, at, from, to, onSeek }: Props) {
  const t0 = new Date(replay.ticks[0]).getTime();
  const span = Math.max(to + 1 - from, 1);
  const left = (tick: number) => `${((tick - from) / span) * 100}%`;
  const time = (tick: number) => new Date(t0 + tick * TICK_MS).toISOString();
  const hours: number[] = [];
  for (let k = Math.ceil(from / 12) * 12; k <= to; k += 12) hours.push(k); // ticks start at midnight: every 12th is on the hour
  const byArea = new Map<string, Alert[]>();
  for (const a of replay.alerts) byArea.set(a.area, [...(byArea.get(a.area) ?? []), a]);

  function seek(e: React.PointerEvent<HTMLDivElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const share = Math.min(Math.max((e.clientX - box.left) / box.width, 0), 1);
    onSeek(Math.min(from + Math.round(share * span), to));
  }

  return (
    <figure className="timeline">
      <figcaption>
        <strong>When FloodSense warned, and when PUB alerted</strong>
        <span>Each Moderate or Elevated rating is a warning for the hour after it.</span>
        <span className="timeline-key">
          <span>
            <i className="swatch" data-tier="Medium" /> {TIER_LABEL.Medium}
          </span>
          <span>
            <i className="swatch" data-tier="High" /> {TIER_LABEL.High}
          </span>
          <span>
            <i className="dot" data-kind="warning" /> PUB drain warning
          </span>
          <span>
            <i className="dot" data-kind="flash_flood" /> PUB flash flood
          </span>
        </span>
      </figcaption>
      <div className="tl">
        {rows.map((r) => (
          <div className="tl-row" key={r.area}>
            <p className="tl-label">
              <strong>{r.area}</strong>
              <span>
                {r.flood == null
                  ? "No flash flood"
                  : r.leadMin != null
                    ? `Warned ${duration(r.leadMin)} before the flood`
                    : "No warning before the flood"}
              </span>
            </p>
            <div
              className="tl-track"
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                seek(e);
              }}
              onPointerMove={(e) => {
                if (e.buttons & 1) seek(e);
              }}
            >
              {hours.map((h) => (
                <i key={h} className="tl-hour" style={{ left: left(h) }} />
              ))}
              {runs(warnings[r.area] ?? "", from, to).map((s) => (
                <i
                  key={s.start}
                  className="tl-run"
                  data-tier={s.tier}
                  style={{ left: left(s.start), width: `${((s.end - s.start) / span) * 100}%` }}
                  title={`${TIER_LABEL[s.tier]} warning from ${clock(time(s.start))} to ${clock(time(s.end))}`}
                />
              ))}
              {(byArea.get(r.area) ?? []).map((a) => {
                const tick = (new Date(a.ts).getTime() - t0) / TICK_MS;
                return tick >= from && tick <= to + 1 ? (
                  <i
                    key={a.ts}
                    className="tl-alert dot"
                    data-kind={a.kind}
                    style={{ left: left(tick) }}
                    title={`PUB ${a.kind === "flash_flood" ? "flash flood" : "drain warning"} at ${clock(a.ts)}: ${a.location}`}
                  />
                ) : null;
              })}
              {at >= from && at <= to ? <i className="tl-now" style={{ left: left(at) }} /> : null}
            </div>
          </div>
        ))}
        <div className="tl-row tl-axis" aria-hidden="true">
          <span />
          <div className="tl-track">
            {hours.map((h) => (
              <span key={h} style={{ left: left(h) }}>
                {clock(replay.ticks[h]).replace(":00", "")}
              </span>
            ))}
          </div>
        </div>
      </div>
    </figure>
  );
}
