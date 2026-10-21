import { useState } from "react";
import type { AreaNow, TidePoint } from "../lib/api";
import { ADVICE, AUDIENCES, PUB_CHANNEL, TIER_MEANING, type Audience } from "../lib/content";
import { chanceInWords, clock, duration } from "../lib/format";

type Props = {
  area: AreaNow;
  place: string | null;
  picked: boolean; // false: shown because it has the highest chance now
  rainAsOf: string | null;
  tide: TidePoint | null;
};

export function AreaPanel({ area, place, picked, rainAsOf, tide }: Props) {
  const [audience, setAudience] = useState<Audience>("driving");
  const rain = area.rain_60min_mm;
  return (
    <section className="panel" data-tier={area.tier} aria-labelledby="panel-title">
      <header className="panel-head">
        <h2 id="panel-title">{area.area}</h2>
        {place ? (
          <p className="panel-place">Includes {place.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())}</p>
        ) : !picked ? (
          <p className="panel-place">Shown because it has the highest chance right now. Pick any area on the map.</p>
        ) : null}
      </header>

      <div className="gauge-row">
        <Gauge tier={area.tier} />
        <div>
          <p className="tier-word">{area.tier}</p>
          <p className="tier-meaning">{TIER_MEANING[area.tier]}</p>
        </div>
      </div>

      <dl className="facts">
        <div>
          <dt>Chance of a PUB flood alert here in the next hour</dt>
          <dd>{chanceInWords(area.p_alert)}</dd>
        </div>
        {area.reason ? (
          <div>
            <dt>Why</dt>
            <dd>{area.reason}</dd>
          </div>
        ) : null}
        <div>
          <dt>Rain in the hour to {clock(rainAsOf)}</dt>
          <dd>{rain == null ? "No gauge within 5 km reported" : rain < 0.5 ? "None at the gauges within 5 km" : `${rain.toFixed(0)} mm at the wettest gauge within 5 km`}</dd>
        </div>
        {tide?.next_high_ts ? (
          <div>
            <dt>Tide</dt>
            <dd>
              {tide.predicted_m.toFixed(1)} m now; high tide at {clock(tide.next_high_ts)}
              {tide.minutes_to_high_tide != null ? ` (in ${duration(tide.minutes_to_high_tide)})` : ""}
            </dd>
          </div>
        ) : null}
      </dl>

      <div className="advice">
        <div className="advice-tabs" role="tablist" aria-label="What to do">
          {AUDIENCES.map((a) => (
            <button
              key={a.id}
              role="tab"
              aria-selected={audience === a.id}
              aria-controls="advice-text"
              onClick={() => setAudience(a.id)}
            >
              {a.label}
            </button>
          ))}
        </div>
        <p id="advice-text" role="tabpanel" className="advice-text">
          {ADVICE[area.tier][audience]}
        </p>
      </div>

      <p className="pub-note">
        FloodSense can warn before PUB, but it is less certain. PUB's own alerts come first:{" "}
        <a href={PUB_CHANNEL} target="_blank" rel="noreferrer">
          follow PUB Flood Alerts on Telegram
        </a>
        .
      </p>
    </section>
  );
}

const LEVEL = { Low: 0.3, Medium: 0.72, High: 0.95 } as const;

/** A drain cross-section filled to the tier: PUB warns when a drain reaches 90%, so the scale reads like the drain
 * signs people already know. */
function Gauge({ tier }: { tier: AreaNow["tier"] }) {
  const fill = LEVEL[tier];
  const top = 8 + (1 - fill) * 72;
  return (
    <svg className="gauge" viewBox="-28 0 84 88" aria-hidden="true">
      <path d="M6 4v72a8 8 0 0 0 8 8h28a8 8 0 0 0 8-8V4" className="gauge-wall" />
      <rect x="8" y={top} width="40" height={80 - top} rx="5" className="gauge-water" />
      <line x1="0" x2="56" y1={8 + 0.1 * 72} y2={8 + 0.1 * 72} className="gauge-mark" />
      <text x="-2" y={8 + 0.1 * 72 + 3} className="gauge-label" textAnchor="end">
        90%
      </text>
    </svg>
  );
}
