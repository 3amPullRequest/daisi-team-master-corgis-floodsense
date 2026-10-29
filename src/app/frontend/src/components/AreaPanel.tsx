import type { AreaNow, TidePoint } from "../lib/api";
import { ADVICE, AUDIENCES, PUB_CHANNEL, TIER_LABEL, TIER_MEANING } from "../lib/content";
import { chanceInWords, clock, duration } from "../lib/format";
import { useAudience } from "../lib/places";

type Props = {
  area: AreaNow;
  place: string | null; // the searched place, already in readable case
  note: string | null; // why this area is shown, when the visitor didn't pick it
  rainAsOf: string | null;
  tide: TidePoint | null;
  saved: boolean;
  onToggleSave: () => void;
};

/** One area, in the order a resident needs it: the rating, what to do, then the evidence. */
export function AreaPanel({ area, place, note, rainAsOf, tide, saved, onToggleSave }: Props) {
  const [audience, setAudience] = useAudience();
  const rain = area.rain_60min_mm;
  const warning = area.warning;
  // the newest rating, when it is below the warning still in force from an earlier one
  const calmer = warning && area.rated && area.rated !== area.tier ? area.rated : null;
  const reason = warning?.reason ?? area.reason;
  return (
    <section className="panel" data-tier={area.tier} aria-labelledby="panel-title">
      <header className="panel-head">
        <div>
          <h2 id="panel-title">{area.area}</h2>
          {place ? <p className="panel-place">Includes {place}</p> : note ? <p className="panel-place">{note}</p> : null}
        </div>
        <button type="button" className="save" aria-pressed={saved} onClick={onToggleSave}>
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path d="M10 2.5l2.3 4.8 5.2.7-3.8 3.6.9 5.2L10 14.3l-4.6 2.5.9-5.2L2.5 8l5.2-.7z" />
          </svg>
          {saved ? "Saved" : "Save place"}
        </button>
      </header>

      <div className="status">
        <Gauge tier={area.tier} />
        <div>
          <p className="tier-word">{TIER_LABEL[area.tier]}</p>
          <p className="tier-meaning">{TIER_MEANING[area.tier]}</p>
          {warning ? (
            <p className="tier-until">
              Until {clock(warning.until)}. FloodSense rated it {TIER_LABEL[warning.tier]} at {clock(warning.issued)},
              and each rating holds for an hour.
              {calmer ? ` Its newest rating, at ${clock(area.ts)}, is ${TIER_LABEL[calmer]}.` : ""}
            </p>
          ) : null}
          {area.nearby ? (
            <p className="nearby-chip" data-tier={area.nearby.tier}>
              {TIER_LABEL[area.nearby.tier]} within 5 km: {area.nearby.area}, {area.nearby.km.toFixed(1)} km away
            </p>
          ) : null}
        </div>
      </div>
      <div className="advice">
        <h3 id="advice-title">What to do</h3>
        <div className="advice-tabs" role="tablist" aria-labelledby="advice-title">
          {AUDIENCES.map((a) => (
            <button
              key={a.id}
              type="button"
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

      <dl className="facts">
        {reason ? (
          <div>
            <dt>Why</dt>
            <dd>{reason}</dd>
          </div>
        ) : null}
        {area.nearby ? (
          <div>
            <dt>Nearby</dt>
            <dd>
              Storms move. PUB's alerts often reach neighbouring drains within the hour. Keep an eye on {area.nearby.area}{" "}
              too.
            </dd>
          </div>
        ) : null}
        <div>
          <dt>Chance of a PUB flood alert here in the next hour, at {clock(area.ts)}</dt>
          <dd>{chanceInWords(area.p_alert)}</dd>
        </div>
        <div>
          <dt>Rain in the hour to {clock(rainAsOf)}</dt>
          <dd>
            {rain == null
              ? "No rain gauge within 5 km reported"
              : rain < 0.5
                ? "None at the rain gauges within 5 km"
                : `${rain.toFixed(0)} mm at the wettest rain gauge within 5 km`}
          </dd>
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

      <p className="pub-note">
        FloodSense can warn before PUB, but it is less certain. PUB's alerts come first.{" "}
        <a href={PUB_CHANNEL} target="_blank" rel="noreferrer">
          Follow PUB Flood Alerts on Telegram
        </a>
        .
      </p>
    </section>
  );
}

const LEVEL = { Low: 0.3, Medium: 0.72, High: 0.95 } as const;

/** A drain filled to the tier, like the water-level signs beside Singapore's canals. */
function Gauge({ tier }: { tier: AreaNow["tier"] }) {
  const top = 8 + (1 - LEVEL[tier]) * 72;
  return (
    <svg className="gauge" viewBox="0 0 56 88" aria-hidden="true">
      <path d="M6 4v72a8 8 0 0 0 8 8h28a8 8 0 0 0 8-8V4" className="gauge-wall" />
      <rect x="8" y={top} width="40" height={80 - top} rx="5" className="gauge-water" />
    </svg>
  );
}
