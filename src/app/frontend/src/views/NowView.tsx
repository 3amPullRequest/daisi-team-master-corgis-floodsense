import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AreaMap, MapLegend } from "../components/AreaMap";
import { AreaPanel } from "../components/AreaPanel";
import { AskSection } from "../components/AskPanel";
import { LocationSearch } from "../components/LocationSearch";
import { Subscribe } from "../components/Subscribe";
import { useAreaShapes, useNow, type AreaNow, type Now, type Tier } from "../lib/api";
import { ASK_QUESTIONS, PUB_CHANNEL, TIER_LABEL } from "../lib/content";
import { chanceInWords, duration, minutesSince } from "../lib/format";
import { usePlaces } from "../lib/places";

const STALE_AFTER_MIN = 20; // a tick every 5 minutes; skipped runs and NEA's own lag stay well under this
const RANK: Record<Tier, number> = { Low: 0, Medium: 1, High: 2 };

/** The areas with a Moderate or Elevated warning in force, most at risk first. */
function atRisk(now: Now): AreaNow[] {
  return now.areas
    .filter((a) => a.tier !== "Low")
    .sort((a, b) => RANK[b.tier] - RANK[a.tier] || (b.p_alert ?? 0) - (a.p_alert ?? 0));
}

function headline(now: Now, warned: AreaNow[]): { title: string; detail: string } {
  const named = (t: Tier) => warned.filter((a) => a.tier === t).map((a) => a.area);
  const list = (names: string[]) =>
    names.length <= 3 ? names.join(", ").replace(/, ([^,]*)$/, " and $1") : `${names.length} areas`;
  const rest = now.areas.length - warned.length;
  const detail = rest === 1 ? "The other area is at Low." : `The other ${rest} areas are at Low.`;
  const high = named("High");
  const medium = named("Medium");
  if (high.length) return { title: `A flash-flood alert is more likely in ${list(high)} in the next hour.`, detail };
  if (medium.length) return { title: `A flash-flood alert is possible in ${list(medium)} in the next hour.`, detail };
  return { title: "No flash floods expected in the next hour.", detail: `All ${now.areas.length} areas are at Low.` };
}

type Selection = { area: string; place: string | null };

export function NowView() {
  const shapes = useAreaShapes();
  const now = useNow();
  const { places, toggle } = usePlaces();
  const [selected, setSelected] = useState<Selection | null>(null);
  const [picks, setPicks] = useState(0); // counts the visitor's picks, to bring the card into view after each
  const panel = useRef<HTMLDivElement>(null);

  const byArea = useMemo(() => new Map((now.data?.areas ?? []).map((a) => [a.area, a])), [now.data]);
  const tiers = useMemo(() => Object.fromEntries([...byArea].map(([k, v]) => [k, v.tier])), [byArea]);
  const warned = useMemo(() => (now.data ? atRisk(now.data) : []), [now.data]);
  const names = useMemo(() => [...byArea.keys()].sort(), [byArea]);

  const select = useCallback((area: string, place: string | null) => {
    setSelected({ area, place });
    setPicks((n) => n + 1);
  }, []);

  // On a phone the card sits below the search and above the map: after a pick, scroll it into view if it isn't.
  useEffect(() => {
    const el = panel.current;
    if (!picks || !el) return;
    const top = el.getBoundingClientRect().top;
    if (top < 0 || top > window.innerHeight - 160) {
      const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      el.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
    }
  }, [picks]);

  // what the card shows: the visitor's pick, else their first saved place, else the area most at risk if any is warned
  const saved = places.find((p) => byArea.has(p.area));
  const shown: { area: AreaNow; place: string | null; note: string | null } | null = selected
    ? byArea.has(selected.area)
      ? { area: byArea.get(selected.area)!, place: selected.place, note: null }
      : null
    : saved
      ? { area: byArea.get(saved.area)!, place: saved.label === saved.area ? null : saved.label, note: "One of your saved places." }
      : warned.length
        ? { area: warned[0], place: null, note: "The area most at risk right now." }
        : null;

  const shownArea = shown?.area.area ?? null;
  const labels = useMemo(
    () => [...(shownArea ? [shownArea] : []), ...warned.map((a) => a.area)],
    [shownArea, warned],
  );
  const detail = useCallback(
    (area: string) => {
      const a = byArea.get(area);
      if (!a) return null;
      return a.p_alert == null ? "Not rated: no rain gauge reported nearby" : `${chanceInWords(a.p_alert)} chance of a PUB alert`;
    },
    [byArea],
  );
  const onMapSelect = useCallback((area: string) => select(area, null), [select]);

  if (now.error && !now.data) {
    return (
      <div className="state state-error" role="alert">
        <h1>FloodSense can't reach its data right now.</h1>
        <p>
          Try again in a minute. For official alerts,{" "}
          <a href={PUB_CHANNEL} target="_blank" rel="noreferrer">
            follow PUB Flood Alerts on Telegram
          </a>
          .
        </p>
        <button onClick={() => now.mutate()}>Try again</button>
        <p className="fine-print">{now.error.message}</p>
      </div>
    );
  }
  if (!now.data || !shapes.data) return <NowSkeleton />;
  if (!now.data.tick) {
    return (
      <div className="state" role="status">
        <h1>No forecasts yet.</h1>
        <p>FloodSense hasn't rated any area yet. Its live job runs every 5 minutes once it is switched on.</p>
      </div>
    );
  }

  const { title, detail: rest } = headline(now.data, warned);
  const age = minutesSince(now.data.tick);
  const stale = now.data.stale || (age != null && age > STALE_AFTER_MIN);
  const savedHere = shown
    ? places.some((p) => p.area === shown.area.area && p.label === (shown.place ?? shown.area.area))
    : false;

  return (
    <div className="now">
      <div className="now-intro">
        {stale ? (
          <p className="stale" role="status">
            This forecast is {duration(age ?? 0)} old. The live feed seems to be paused. It may not show the rain now.
          </p>
        ) : null}
        <h1 className="headline">{title}</h1>
        {warned.length ? <Watch warned={warned} onPick={onMapSelect} /> : null}
        <p className="headline-detail">{rest}</p>
      </div>

      <div className="now-side">
        {places.length ? (
          <section className="places" aria-labelledby="places-title">
            <h2 id="places-title">Your places</h2>
            <ul>
              {places.map((p) => {
                const tier = byArea.get(p.area)?.tier;
                const current = shown?.area.area === p.area && (shown.place ?? shown.area.area) === p.label;
                return (
                  <li key={`${p.area}|${p.label}`}>
                    <button
                      type="button"
                      aria-pressed={current}
                      onClick={() => select(p.area, p.label === p.area ? null : p.label)}
                    >
                      <i className="swatch" data-tier={tier ?? "none"} />
                      <span className="place-name">{p.label}</span>
                      <span className="place-tier">{tier ? TIER_LABEL[tier] : "No rating"}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        ) : null}
        <LocationSearch areas={names} onFound={select} />
        <div ref={panel} className="panel-slot">
          {shown ? (
            <AreaPanel
              key={`${shown.area.area}|${shown.place ?? ""}`}
              area={shown.area}
              place={shown.place}
              note={shown.note}
              rainAsOf={now.data.rain_as_of}
              tide={now.data.tide.now}
              saved={savedHere}
              onToggleSave={() => toggle({ area: shown.area.area, label: shown.place ?? shown.area.area })}
            />
          ) : null}
        </div>
        <Subscribe areas={names} places={places} shown={shownArea} />
        {!warned.length ? (
          <aside className="invite">
            <p>
              <strong>No storm right now?</strong> Replay a past storm to see how FloodSense warns ahead of PUB.
            </p>
            <a className="button-link" href="#replay">
              Replay a storm
            </a>
          </aside>
        ) : null}
        <p className="visually-hidden" aria-live="polite">
          {shown && picks ? `${shown.area.area}: ${TIER_LABEL[shown.area.tier]}` : ""}
        </p>
      </div>

      <figure className="now-map">
        <AreaMap
          shapes={shapes.data}
          tiers={tiers}
          selected={shownArea}
          onSelect={onMapSelect}
          labels={labels}
          detail={detail}
          label="Map of Singapore's 47 forecast areas, coloured by flood risk in the next hour. Select an area for details."
        />
        <figcaption>
          <MapLegend />
          <span className="caption-note">
            Each warning lasts an hour from FloodSense's rating. Select an area to see what to do there.
          </span>
        </figcaption>
      </figure>

      <div className="now-ask">
        <AskSection
          questions={ASK_QUESTIONS.now}
          intro="Ask in plain English about the warnings now, or PUB's flood alerts since 2021."
        />
      </div>
    </div>
  );
}

/** The warned areas by name, grouped by tier: the map's colours as words, and a quick way to open each. */
function Watch({ warned, onPick }: { warned: AreaNow[]; onPick: (area: string) => void }) {
  const groups = (["High", "Medium"] as const)
    .map((tier) => [tier, warned.filter((a) => a.tier === tier)] as const)
    .filter(([, list]) => list.length);
  return (
    <dl className="watch">
      {groups.map(([tier, list]) => (
        <div key={tier} data-tier={tier}>
          <dt>
            <i className="swatch" data-tier={tier} />
            {TIER_LABEL[tier]}
          </dt>
          <dd>
            {list.map((a) => (
              <button key={a.area} type="button" onClick={() => onPick(a.area)}>
                {a.area}
              </button>
            ))}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function NowSkeleton() {
  return (
    <div className="now" aria-busy="true" aria-label="Loading the latest forecast">
      <div className="now-intro">
        <div className="skeleton skeleton-title" />
        <div className="skeleton skeleton-line" />
      </div>
      <div className="now-side">
        <div className="skeleton skeleton-block" />
      </div>
      <div className="now-map">
        <div className="skeleton skeleton-map" />
      </div>
    </div>
  );
}
