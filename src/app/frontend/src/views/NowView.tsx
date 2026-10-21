import { useMemo, useState } from "react";
import { AreaMap } from "../components/AreaMap";
import { AreaPanel } from "../components/AreaPanel";
import { LocationSearch } from "../components/LocationSearch";
import { TideLine } from "../components/TideLine";
import { useAreaShapes, useNow, type AreaNow, type Now, type Tier } from "../lib/api";
import { clock, duration, minutesSince } from "../lib/format";

const STALE_AFTER_MIN = 20; // a tick every 5 minutes; skipped runs and NEA's own lag stay well under this

function headline(now: Now): { title: string; detail: string } {
  const by = (t: Tier) => now.areas.filter((a) => a.tier === t).map((a) => a.area);
  const high = by("High");
  const medium = by("Medium");
  const list = (names: string[]) =>
    names.length <= 3 ? names.join(", ").replace(/, ([^,]*)$/, " and $1") : `${names.length} areas`;
  if (high.length) return { title: `Flash floods likely in ${list(high)} in the next hour.`, detail: medium.length ? `Also possible in ${list(medium)}.` : "" };
  if (medium.length) return { title: `Flash floods possible in ${list(medium)} in the next hour.`, detail: "Everywhere else is unlikely to flood." };
  return { title: "No flash floods expected in the next hour.", detail: `All ${now.areas.length} areas are low risk.` };
}

export function NowView() {
  const shapes = useAreaShapes();
  const now = useNow();
  const [selected, setSelected] = useState<string | null>(null);
  const [place, setPlace] = useState<string | null>(null);

  const byArea = useMemo(() => new Map((now.data?.areas ?? []).map((a) => [a.area, a])), [now.data]);
  const tiers = useMemo(() => Object.fromEntries([...byArea].map(([k, v]) => [k, v.tier])), [byArea]);
  // with nothing chosen yet, show the area most at risk
  const shown: AreaNow | undefined = selected
    ? byArea.get(selected)
    : now.data?.areas.reduce<AreaNow | undefined>((best, a) => (!best || (a.p_alert ?? 0) > (best.p_alert ?? 0) ? a : best), undefined);

  if (now.error && !now.data) {
    return (
      <div className="state state-error" role="alert">
        <h1>FloodSense can't reach its data right now.</h1>
        <p>{now.error.message}</p>
        <button onClick={() => now.mutate()}>Try again</button>
      </div>
    );
  }
  if (!now.data || !shapes.data) return <NowSkeleton />;
  if (!now.data.tick) {
    return (
      <div className="state" role="status">
        <h1>No forecasts yet.</h1>
        <p>FloodSense hasn't scored a tick yet. Its live job runs every 5 minutes once it's switched on.</p>
      </div>
    );
  }

  const { title, detail } = headline(now.data);
  const age = minutesSince(now.data.tick);
  const stale = now.data.stale || (age != null && age > STALE_AFTER_MIN);

  return (
    <div className="now">
      <div className="now-intro">
        <h1 className="headline">{title}</h1>
        <p className="headline-detail">
          {detail} Forecast for {clock(now.data.tick)}, Singapore time.
        </p>
        {stale ? (
          <p className="stale" role="status">
            This forecast is {duration(age ?? 0)} old: the live feed seems to be paused, so it may not reflect the rain now.
          </p>
        ) : null}
      </div>

      <div className="now-side">
        <LocationSearch
          onFound={(area, where) => {
            setSelected(area);
            setPlace(where);
          }}
        />
        {shown ? (
          <AreaPanel
            key={shown.area}
            area={shown}
            place={selected ? place : null}
            picked={selected != null}
            rainAsOf={now.data.rain_as_of}
            tide={now.data.tide.now}
          />
        ) : null}
        <div className="tide-box">
          <p>Tide at Tanjong Pagar, 6 hours back to 18 hours ahead (dashed line: now)</p>
          <TideLine curve={now.data.tide.curve} now={now.data.tide.now} />
        </div>
      </div>

      <figure className="now-map">
        <AreaMap
          shapes={shapes.data}
          tiers={tiers}
          selected={shown?.area ?? null}
          onSelect={(area) => {
            setSelected(area);
            setPlace(null);
          }}
          label="Map of Singapore's 47 forecast areas, coloured by flood risk in the next hour. Select an area for details."
        />
        <figcaption>
          <Legend />
          <span className="caption-note">Select an area for details. Areas follow NEA's 47 weather-forecast areas.</span>
        </figcaption>
      </figure>
    </div>
  );
}

export function Legend() {
  return (
    <ul className="legend" aria-label="Legend">
      <li data-tier="Low">Low</li>
      <li data-tier="Medium">Medium</li>
      <li data-tier="High">High</li>
    </ul>
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
