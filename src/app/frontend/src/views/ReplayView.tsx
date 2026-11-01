import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { AreaMap, MapLegend, type Marker } from "../components/AreaMap";
import { AskSection } from "../components/AskPanel";
import { StormTimeline, type LeadRow } from "../components/StormTimeline";
import { getJson, useAreaShapes, type Alert, type Replay, type Replays, type Tier } from "../lib/api";
import { ASK_QUESTIONS } from "../lib/content";
import { clock, duration, longDay } from "../lib/format";
import { inForce, warnedSince } from "../lib/warnings";

const CODE: Record<string, Tier> = { L: "Low", M: "Medium", H: "High" };
const HOUR_MS = 3_600_000;
const STEP_MS = 240; // a 5-minute tick per frame: an hour of storm in about 3 seconds, slow enough to talk over
const DAY = /^\d{4}-\d{2}-\d{2}$/;

/** The day in a link such as #replay/2026-04-11, so a storm can be opened straight from a bookmark. */
function dayFromHash(): string | null {
  const day = window.location.hash.slice(1).split("/")[1];
  return day && DAY.test(day) ? day : null;
}

/** Since when a FloodSense warning had been in force at `at` (ms), and for how many minutes: each rating warns for the
 * hour after it, so the warning began at the first of its ratings (lib/warnings.ts). Nulls if none was. */
function leadAt(times: number[], codes: string, at: number) {
  const since = warnedSince(times, codes, at);
  return {
    warnedFrom: since == null ? null : new Date(since).toISOString(),
    leadMin: since == null ? null : (at - since) / 60_000,
  };
}

/** For each area PUB alerted, its first flash flood that day if it had one, and the FloodSense warning in force when
 * it struck. */
function compare(replay: Replay): LeadRow[] {
  const times = replay.ticks.map((t) => new Date(t).getTime());
  const areas = [...new Set(replay.alerts.map((a) => a.area))];
  return areas.map((area) => {
    const flood = replay.alerts.find((a) => a.area === area && a.kind === "flash_flood");
    const lead = flood ? leadAt(times, replay.tiers[area] ?? "", new Date(flood.ts).getTime()) : null;
    return { area, flood: flood?.ts ?? null, warnedFrom: lead?.warnedFrom ?? null, leadMin: lead?.leadMin ?? null };
  });
}

/** Every flash flood PUB reported that day, each on its own, with the FloodSense warning in force when it struck. */
function floodLeads(replay: Replay): (Alert & ReturnType<typeof leadAt>)[] {
  const times = replay.ticks.map((t) => new Date(t).getTime());
  return replay.alerts
    .filter((a) => a.kind === "flash_flood")
    .map((a) => ({ ...a, ...leadAt(times, replay.tiers[a.area] ?? "", new Date(a.ts).getTime()) }));
}

/** The ticks the timeline shows: from just before the earliest warning in force at an area's first alert (or an hour
 * before PUB's first alert) to an hour after PUB's last alert. */
function storyWindow(replay: Replay): [number, number] {
  const last = replay.ticks.length - 1;
  if (!replay.alerts.length) return [0, last];
  const times = replay.ticks.map((t) => new Date(t).getTime());
  const t0 = times[0];
  const alertTimes = replay.alerts.map((a) => new Date(a.ts).getTime());
  const firstAlert = new Map<string, number>();
  for (const a of replay.alerts) if (!firstAlert.has(a.area)) firstAlert.set(a.area, new Date(a.ts).getTime());
  const warnings = [...firstAlert].flatMap(([area, at]) => {
    const since = warnedSince(times, replay.tiers[area] ?? "", at);
    return since == null ? [] : [since];
  });
  const start = Math.min(Math.min(...alertTimes) - HOUR_MS, ...warnings) - 15 * 60_000;
  const end = Math.max(...alertTimes) + HOUR_MS;
  const tick = (t: number) => Math.min(Math.max(Math.round((t - t0) / 300_000), 0), last);
  return [tick(start), tick(end)];
}

export default function ReplayView() {
  const list = useSWR<Replays>("/api/replays", getJson, { revalidateOnFocus: false });
  const [day, setDay] = useState<string | null>(dayFromHash);
  const chosen = day ?? list.data?.days[0]?.day ?? null;

  useEffect(() => {
    const onHash = () => {
      const linked = dayFromHash();
      if (linked) setDay(linked);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  if (list.error) {
    return (
      <div className="state state-error" role="alert">
        <h1>The storm list didn't load.</h1>
        <p>{list.error.message}</p>
        <button onClick={() => list.mutate()}>Try again</button>
      </div>
    );
  }
  return (
    <div className="replay">
      <div className="replay-intro">
        <h1 className="headline">Replay a storm.</h1>
        <p className="headline-detail">
          Watch FloodSense rate every area through a real storm day, next to the alerts PUB sent.
        </p>
      </div>
      {list.data ? (
        list.data.days.length ? (
          <ul className="storms" aria-label="Storm days">
            {list.data.days.map((d) => (
              <li key={d.day}>
                <button
                  aria-pressed={d.day === chosen}
                  onClick={() => {
                    setDay(d.day);
                    window.history.replaceState(null, "", `#replay/${d.day}`);
                  }}
                >
                  <strong>{longDay(d.day)}</strong>
                  <span>
                    {d.alerts} PUB alert{d.alerts === 1 ? "" : "s"}
                    {d.floods ? `, ${d.floods} flash flood${d.floods === 1 ? "" : "s"}` : ""}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="state">No storms since the training data ended yet. They appear here as PUB sends alerts.</p>
        )
      ) : (
        <div className="skeleton skeleton-strip" />
      )}
      {chosen ? <ReplayDay key={chosen} day={chosen} /> : null}
      <AskSection
        questions={ASK_QUESTIONS.replay}
        intro="Ask about PUB's alerts in any storm since 2021, or how long before each flash flood FloodSense warned."
      />
    </div>
  );
}

function ReplayDay({ day }: { day: string }) {
  const shapes = useAreaShapes();
  const replay = useSWR<Replay>(`/api/replays/${day}`, getJson, { revalidateOnFocus: false, revalidateIfStale: false });
  const data = replay.data;
  // the warning in force at every tick: what the map and the timeline show
  const warnings = useMemo(
    () => (data ? Object.fromEntries(Object.entries(data.tiers).map(([area, codes]) => [area, inForce(codes)])) : {}),
    [data],
  );
  const rows = useMemo(() => (data ? compare(data) : []), [data]);
  const leads = useMemo(() => (data ? floodLeads(data) : []), [data]);
  const [from, to] = useMemo(() => (data ? storyWindow(data) : [0, 0]), [data]);
  // start an hour before PUB's first alert, so the build-up is in view
  const startIndex = useMemo(() => {
    if (!data?.alerts.length) return 0;
    const first = new Date(data.alerts[0].ts).getTime() - HOUR_MS;
    return Math.max(0, data.ticks.findIndex((t) => new Date(t).getTime() >= first));
  }, [data]);
  const [index, setIndex] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const at = index ?? startIndex;
  const stopAt = useRef(0);

  useEffect(() => {
    if (!playing || !data) return;
    const timer = window.setInterval(() => {
      setIndex((i) => {
        const next = (i ?? startIndex) + 1;
        if (next >= stopAt.current) {
          setPlaying(false);
          return stopAt.current;
        }
        return next;
      });
    }, STEP_MS);
    return () => window.clearInterval(timer);
  }, [playing, data, startIndex]);

  const seek = useCallback((tick: number) => {
    setPlaying(false);
    setIndex(tick);
  }, []);

  const tiers = useMemo(
    () => Object.fromEntries(Object.entries(warnings).map(([area, codes]) => [area, CODE[codes[at]] ?? "Low"])),
    [warnings, at],
  );
  const markers = useMemo<Marker[]>(() => {
    if (!data) return [];
    const now = new Date(data.ticks[at]).getTime();
    return data.alerts
      .filter((a) => new Date(a.ts).getTime() <= now)
      .map((a) => ({ area: a.area, kind: a.kind, recent: now - new Date(a.ts).getTime() < HOUR_MS }));
  }, [data, at]);
  const labels = useMemo(() => rows.map((r) => r.area), [rows]);
  const alsoWarned = useMemo(() => {
    if (!data) return [];
    const alerted = new Set(rows.map((r) => r.area));
    return Object.entries(data.tiers)
      .filter(([area, codes]) => !alerted.has(area) && /[MH]/.test(codes))
      .map(([area]) => area)
      .sort();
  }, [data, rows]);

  if (replay.error) {
    return (
      <div className="state state-error" role="alert">
        <p>{replay.error.message}</p>
        <button onClick={() => replay.mutate()}>Try again</button>
      </div>
    );
  }
  if (!data || !shapes.data) {
    return (
      <div className="replay-stage" aria-busy="true">
        <p className="loading-note" role="status">
          FloodSense is rating all 47 areas for {longDay(day)}, every 5 minutes of the day. The first replay of a day can
          take a minute.
        </p>
        <div className="skeleton skeleton-map" />
      </div>
    );
  }

  const tickTime = data.ticks[at];
  const end = data.ticks.length - 1;
  return (
    <div className="replay-stage">
      <div className="replay-main">
        <figure className="replay-map">
          <p className="replay-clock" aria-live="polite">
            {clock(tickTime)}
          </p>
          <AreaMap
            shapes={shapes.data}
            tiers={tiers}
            selected={null}
            onSelect={() => undefined}
            markers={markers}
            labels={labels}
            dots={labels}
            label={`FloodSense ratings at ${clock(tickTime)} on ${longDay(day)}, with PUB's alerts so far.`}
          />
          <figcaption>
            <MapLegend />
            <span className="caption-note">
              <i className="dot" data-kind="warning" /> PUB drain warning
              <i className="dot" data-kind="flash_flood" /> PUB flash flood
            </span>
          </figcaption>
        </figure>

        <div className="scrubber">
          <button
            onClick={() => {
              if (playing) return setPlaying(false);
              // play through the storm and stop an hour after PUB's last alert; past that, start the storm again
              if (at >= to) setIndex(startIndex);
              stopAt.current = to;
              setPlaying(true);
            }}
          >
            {playing ? "Pause" : at >= to ? "Replay" : "Play"}
          </button>
          <input
            type="range"
            min={0}
            max={end}
            value={at}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label="Time of day"
            aria-valuetext={clock(tickTime)}
          />
        </div>
      </div>

      <div className="replay-story">
        {rows.length ? (
          <StormTimeline replay={data} warnings={warnings} rows={rows} at={at} from={from} to={to} onSeek={seek} />
        ) : (
          <p>PUB sent no alerts on this day.</p>
        )}
        {alsoWarned.length ? (
          <p className="also-warned">
            FloodSense also warned{" "}
            {alsoWarned.length === 1
              ? `${alsoWarned[0]}, where PUB sent no alert that day.`
              : `${alsoWarned.length} areas where PUB sent no alert that day: ${alsoWarned.join(", ").replace(/, ([^,]*)$/, " and $1")}.`}
          </p>
        ) : null}
        {leads.length ? (
          <details className="leads-box">
            <summary>See these times as a table</summary>
            <table className="leads">
              <caption>Did FloodSense warn before the flash floods on {longDay(day)}?</caption>
              <thead>
                <tr>
                  <th scope="col">Area</th>
                  <th scope="col">Flash flood</th>
                  <th scope="col">FloodSense warning from</th>
                  <th scope="col">Ahead of the flood</th>
                </tr>
              </thead>
              <tbody>
                {leads.map((r) => (
                  <tr key={`${r.area}|${r.ts}|${r.location}`}>
                    <th scope="row" title={r.location}>
                      {r.area}
                    </th>
                    <td>{clock(r.ts)}</td>
                    <td>{r.warnedFrom ? clock(r.warnedFrom) : "No warning in force"}</td>
                    <td>{r.leadMin != null ? duration(r.leadMin) : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ) : null}
      </div>
    </div>
  );
}
