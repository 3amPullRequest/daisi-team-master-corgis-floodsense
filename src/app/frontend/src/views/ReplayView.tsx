import { useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { AreaMap, type Marker } from "../components/AreaMap";
import { getJson, useAreaShapes, type Alert, type Replay, type Replays, type Tier } from "../lib/api";
import { clock, duration, longDay } from "../lib/format";
import { Legend } from "./NowView";

const CODE: Record<string, Tier> = { L: "Low", M: "Medium", H: "High" };
const HOUR_MS = 3_600_000;
const STEP_MS = 180; // a 5-minute tick per frame: an hour of storm in about 2 seconds

type Row = { area: string; firstAlert: string; floods: number; warnedFrom: string | null; leadMin: number | null };

/** For each area PUB alerted, whether FloodSense's warning was already up in the hour before PUB's first alert, and
 * since when (the start of that unbroken run of Medium or High ticks). */
function compare(replay: Replay): Row[] {
  const times = replay.ticks.map((t) => new Date(t).getTime());
  const first = new Map<string, Alert[]>();
  for (const a of replay.alerts) first.set(a.area, [...(first.get(a.area) ?? []), a]);
  return [...first].map(([area, alerts]) => {
    const at = new Date(alerts[0].ts).getTime();
    const codes = replay.tiers[area] ?? "";
    let i = -1;
    for (let k = 0; k < times.length && times[k] <= at; k++) if (times[k] > at - HOUR_MS && codes[k] !== "L") i = k;
    let start = i;
    while (start > 0 && codes[start - 1] !== "L") start--;
    const warnedFrom = i >= 0 ? replay.ticks[start] : null;
    return {
      area,
      firstAlert: alerts[0].ts,
      floods: alerts.filter((a) => a.kind === "flash_flood").length,
      warnedFrom,
      leadMin: warnedFrom ? (at - new Date(warnedFrom).getTime()) / 60_000 : null,
    };
  });
}

export default function ReplayView() {
  const list = useSWR<Replays>("/api/replays", getJson, { revalidateOnFocus: false });
  const [day, setDay] = useState<string | null>(null);
  const chosen = day ?? list.data?.days[0]?.day ?? null;

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
          Watch FloodSense rate every area through a real storm day, minute by minute, next to the alerts PUB actually sent.
          {list.data
            ? `These storms are all from ${longDay(list.data.from)} on, after the model's training data ended, so it never saw them.`
            : "These storms came after the model's training data ended, so it never saw them."}
        </p>
        {list.data ? (
          list.data.days.length ? (
            <ul className="storms" aria-label="Storm days">
              {list.data.days.map((d) => (
                <li key={d.day}>
                  <button aria-pressed={d.day === chosen} onClick={() => setDay(d.day)}>
                    <strong>{longDay(d.day)}</strong>
                    <span>
                      {d.alerts} PUB alert{d.alerts === 1 ? "" : "s"}
                      {d.floods ? `, ${d.floods} flash flood${d.floods === 1 ? "" : "s"}` : ""} in {d.areas} area
                      {d.areas === 1 ? "" : "s"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="state">No storms since the training data ended yet. They appear here as PUB sends alerts.</p>
          )
        ) : (
          <div className="skeleton skeleton-block" />
        )}
      </div>
      {chosen ? <ReplayDay key={chosen} day={chosen} /> : null}
    </div>
  );
}

function ReplayDay({ day }: { day: string }) {
  const shapes = useAreaShapes();
  const replay = useSWR<Replay>(`/api/replays/${day}`, getJson, { revalidateOnFocus: false, revalidateIfStale: false });
  const data = replay.data;
  const rows = useMemo(() => (data ? compare(data) : []), [data]);
  // start an hour before PUB's first alert, so the build-up is in view
  const startIndex = useMemo(() => {
    if (!data?.alerts.length) return 0;
    const first = new Date(data.alerts[0].ts).getTime() - HOUR_MS;
    return Math.max(0, data.ticks.findIndex((t) => new Date(t).getTime() >= first));
  }, [data]);
  const [index, setIndex] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const at = index ?? startIndex;
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!playing || !data) return;
    timer.current = window.setInterval(() => {
      setIndex((i) => {
        const next = (i ?? startIndex) + 1;
        if (next >= data.ticks.length) {
          setPlaying(false);
          return data.ticks.length - 1;
        }
        return next;
      });
    }, STEP_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [playing, data, startIndex]);

  const tiers = useMemo(() => {
    if (!data) return {};
    return Object.fromEntries(Object.entries(data.tiers).map(([area, codes]) => [area, CODE[codes[at]] ?? "Low"]));
  }, [data, at]);
  const markers = useMemo<Marker[]>(() => {
    if (!data) return [];
    const now = new Date(data.ticks[at]).getTime();
    return data.alerts
      .filter((a) => new Date(a.ts).getTime() <= now)
      .map((a) => ({ area: a.area, kind: a.kind, recent: now - new Date(a.ts).getTime() < HOUR_MS }));
  }, [data, at]);

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
          Running the model over {longDay(day)}: 47 areas, every 5 minutes. The first replay of a day takes up to a minute.
        </p>
        <div className="skeleton skeleton-map" />
      </div>
    );
  }

  const tickTime = data.ticks[at];
  return (
    <div className="replay-stage">
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
          label={`FloodSense ratings at ${clock(tickTime)} on ${longDay(day)}, with PUB's alerts so far.`}
        />
        <figcaption>
          <Legend />
          <span className="caption-note">
            <span className="dot" data-kind="warning" /> PUB drain warning
            <span className="dot" data-kind="flash_flood" /> PUB flash flood
          </span>
        </figcaption>
      </figure>

      <div className="scrubber">
        <button
          onClick={() => {
            if (playing) return setPlaying(false);
            if (at >= data.ticks.length - 1) setIndex(0); // at the end: play from the start
            setPlaying(true);
          }}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={data.ticks.length - 1}
          value={at}
          onChange={(e) => {
            setPlaying(false);
            setIndex(Number(e.target.value));
          }}
          aria-label="Time of day"
          aria-valuetext={clock(tickTime)}
        />
      </div>

      <table className="leads">
        <caption>Did FloodSense warn before PUB on {longDay(day)}?</caption>
        <thead>
          <tr>
            <th scope="col">Area</th>
            <th scope="col">PUB's first alert</th>
            <th scope="col">FloodSense warned from</th>
            <th scope="col">Earlier by</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.area}>
              <th scope="row">
                {r.area}
                {r.floods ? <span className="flood-tag">flash flood</span> : null}
              </th>
              <td>{clock(r.firstAlert)}</td>
              <td>{r.warnedFrom ? clock(r.warnedFrom) : "No warning in the hour before"}</td>
              <td>{r.leadMin != null ? duration(r.leadMin) : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="fine-print">
        A warning counts when the area was Medium or High at some point in the hour before PUB's first alert; "warned from" is
        when that unbroken warning began. Areas without alerts that turned Medium are possible false alarms. Rated by
        {" "}{data.model}.
      </p>
    </div>
  );
}
