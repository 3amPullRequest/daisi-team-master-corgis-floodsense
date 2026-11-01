import { lazy, Suspense, useEffect, useState } from "react";
import { useNow } from "./lib/api";
import { clock, minutesSince } from "./lib/format";
import { NowView } from "./views/NowView";

// the live view loads first; the replay (with its model run), how it works (with the track record) and the agencies'
// view load when opened
const ReplayView = lazy(() => import("./views/ReplayView"));
const ReliabilityView = lazy(() => import("./views/ReliabilityView"));
const AgenciesView = lazy(() => import("./views/AgenciesView"));

const VIEWS = [
  { id: "now", label: "Now" },
  { id: "replay", label: "Replay a storm" },
  { id: "reliability", label: "How it works" },
  { id: "agencies", label: "For agencies" },
] as const;
type View = (typeof VIEWS)[number]["id"];

// #replay/2026-04-11 opens the replay on that day, #agencies/ask the agencies' questions
function fromHash(): View {
  const id = window.location.hash.slice(1).split("/")[0];
  return VIEWS.some((v) => v.id === id) ? (id as View) : "now";
}

export function App() {
  const [view, setView] = useState<View>(fromHash);
  useEffect(() => {
    const onHash = () => setView(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <a href="#now" className="wordmark" aria-label="FloodSense, home">
          FloodSense
        </a>
        <nav aria-label="Views">
          {VIEWS.map((v) => (
            <a key={v.id} href={`#${v.id}`} aria-current={view === v.id ? "page" : undefined}>
              {v.label}
            </a>
          ))}
        </nav>
        <Freshness />
      </header>
      <main>
        <Suspense fallback={<div className="skeleton skeleton-block" aria-busy="true" />}>
          {view === "now" ? (
            <NowView />
          ) : view === "replay" ? (
            <ReplayView />
          ) : view === "reliability" ? (
            <ReliabilityView />
          ) : (
            <AgenciesView />
          )}
        </Suspense>
      </main>
      <footer className="footer">
        <p>
          FloodSense rates the flash-flood risk in NEA's 47 forecast areas for the next hour, every 5 minutes. It uses open
          data from NEA (data.gov.sg, Singapore Open Data Licence), tide records from UHSLC, and PUB's past flood alerts.
          It is not an official warning. For official alerts, follow PUB.
        </p>
      </footer>
    </div>
  );
}

/** When the newest forecast was made; SWR shares the request with the live view. */
function Freshness() {
  const { data } = useNow();
  const age = minutesSince(data?.tick);
  if (!data?.tick || age == null) return <span className="freshness" />;
  const live = age <= 20 && !data.stale;
  return (
    <span className="freshness" data-live={live || undefined}>
      {live ? "Live" : "Paused"}, updated {clock(data.tick)}
    </span>
  );
}
