import { lazy, Suspense, useEffect, useState } from "react";
import { useNow } from "./lib/api";
import { clock, minutesSince } from "./lib/format";
import { NowView } from "./views/NowView";

// the live view loads first; the replay (with its model run) and the track record load when opened
const ReplayView = lazy(() => import("./views/ReplayView"));
const ReliabilityView = lazy(() => import("./views/ReliabilityView"));

const VIEWS = [
  { id: "now", label: "Now" },
  { id: "replay", label: "Replay a storm" },
  { id: "reliability", label: "How reliable is it?" },
] as const;
type View = (typeof VIEWS)[number]["id"];

function fromHash(): View {
  const id = window.location.hash.slice(1);
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
          {view === "now" ? <NowView /> : view === "replay" ? <ReplayView /> : <ReliabilityView />}
        </Suspense>
      </main>
      <footer className="footer">
        <p>
          Flash-flood alerts for Singapore's 47 forecast areas, one hour ahead, every 5 minutes. Built on open data from
          NEA, PUB and UHSLC (data.gov.sg, Singapore Open Data Licence). Not an official warning: follow PUB.
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
