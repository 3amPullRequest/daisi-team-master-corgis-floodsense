import { useEffect, useState } from "react";
import useSWR from "swr";
import { Ask } from "../components/AskPanel";
import { type Agencies, getJson } from "../lib/api";
import { AGENCY_QUESTIONS } from "../lib/content";

const TABS = [
  { id: "scoreboard", label: "Scoreboard" },
  { id: "ask", label: "Ask a question" },
] as const;
type Tab = (typeof TABS)[number]["id"];

// #agencies/ask opens on the questions
function tabFromHash(): Tab {
  return window.location.hash.slice(1).split("/")[1] === "ask" ? "ask" : "scoreboard";
}

export default function AgenciesView() {
  const { data, error, mutate } = useSWR<Agencies>("/api/agencies", getJson, { revalidateOnFocus: false });
  const [tab, setTab] = useState<Tab>(tabFromHash);
  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <article className="agencies">
      <h1 className="headline">For agencies</h1>
      <p className="headline-detail">
        How FloodSense did, storm by storm, and every PUB alert since 2021. For town councils and agencies planning
        manpower and flood-relief resources.
      </p>
      <div className="advice-tabs agency-tabs" role="tablist" aria-label="For agencies">
        {TABS.map((t) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            role="tab"
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            onClick={() => {
              setTab(t.id);
              window.history.replaceState(null, "", `#agencies/${t.id}`);
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {error ? (
          <div className="state state-error" role="alert">
            <h1>The scoreboard didn't load.</h1>
            <p>{error.message}</p>
            <button onClick={() => mutate()}>Try again</button>
          </div>
        ) : !data ? (
          <div className="skeleton skeleton-block" aria-busy="true" />
        ) : tab === "scoreboard" ? (
          <Scoreboard links={data} />
        ) : (
          <Ask links={data} questions={AGENCY_QUESTIONS} />
        )}
      </div>
    </article>
  );
}

function Scoreboard({ links }: { links: Agencies }) {
  if (!links.dashboard) {
    return <p className="state">This copy of the app has no scoreboard set up.</p>;
  }
  return (
    <section className="board">
      <iframe
        className="board-frame"
        src={links.dashboard.embed_url}
        title="FloodSense scoreboard, an AI/BI dashboard"
        loading="lazy"
      />
      <p className="fine-print">
        A Databricks AI/BI dashboard, shown here when you're signed in to Databricks in this browser. Blank, or asking
        you to sign in?{" "}
        <a href={links.dashboard.url} target="_blank" rel="noreferrer">
          Open the scoreboard in Databricks
        </a>
        .
      </p>
    </section>
  );
}
