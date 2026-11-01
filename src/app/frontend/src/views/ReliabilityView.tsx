import useSWR from "swr";
import { AskSection } from "../components/AskPanel";
import { TideLine } from "../components/TideLine";
import { getJson, useNow, type TrackRecord } from "../lib/api";
import { ASK_QUESTIONS, PUB_CHANNEL } from "../lib/content";
import { longDay, percent } from "../lib/format";

/** A share as people say it: "about 1 in 13", or "about 8 in 10" from a half up. */
function inWords(x: number | null | undefined): string {
  if (!x) return "none";
  return x >= 0.5 ? `about ${Math.round(x * 10)} in 10` : `about 1 in ${Math.round(1 / x)}`;
}

export default function ReliabilityView() {
  const { data, error, mutate } = useSWR<TrackRecord>("/api/track-record", getJson, { revalidateOnFocus: false });
  const now = useNow();
  const tide = now.data?.tide;
  // the tiers as fixed chances (final_1 on) log their cut-offs
  const fixed = data?.medium_p != null && data?.high_p != null;

  return (
    <article className="reliability">
      <h1 className="headline">How it works</h1>

      <section className="how">
        <ol className="steps">
          <li>
            <strong>Every 5 minutes, it reads open data.</strong> NEA's rain gauges, NEA's 2-hour forecast and the tide at
            Tanjong Pagar.
          </li>
          <li>
            <strong>It compares each area's weather with the weather before past PUB alerts.</strong> It learned from
            PUB's alerts from 2021 to 2025, area by area.
          </li>
          <li>
            <strong>It rates each of NEA's 47 forecast areas for the next hour.</strong> Low, Moderate or Elevated, with
            the reason in plain words.
            {fixed
              ? ` Moderate starts at a ${percent(data.medium_p)} chance of a PUB alert within the hour, and Elevated at ${percent(data.high_p)}.`
              : ""}
          </li>
        </ol>
        {tide && tide.curve.length > 1 ? (
          <figure className="tide-box">
            <TideLine curve={tide.curve} now={tide.now} />
            <figcaption>Predicted tide at Tanjong Pagar, from 6 hours ago to 18 hours ahead.</figcaption>
          </figure>
        ) : null}
      </section>

      <section className="limits">
        <h2>What it can't do yet</h2>
        <ul>
          <li>Most of its warnings end without a PUB alert. Treat it as an early heads-up, not a replacement for PUB.</li>
          <li>It can't tell when a flash flood itself is likely, only when a PUB alert is more likely.</li>
          <li>It rates NEA's 47 forecast areas, not single streets.</li>
        </ul>
        <p>
          For official warnings,{" "}
          <a href={PUB_CHANNEL} target="_blank" rel="noreferrer">
            follow PUB Flood Alerts on Telegram
          </a>
          .
        </p>
      </section>

      <AskSection
        questions={ASK_QUESTIONS.reliability}
        intro="Ask how FloodSense did since 20 Dec 2025: flood by flood, or area by area."
      />

      <Record data={data} error={error} retry={() => mutate()} />

      <p>
        Town councils and agencies can check every flash flood on the <a href="#agencies">scoreboard</a>.
      </p>

      {data ? (
        <p className="fine-print">
          Model {data.model}, trained {longDay(new Date(data.trained_at).toISOString())}.
          {data.test
            ? ` The figures above are for ${data.test.model === data.model ? "it" : `model ${data.test.model}`}, from the scoreboard published for agencies in Unity Catalog.`
            : ""}
        </p>
      ) : null}
    </article>
  );
}

/** How reliable FloodSense has been on the storms after its training data ended. It loads on its own, so the rest of
 * the page shows while it does, or if it fails. */
function Record({ data, error, retry }: { data?: TrackRecord; error?: Error; retry: () => void }) {
  if (error) {
    return (
      <section className="record" role="alert">
        <h2>How reliable is it?</h2>
        <p>The track record didn't load. {error.message}</p>
        <button onClick={retry}>Try again</button>
      </section>
    );
  }
  if (!data) return <div className="skeleton skeleton-block" aria-busy="true" />;
  const t = data.test;
  if (!t) {
    return (
      <section className="record">
        <h2>How reliable is it?</h2>
        <p>Its scoreboard since 20 Dec 2025 hasn't been published yet.</p>
      </section>
    );
  }
  const m = t.moderate;
  const e = t.elevated;
  const lead = m.flood_mean_lead_min;
  // a warning in force, but from under 5 minutes or over an hour before the flood
  const outside = m.floods_warned - m.floods_caught;

  return (
    <section className="record" aria-labelledby="record-title">
      <h2 id="record-title">How reliable is it?</h2>
      <p>
        Tested on the storms from {longDay(t.from)} to {longDay(t.to)}, after its training data ended: {m.flash_floods}{" "}
        flash floods and {m.warnings} warnings.
      </p>
      <ul className="claims">
        <li>
          <p className="claim-number">
            {m.floods_caught} of {m.flash_floods}
          </p>
          <p>
            flash floods caught: a FloodSense warning came 5 minutes to an hour before each.
            {outside > 0
              ? ` A warning was in force at ${outside} more, but it came less than 5 minutes or more than an hour before the flood.`
              : ""}
            {m.floods_caught_share_lo != null && m.floods_caught_share_hi != null
              ? ` With so few floods, the true share could be anywhere from ${percent(m.floods_caught_share_lo)} to ${percent(m.floods_caught_share_hi)}.`
              : ""}
          </p>
        </li>
        <li>
          <p className="claim-number">{lead != null ? `${lead.toFixed(1)} min` : "–"}</p>
          <p>
            on average from FloodSense's warning to the flash flood, over the {m.floods_warned} floods a warning was in
            force at. The warning starts at its first Moderate or Elevated rating.
          </p>
        </li>
        <li>
          <p className="claim-number">{percent(m.warnings_followed_by_alert, 1)}</p>
          <p>
            chance of a PUB alert, given a Moderate or Elevated warning: {inWords(m.warnings_followed_by_alert)} warnings
            had one while in force.
            {e?.warnings_followed_by_alert != null
              ? ` At Elevated alone, ${percent(e.warnings_followed_by_alert, 1)}.`
              : ""}
          </p>
        </li>
        <li>
          <p className="claim-number">{percent(m.warnings_followed_by_flood, 1)}</p>
          <p>
            chance of a flash flood, given a Moderate or Elevated warning: {inWords(m.warnings_followed_by_flood)}{" "}
            warnings had one while in force.
            {e?.warnings_followed_by_flood != null
              ? ` At Elevated alone, ${percent(e.warnings_followed_by_flood, 1)}.`
              : ""}
          </p>
        </li>
      </ul>
    </section>
  );
}
