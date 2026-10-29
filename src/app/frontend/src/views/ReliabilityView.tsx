import useSWR from "swr";
import { AskSection } from "../components/AskPanel";
import { TideLine } from "../components/TideLine";
import { getJson, useNow, type TrackRecord } from "../lib/api";
import { ASK_QUESTIONS, PUB_CHANNEL } from "../lib/content";
import { longDay, percent } from "../lib/format";

const DAYS_PER_MONTH = 30.4;

/** A share as people say it: "About 8 in 10", or "About 1 in 4" below a half. */
function inWords(x: number | undefined): string {
  if (!x) return "–";
  return x >= 0.5 ? `About ${Math.round(x * 10)} in 10` : `About 1 in ${Math.round(1 / x)}`;
}

/** The time between false alarms in one area: weeks when they are rare, days when they are not. */
function between(perAreaMonth: number | undefined): string {
  if (!perAreaMonth) return "–";
  const days = DAYS_PER_MONTH / perAreaMonth;
  return days >= 14 ? `${Math.round(days / 7)} weeks` : `${Math.round(days)} days`;
}

export default function ReliabilityView() {
  const { data, error, mutate } = useSWR<TrackRecord>("/api/track-record", getJson, { revalidateOnFocus: false });
  const now = useNow();

  if (error) {
    return (
      <div className="state state-error" role="alert">
        <h1>The track record didn't load.</h1>
        <p>{error.message}</p>
        <button onClick={() => mutate()}>Try again</button>
      </div>
    );
  }
  if (!data) return <div className="skeleton skeleton-block" aria-busy="true" />;

  const m = data.metrics;
  const share = m.oof_alert_flagged_before_pub;
  // The mean lead counts misses as 0; divided by the share warned, it is the lead when FloodSense did warn first. The
  // scoreboard counts each rating as a warning for the hour after it (ml/scoreboard.py), as the replays do.
  const leadWhenWarned = share ? m.oof_alert_mean_lead_min / share : undefined;
  // the tiers as fixed chances (final_1) log High's cut-off, and Elevated alone is scored at about the rule's false alarms
  const fixed = m.pair_high_p != null;
  // flash floods one by one (oof_floods.json): the mean lead counts misses as 0, so per warned flood it is larger
  const floodLead = m.oof_floods_warned ? (m.oof_floods_mean_lead_min * m.oof_floods_judged) / m.oof_floods_warned : undefined;
  const [from, to] = (data.development ?? "").split(" to ");
  const tide = now.data?.tide;

  return (
    <article className="reliability">
      <h1 className="headline">How reliable is it?</h1>
      <p className="headline-detail">
        We tested FloodSense on {m.oof_alert_events} PUB alerts from {m.oof_alert_days} storm days
        {from ? `, between ${longDay(from)} and ${longDay(to)}` : ""}. Each test used a version of FloodSense that had not
        seen that storm.
      </p>

      <ul className="claims">
        <li>
          <p className="claim-number">{inWords(share)}</p>
          <p>
            PUB alerts had a FloodSense warning out before PUB's own. That is {percent(share)}. With so few storms, the
            true share could be anywhere from {percent(m.oof_alert_flagged_before_pub_lo)} to{" "}
            {percent(m.oof_alert_flagged_before_pub_hi)}.
          </p>
        </li>
        <li>
          <p className="claim-number">{leadWhenWarned ? `${Math.round(leadWhenWarned)} min` : "–"}</p>
          <p>
            earlier than PUB, on average, when FloodSense warned first. Each Moderate or Elevated rating is a warning
            for the hour after it, and the head start runs from the warning's first rating.
          </p>
        </li>
        {m.oof_floods_judged ? (
          <li>
            <p className="claim-number">
              {m.oof_floods_warned} of {m.oof_floods_judged}
            </p>
            <p>
              flash floods had a FloodSense warning out when they hit
              {floodLead ? `, on average ${Math.round(floodLead)} min before` : ""}. For {m.oof_floods_warned_before_pub} of
              them, the warning came before PUB's first alert of the storm.
            </p>
          </li>
        ) : null}
        <li>
          <p className="claim-number">{between(m.oof_false_alarms_per_area_month)}</p>
          <p>
            between false alarms in a typical area. A false alarm is a Moderate or Elevated warning with no PUB alert
            while it was in force. That is {m.oof_false_alarms_per_area_month?.toFixed(2)} false alarms per area per
            month.
          </p>
        </li>
        {m.oof_medium_warning_alert_share != null ? (
          <li>
            <p className="claim-number">{inWords(m.oof_medium_warning_alert_share)}</p>
            <p>
              warnings had a PUB alert while they were in force.
              {m.oof_high_warning_alert_share != null
                ? ` At Elevated, ${inWords(m.oof_high_warning_alert_share).toLowerCase()} did.`
                : ""}
            </p>
          </li>
        ) : null}
      </ul>

      <section className="compare">
        <h2>Better than a simple rain alarm</h2>
        <p>
          A simple rule warns once heavy rain has fallen nearby. The rule warned early for{" "}
          {percent(m.rule_dev_alert_flagged_before_pub)} of the same alerts, with{" "}
          {m.rule_dev_false_alarms_per_area_month?.toFixed(2)} false alarms per area per month.
          {fixed && m.oof_high_alert_flagged_before_pub != null
            ? ` At Elevated alone, FloodSense raised about as many false alarms (${m.oof_high_false_alarms_per_area_month?.toFixed(2)} per area per month) and warned early for ${percent(m.oof_high_alert_flagged_before_pub)}.`
            : ""}{" "}
          FloodSense also reads NEA's 2-hour forecast and the tide, so it can warn before the rain arrives.
        </p>
        {m.without_tide_alert_flagged_only_with != null ? (
          <p>
            The tide helps. Adding it gained {m.without_tide_alert_flagged_only_with} early warnings and lost{" "}
            {m.without_tide_alert_flagged_only_without}.
          </p>
        ) : null}
      </section>

      <section className="how">
        <h2>How FloodSense works</h2>
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
              ? ` Moderate starts at a ${percent(m.pair_medium_p)} chance of a PUB alert within the hour, and Elevated at ${percent(m.pair_high_p)}.`
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
        intro="Ask how FloodSense did, by area, by year or against the rainfall rule."
      />

      <p>
        Town councils and agencies can check every storm on the <a href="#agencies">scoreboard</a>.
      </p>

      <p className="fine-print">
        Model {data.model}, trained {longDay(new Date(data.trained_at).toISOString())}. The app reads these figures from
        the model's MLflow run in Unity Catalog (alias @champion).
      </p>
    </article>
  );
}
