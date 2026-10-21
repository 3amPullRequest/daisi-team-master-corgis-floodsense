import useSWR from "swr";
import { getJson, type TrackRecord } from "../lib/api";
import { PUB_CHANNEL } from "../lib/content";
import { longDay, percent } from "../lib/format";

export default function ReliabilityView() {
  const { data, error, mutate } = useSWR<TrackRecord>("/api/track-record", getJson, { revalidateOnFocus: false });

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
  // the mean lead counts misses as 0; divided by the share warned, it is the lead when FloodSense did warn first
  const leadWhenWarned = share ? m.oof_alert_mean_lead_min / share : undefined;
  const weeksPerFalseAlarm = m.oof_false_alarms_per_area_month ? 4.35 / m.oof_false_alarms_per_area_month : undefined;
  const [from, to] = (data.development ?? "").split(" to ");

  return (
    <article className="reliability">
      <h1 className="headline">How reliable is it?</h1>
      <p className="headline-detail">
        FloodSense was tested on {m.oof_alert_events} PUB alerts over {m.oof_alert_days} storm days
        {from ? ` between ${longDay(from)} and ${longDay(to)}` : ""}, each judged by a version of the model that hadn't
        seen that storm.
      </p>

      <ul className="claims">
        <li>
          <p className="claim-number">About 1 in {share ? Math.round(1 / share) : "–"}</p>
          <p>
            PUB alerts had a FloodSense warning out before PUB's own ({percent(share)}; somewhere between{" "}
            {percent(m.oof_alert_flagged_before_pub_lo)} and {percent(m.oof_alert_flagged_before_pub_hi)} given how
            few storms there were).
          </p>
        </li>
        <li>
          <p className="claim-number">{leadWhenWarned ? `${Math.round(leadWhenWarned)} min` : "–"}</p>
          <p>earlier than PUB, on average, when FloodSense did warn first.</p>
        </li>
        <li>
          <p className="claim-number">{weeksPerFalseAlarm ? `${Math.round(weeksPerFalseAlarm)} weeks` : "–"}</p>
          <p>
            between false alarms in a typical area: a Medium or High warning with no PUB alert following within the
            hour ({m.oof_false_alarms_per_area_month?.toFixed(2)} per area per month).
          </p>
        </li>
      </ul>

      <section className="compare">
        <h2>Better than a simple rain alarm</h2>
        <p>
          A rule that warns once heavy rain has fallen nearby caught {percent(m.rule_dev_alert_flagged_before_pub)} of the
          same alerts early, with {m.rule_dev_false_alarms_per_area_month?.toFixed(2)} false alarms per area per month.
          FloodSense also reads NEA's 2-hour forecast and the tide, so it can warn before the rain arrives.
        </p>
        {m.without_tide_alert_flagged_only_with != null ? (
          <p>
            Adding the tide helped: {m.without_tide_alert_flagged_only_with} storms were caught early only with it, against{" "}
            {m.without_tide_alert_flagged_only_without} only without it.
          </p>
        ) : null}
      </section>

      <section className="limits">
        <h2>What it can't do yet</h2>
        <ul>
          <li>It misses most alerts: it is an early heads-up, not a replacement for PUB.</li>
          <li>Its High level isn't proven yet: too few flash floods in the test years to check it.</li>
          <li>It works at the level of NEA's 47 forecast areas, not individual streets.</li>
        </ul>
        <p>
          For official warnings, <a href={PUB_CHANNEL} target="_blank" rel="noreferrer">follow PUB Flood Alerts on Telegram</a>.
        </p>
      </section>

      <p className="fine-print">
        Model {data.model}, trained {longDay(new Date(data.trained_at).toISOString())}, read from its MLflow run in Unity
        Catalog (@champion).
      </p>
    </article>
  );
}
