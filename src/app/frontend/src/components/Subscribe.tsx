import { useId, useMemo, useState } from "react";
import useSWR from "swr";
import { getJson, postJson, type Subscription } from "../lib/api";
import { clock } from "../lib/format";
import { type Place, useSubscriptionIds } from "../lib/places";

const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const SG_MOBILE = /^(?:\+?65)?[89]\d{7}$/; // the server checks the same
const MAX_AREAS = 10;
// Texts are off (31 Oct): Singapore relabels an unregistered SMS sender "Likely-SCAM", and registering one needs a UEN.
// The server and broadcast.py still accept "sms", so putting it back is one line here.
const CHANNELS = ["email"] as const satisfies readonly ("email" | "sms")[];

type Props = {
  areas: string[]; // all 47, for adding one by name
  places: Place[]; // the visitor's saved places
  shown: string | null; // the area in the card now
};

/** Sign up for FloodSense's warnings by email for chosen areas, and see or stop the ones already made.
 * Following the visitor's own location is shown, switched off, as coming soon. */
export function Subscribe({ areas, places, shown }: Props) {
  const { ids, remember, forget } = useSubscriptionIds();
  const key = `/api/subscriptions?ids=${ids.join(",")}`;
  const mine = useSWR<{ subscriptions: Subscription[] }>(key, getJson, { revalidateOnFocus: false });
  const [open, setOpen] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const title = useId();

  return (
    <section className="subscribe" aria-labelledby={title}>
      <h2 id={title}>Warnings by email</h2>
      <p className="subscribe-intro">
        Get a message when FloodSense warns for your places. Each warning covers the next hour.
      </p>
      {mine.data?.subscriptions.length ? (
        <ul className="subs">
          {mine.data.subscriptions.map((s) => (
            <li key={s.id}>
              <p>
                <strong>{s.areas.join(", ")}</strong>
                <span>
                  {s.channel === "email" ? "Email to" : "Texts to"} {s.address}
                </span>
                <span className="sub-status">{welcomeStatus(s)}</span>
              </p>
              <button
                type="button"
                className="link"
                aria-label={`Stop warnings for ${s.areas.join(", ")}`}
                onClick={async () => {
                  setProblem(null);
                  try {
                    await postJson(`/api/subscriptions/${s.id}`, undefined, "DELETE");
                    forget(s.id);
                  } catch (err) {
                    setProblem((err as Error).message);
                  } finally {
                    void mine.mutate();
                  }
                }}
              >
                Stop
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {problem ? (
        <p className="search-message" role="alert">
          {problem}
        </p>
      ) : null}
      {open ? (
        <SignUp
          areas={areas}
          places={places}
          shown={shown}
          onDone={(made) => {
            remember(made.id);
            setOpen(false);
            void mine.mutate();
          }}
          onCancel={() => setOpen(false)}
        />
      ) : (
        <button type="button" className="secondary" onClick={() => setOpen(true)}>
          Get warnings by email
        </button>
      )}
    </section>
  );
}

/** How the welcome message went, in words: flood-live sends it on its next run (ml/broadcast.py). */
function welcomeStatus(s: Subscription): string {
  if (s.welcome === "sent") return `Welcome message sent at ${clock(s.welcomed_at)}`;
  if (!s.welcome) return "FloodSense sends a welcome message on its next run.";
  if (s.welcome.startsWith("texts aren't set up")) return "Texts aren't switched on yet. You'll get them once they are.";
  if (s.welcome.startsWith("email isn't set up")) return "Email isn't switched on yet. You'll get it once it is.";
  return "The welcome message didn't go through. FloodSense will try again.";
}

type SignUpProps = {
  areas: string[];
  places: Place[];
  shown: string | null;
  onDone: (made: Subscription) => void;
  onCancel: () => void;
};

function SignUp({ areas, places, shown, onDone, onCancel }: SignUpProps) {
  // the saved places, then the area in the card, each ticked to start with
  const options = useMemo(() => {
    const out: Place[] = [...places];
    if (shown && !out.some((p) => p.area === shown)) out.push({ label: shown, area: shown });
    return out;
  }, [places, shown]);
  const [added, setAdded] = useState<string[]>([]);
  const [picked, setPicked] = useState<string[]>(() => [...new Set(options.map((p) => p.area))]);
  const [channel, setChannel] = useState<"email" | "sms">(CHANNELS[0]);
  const [address, setAddress] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const addressId = useId();

  const choices = [...options, ...added.filter((a) => !options.some((p) => p.area === a)).map((a) => ({ label: a, area: a }))];
  const toggle = (area: string) =>
    setPicked((now) => (now.includes(area) ? now.filter((a) => a !== area) : [...now, area].slice(0, MAX_AREAS)));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const typed = address.trim();
    if (!picked.length) return setMessage("Tick at least one place.");
    if (channel === "email" && !EMAIL.test(typed)) return setMessage("That doesn't look like an email address.");
    if (channel === "sms" && !SG_MOBILE.test(typed.replace(/[\s()-]/g, "")))
      return setMessage("Enter a Singapore mobile number: 8 digits starting with 8 or 9.");
    setBusy(true);
    setMessage(null);
    try {
      const labels = choices.filter((p) => picked.includes(p.area) && p.label !== p.area).map((p) => p.label);
      const made = await postJson<Subscription>("/api/subscriptions", {
        channel,
        address: typed,
        areas: picked,
        places: labels,
      });
      onDone(made);
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="signup" onSubmit={submit} noValidate>
      <fieldset>
        <legend>Places</legend>
        <ul className="signup-places">
          {choices.map((p) => (
            <li key={`${p.area}|${p.label}`}>
              <label>
                <input type="checkbox" checked={picked.includes(p.area)} onChange={() => toggle(p.area)} />
                <span>
                  {p.label}
                  {p.label !== p.area ? <span className="signup-area"> ({p.area})</span> : null}
                </span>
              </label>
            </li>
          ))}
          <li className="soon">
            <label>
              <input type="checkbox" disabled />
              <span>Wherever I am, from my phone's location</span>
            </label>
            <span className="soon-tag">Coming soon</span>
          </li>
        </ul>
        <select
          aria-label="Add an area"
          value=""
          onChange={(e) => {
            const area = e.target.value;
            if (!area) return;
            setAdded((now) => (now.includes(area) ? now : [...now, area]));
            setPicked((now) => (now.includes(area) ? now : [...now, area].slice(0, MAX_AREAS)));
          }}
        >
          <option value="">Add an area</option>
          {areas.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </fieldset>

      <fieldset>
        <legend>Send them by</legend>
        {CHANNELS.length > 1 ? (
          <div className="advice-tabs" role="radiogroup" aria-label="Send them by">
            {CHANNELS.map((c) => (
              <button
                key={c}
                type="button"
                role="radio"
                aria-checked={channel === c}
                onClick={() => {
                  setChannel(c);
                  setMessage(null);
                }}
              >
                {c === "email" ? "Email" : "Text message"}
              </button>
            ))}
          </div>
        ) : null}
        <label htmlFor={addressId} className="signup-label">
          {channel === "email" ? "Email address" : "Singapore mobile number"}
        </label>
        <input
          id={addressId}
          className="signup-input"
          type={channel === "email" ? "email" : "tel"}
          inputMode={channel === "email" ? "email" : "tel"}
          autoComplete={channel === "email" ? "email" : "tel"}
          placeholder={channel === "email" ? "name@example.com" : "9123 4567"}
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
      </fieldset>

      {message ? (
        <p className="search-message" role="alert">
          {message}
        </p>
      ) : null}
      <div className="signup-actions">
        <button type="submit" disabled={busy}>
          {busy ? "Saving…" : "Get warnings"}
        </button>
        <button type="button" className="link" onClick={onCancel}>
          Cancel
        </button>
      </div>
      <p className="fine-print">
        FloodSense sends a welcome message first. After that, it writes when a warning starts, steps up or steps down,
        and stays quiet otherwise. It is not an official warning: PUB's alerts come first. Stop any time here.
      </p>
    </form>
  );
}
