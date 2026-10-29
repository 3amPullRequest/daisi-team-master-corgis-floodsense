import { useRef, useState } from "react";
import { getJson, type Located } from "../lib/api";
import { placeName } from "../lib/places";

type Props = {
  areas: string[];
  /** the area found, and the place the visitor asked about (null when they picked the area itself) */
  onFound: (area: string, place: string | null) => void;
};

/** A street, building or postal code, the phone's own location, or an area from the list, to its forecast area. */
export function LocationSearch({ areas, onFound }: Props) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    if (q.trim().length < 2) {
      setMessage("Type a street, building or postal code first.");
      input.current?.focus();
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const found = await getJson<Located>(`/api/locate?q=${encodeURIComponent(q.trim())}`);
      onFound(found.area, placeName(found.place));
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function useMyLocation() {
    if (!navigator.geolocation) {
      setMessage("This browser can't share your location. Search for a street or postal code instead.");
      return;
    }
    setBusy(true);
    setMessage(null);
    navigator.geolocation.getCurrentPosition(
      async ({ coords }) => {
        try {
          const { area } = await getJson<{ area: string }>(`/api/nearest?lat=${coords.latitude}&lon=${coords.longitude}`);
          onFound(area, "your location");
        } catch {
          setMessage("Your location seems to be outside Singapore. Search for a street or postal code instead.");
        } finally {
          setBusy(false);
        }
      },
      () => {
        setBusy(false);
        setMessage("This browser didn't share your location. Search for a street or postal code instead.");
      },
      { timeout: 10_000 },
    );
  }

  return (
    <form className="search" onSubmit={search} role="search">
      <label htmlFor="place">Check your street, building or postal code</label>
      <div className="search-row">
        <input
          id="place"
          ref={input}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. 588996 or Dunearn Road"
          autoComplete="street-address"
          enterKeyHint="search"
        />
        <button type="submit" disabled={busy}>
          {busy ? "Checking…" : "Check"}
        </button>
      </div>
      <div className="search-more">
        <button type="button" className="link" onClick={useMyLocation} disabled={busy}>
          Use my location
        </button>
        <span aria-hidden="true">or</span>
        <select
          aria-label="Pick an area from the list"
          value=""
          onChange={(e) => {
            if (e.target.value) onFound(e.target.value, null);
          }}
        >
          <option value="">Pick an area</option>
          {areas.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </div>
      {message ? (
        <p className="search-message" role="status">
          {message}
        </p>
      ) : null}
    </form>
  );
}
