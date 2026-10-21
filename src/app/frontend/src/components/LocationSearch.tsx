import { useState } from "react";
import { getJson, type Located } from "../lib/api";

type Props = { onFound: (area: string, place: string) => void };

/** A street, building or postal code, or the phone's own location, to the forecast area it falls in. */
export function LocationSearch({ onFound }: Props) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    if (q.trim().length < 2) return;
    setBusy(true);
    setMessage(null);
    try {
      const found = await getJson<Located>(`/api/locate?q=${encodeURIComponent(q.trim())}`);
      onFound(found.area, found.place);
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
        setMessage("Location access was turned down. Search for a street or postal code instead.");
      },
      { timeout: 10_000 },
    );
  }

  return (
    <form className="search" onSubmit={search} role="search">
      <label htmlFor="place">Check a street, building or postal code</label>
      <div className="search-row">
        <input
          id="place"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. 588996 or Dunearn Road"
          autoComplete="street-address"
          enterKeyHint="search"
        />
        <button type="submit" disabled={busy || q.trim().length < 2}>
          {busy ? "Checking…" : "Check"}
        </button>
      </div>
      <button type="button" className="link" onClick={useMyLocation} disabled={busy}>
        Use my location
      </button>
      {message ? (
        <p className="search-message" role="status">
          {message}
        </p>
      ) : null}
    </form>
  );
}
