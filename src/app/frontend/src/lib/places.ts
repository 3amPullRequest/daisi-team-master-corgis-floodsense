import { useCallback, useState } from "react";
import { AUDIENCES, type Audience } from "./content";

/** A place the visitor saved: what they searched for (or the area's name) and the forecast area it falls in. */
export type Place = { label: string; area: string };

const PLACES_KEY = "floodsense.places";
const AUDIENCE_KEY = "floodsense.audience";
const SUBSCRIPTIONS_KEY = "floodsense.subscriptions";
const MAX_PLACES = 4;

// Browser storage can be missing or refuse writes (private windows, blocked site data): the app then works without
// remembering anything.
function read<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw == null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // not remembered: fine for a convenience
  }
}

/** The visitor's saved places, kept in this browser only, newest last. */
export function usePlaces() {
  const [places, setPlaces] = useState<Place[]>(() => {
    const saved = read<Place[]>(PLACES_KEY, []);
    return Array.isArray(saved) ? saved.filter((p) => p && typeof p.area === "string" && typeof p.label === "string") : [];
  });
  const toggle = useCallback((place: Place) => {
    setPlaces((current) => {
      const has = current.some((p) => p.area === place.area && p.label === place.label);
      const next = has
        ? current.filter((p) => !(p.area === place.area && p.label === place.label))
        : [...current, place].slice(-MAX_PLACES);
      write(PLACES_KEY, next);
      return next;
    });
  }, []);
  return { places, toggle };
}

/** Which advice the visitor last read: commuters see theirs first. */
export function useAudience() {
  const [audience, setAudience] = useState<Audience>(() => {
    const saved = read<string>(AUDIENCE_KEY, "commute");
    return AUDIENCES.some((a) => a.id === saved) ? (saved as Audience) : "commute";
  });
  const choose = useCallback((a: Audience) => {
    setAudience(a);
    write(AUDIENCE_KEY, a);
  }, []);
  return [audience, choose] as const;
}

/** The ids of the warning subscriptions made in this browser, so it can show and stop them. Signed-in visitors also
 * see the ones they made elsewhere: the server knows them by their Databricks account. */
export function useSubscriptionIds() {
  const [ids, setIds] = useState<string[]>(() => {
    const saved = read<string[]>(SUBSCRIPTIONS_KEY, []);
    return Array.isArray(saved) ? saved.filter((id) => typeof id === "string") : [];
  });
  const remember = useCallback((id: string) => {
    setIds((current) => {
      const next = [...current.filter((i) => i !== id), id].slice(-20);
      write(SUBSCRIPTIONS_KEY, next);
      return next;
    });
  }, []);
  const forget = useCallback((id: string) => {
    setIds((current) => {
      const next = current.filter((i) => i !== id);
      write(SUBSCRIPTIONS_KEY, next);
      return next;
    });
  }, []);
  return { ids, remember, forget };
}

/** An address from OneMap ("1 DUNEARN ROAD SINGAPORE 309415") in the case people write it. */
export function placeName(raw: string): string {
  return raw.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}
