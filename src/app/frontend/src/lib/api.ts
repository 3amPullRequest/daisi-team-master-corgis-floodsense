import useSWR from "swr";

export type Tier = "Low" | "Medium" | "High";

/** The strongest warning in another area whose label point is within 5 km: shown beside the area's own tier. */
export type Nearby = { area: string; tier: Tier; km: number; p_alert: number | null };

/** The warning in force: from its first rating (`since`), its tier held until an hour after `issued`, the rating that
 * holds it, whose chance and reason it carries (lib/warnings.ts has the rule). */
export type Warning = {
  tier: Exclude<Tier, "Low">;
  since: string;
  issued: string;
  until: string;
  p_alert: number | null;
  reason: string | null;
};

export type AreaNow = {
  area: string;
  ts: string;
  tier: Tier; // the warning in force, which the map shows
  rated?: Tier; // the newest rating alone; absent in a /api/now cached before 26 Oct, when tier was the rating
  warning?: Warning | null;
  p_alert: number | null; // the newest rating's
  reason: string | null;
  rain_60min_mm: number | null;
  nearby?: Nearby | null; // absent in a /api/now cached before 23 Oct
};

export type TidePoint = {
  ts: string;
  predicted_m: number;
  next_high_ts: string | null;
  next_high_m: number | null;
  minutes_to_high_tide: number | null;
};

export type Now = {
  tick: string | null;
  model: string | null;
  areas: AreaNow[];
  rain_as_of: string | null;
  fetched_at: string | null;
  tide: { now: TidePoint | null; curve: TidePoint[] };
  stale?: boolean;
};

export type Located = { query: string; place: string; lat: number; lon: number; area: string };

/** How FloodSense did at one warning level since 20 Dec 2025 (ml.scoreboard): the flash floods PUB reported, those a
 * warning was in force at, those it caught (warned 5 minutes to an hour ahead), the mean lead over the warned ones, and
 * the share of its warnings a PUB alert or a flash flood followed. */
export type LevelRecord = {
  flash_floods: number;
  floods_warned: number;
  floods_caught: number;
  floods_caught_share_lo: number | null;
  floods_caught_share_hi: number | null;
  flood_mean_lead_min: number | null;
  warnings: number;
  warnings_followed_by_alert: number | null;
  warnings_followed_by_flood: number | null;
};

export type TrackRecord = {
  model: string;
  version: string;
  trained_at: number;
  medium_p: number | null; // the tiers' cut-offs: chances of a PUB alert within the hour
  high_p: number | null;
  // the storms after the training data ended, as flood-scoreboard last published them; null before it has
  test: { model: string; from: string; to: string; moderate: LevelRecord; elevated: LevelRecord | null } | null;
};

export type StormDay = { day: string; alerts: number; floods: number; areas: number; some_areas: string };
export type Replays = { from: string; days: StormDay[] };

export type Alert = { area: string; ts: string; kind: "warning" | "flash_flood"; location: string };
export type Replay = {
  day: string;
  model: string;
  ticks: string[];
  tiers: Record<string, string>; // one letter per tick: L, M or H
  first_warned: Record<string, string>;
  alerts: Alert[];
  areas_with_features: number;
};

/** Where the agencies' AI/BI scoreboard and Genie space are; null when the app has none configured. */
export type Agencies = {
  dashboard: { embed_url: string; url: string } | null;
  genie: { url: string } | null;
};

export type AskStarted = { conversation_id: string; message_id: string };

/** Genie's answer to one question: `done` once it has finished, whether or not it could answer. */
export type Answer = {
  status: string;
  done: boolean;
  text: string | null;
  sql: string | null;
  description: string | null;
  columns: string[];
  rows: (string | null)[][];
  row_count: number | null;
  suggested: string[];
  error: string | null;
};

/** A sign-up for warnings by email or text, as the app shows it back: the address masked, how the welcome went. */
export type Subscription = {
  id: string;
  channel: "email" | "sms";
  address: string;
  areas: string[];
  places: string[];
  created_at: string;
  welcome: string | null; // "sent", or why not yet; null until flood-live's next run
  welcomed_at: string | null;
};

/** The Telegram bot that warns people wherever they are, by their live location; null when the app has none. */
export type Follow = { bot: string | null; url: string | null };

export type AreaFeature = {
  type: "Feature";
  properties: { area: string; lat: number; lon: number };
  geometry: GeoJSON.Geometry;
};
export type AreaCollection = { type: "FeatureCollection"; features: AreaFeature[] };

export class ApiError extends Error {}

export async function getJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      detail = (await resp.json()).detail ?? detail;
    } catch {
      // not JSON: keep the status line
    }
    throw new ApiError(detail);
  }
  return resp.json() as Promise<T>;
}

export async function postJson<T>(url: string, body: unknown, method = "POST"): Promise<T> {
  const resp = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const d = (await resp.json()).detail;
      detail = typeof d === "string" ? d : detail; // a 422 lists the fields instead
    } catch {
      // not JSON: keep the status line
    }
    throw new ApiError(detail);
  }
  return resp.json() as Promise<T>;
}

/** The map's shapes never change while the app runs. */
export function useAreaShapes() {
  return useSWR<AreaCollection>("/areas.geojson", getJson, {
    revalidateOnFocus: false,
    revalidateIfStale: false,
  });
}

/** A new tick every 5 minutes; checking every minute keeps the map at most a minute behind. */
export function useNow() {
  return useSWR<Now>("/api/now", getJson, { refreshInterval: 60_000 });
}
