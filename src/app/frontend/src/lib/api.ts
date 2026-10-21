import useSWR from "swr";

export type Tier = "Low" | "Medium" | "High";

export type AreaNow = {
  area: string;
  ts: string;
  tier: Tier;
  p_alert: number | null;
  reason: string | null;
  rain_60min_mm: number | null;
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

export type TrackRecord = {
  model: string;
  version: string;
  trained_at: number;
  development: string | null;
  rule: string | null;
  metrics: Record<string, number>;
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
