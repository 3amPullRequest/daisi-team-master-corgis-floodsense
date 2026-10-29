// Every Moderate or Elevated rating is a warning for the next hour, as the model was trained: a PUB alert within the
// hour after it. So an area stays warned until an hour after the last rating that warned it, and ratings at most an
// hour apart are one warning. Elevated has priority: while any Elevated rating is under an hour old the area shows
// Elevated, so Moderate can follow only an hour after the last Elevated rating. The live map (backend /api/now), the
// Telegram posts (ml/broadcast.py) and the agencies' scoreboard count warnings the same way.

export const WARNING_TICKS = 12; // an hour of 5-minute ticks
const HOUR_MS = 3_600_000;

/** The warning in force at each tick, one letter per tick (L, M or H), from the ratings in the same form. */
export function inForce(codes: string): string {
  let lastHigh = -Infinity;
  let lastWarned = -Infinity;
  let out = "";
  for (let k = 0; k < codes.length; k++) {
    if (codes[k] === "H") lastHigh = k;
    if (codes[k] === "H" || codes[k] === "M") lastWarned = k;
    out += k - lastHigh < WARNING_TICKS ? "H" : k - lastWarned < WARNING_TICKS ? "M" : "L";
  }
  return out;
}

/** When the warning in force at `at` began, or null if none was: the newest Moderate or Elevated rating in the hour
 * before `at` (strictly before it, as PUB's alert must come after the warning), then back through earlier ratings at
 * most an hour apart. `times` are the ticks in ms, `codes` the ratings. */
export function warnedSince(times: number[], codes: string, at: number): number | null {
  let i = -1;
  for (let k = 0; k < times.length && times[k] < at; k++) if (codes[k] !== "L" && times[k] >= at - HOUR_MS) i = k;
  if (i < 0) return null;
  let start = i;
  for (let k = i - 1; k >= 0 && times[start] - times[k] <= HOUR_MS; k--) if (codes[k] !== "L") start = k;
  return times[start];
}
