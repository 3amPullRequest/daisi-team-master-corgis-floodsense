const SGT = "Asia/Singapore";

const timeFmt = new Intl.DateTimeFormat("en-SG", { timeZone: SGT, hour: "numeric", minute: "2-digit" });
const dayFmt = new Intl.DateTimeFormat("en-SG", { timeZone: SGT, weekday: "short", day: "numeric", month: "short", year: "numeric" });
const shortDayFmt = new Intl.DateTimeFormat("en-SG", { timeZone: SGT, day: "numeric", month: "short" });

export const clock = (iso: string | null | undefined) => (iso ? timeFmt.format(new Date(iso)) : "–");
export const longDay = (iso: string) => dayFmt.format(new Date(iso));
export const shortDay = (iso: string) => shortDayFmt.format(new Date(iso));

export function minutesSince(iso: string | null | undefined, now = Date.now()): number | null {
  return iso ? Math.round((now - new Date(iso).getTime()) / 60_000) : null;
}

export function duration(minutes: number): string {
  const m = Math.round(Math.abs(minutes));
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rest = m % 60;
  return rest ? `${h} h ${rest} min` : `${h} h`;
}

const NICE = [2, 3, 4, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000];

/** The model's probability as people say odds: "about 1 in 20". Tiny ones are bucketed rather than given false
 * precision. */
export function chanceInWords(p: number | null): string {
  if (p == null || Number.isNaN(p)) return "Not rated (no rain gauge reported nearby)";
  if (p < 0.001) return "Less than 1 in 1,000";
  if (p >= 0.5) return `About ${Math.round(p * 10)} in 10`;
  const n = 1 / p;
  const nearest = NICE.reduce((a, b) => (Math.abs(b - n) < Math.abs(a - n) ? b : a));
  return `About 1 in ${nearest.toLocaleString("en-SG")}`;
}

export const percent = (x: number | null | undefined, digits = 0) =>
  x == null ? "–" : `${(x * 100).toFixed(digits)}%`;
