import { geoMercator, geoPath } from "d3-geo";
import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { AreaCollection, Tier } from "../lib/api";
import { TIER_LABEL, TIER_SHORT } from "../lib/content";

const WIDTH = 800;
const HEIGHT = 470;
const LABEL_PX = 13; // label sizes on screen, whatever width the map is drawn at
const LANDMARK_PX = 11.5;
// names everyone knows, in grey, so the map reads as Singapore before any area is warned
const LANDMARKS = ["Woodlands", "Jurong West", "Ang Mo Kio", "Tampines", "Changi", "City", "Bukit Timah"];

export type Marker = { area: string; kind: "warning" | "flash_flood"; recent: boolean };

type Props = {
  shapes: AreaCollection;
  tiers: Record<string, Tier | undefined>;
  selected: string | null;
  onSelect: (area: string) => void;
  markers?: Marker[];
  /** areas to name on the map, most important first; a name that would overlap a more important one is left out */
  labels?: string[];
  /** areas that have or will get an alert dot, whose names keep the dot clear (default: the markers' areas) */
  dots?: string[];
  /** a second line for the hover card, such as the chance in words */
  detail?: (area: string) => string | null;
  label: string;
};

/** The projected outlines and label points, computed once per shape set: they don't change between ticks. */
function useProjected(shapes: AreaCollection) {
  return useMemo(() => {
    const projection = geoMercator().fitExtent(
      [
        [16, 16],
        [WIDTH - 16, HEIGHT - 16],
      ],
      shapes as unknown as GeoJSON.FeatureCollection,
    );
    const path = geoPath(projection);
    return shapes.features.map((f) => {
      const [x, y] = projection([f.properties.lon, f.properties.lat]) ?? [0, 0];
      return { area: f.properties.area, d: path(f as unknown as GeoJSON.Feature) ?? "", x, y };
    });
  }, [shapes]);
}

type Cell = ReturnType<typeof useProjected>[number];
type Placed = { area: string; x: number; y: number; size: number; strong: boolean };

/** Greedy placement in priority order: a name tries its area's point, then just above it, then just below, and is
 * left out if all three overlap a name already placed. An area with an alert dot keeps the dot clear. Sizes are in
 * map units, so a name is the same size on a phone as on a laptop. */
function placeLabels(wanted: string[], at: Map<string, Cell>, scale: number, dotted: Set<string>): Placed[] {
  const boxes: number[][] = [];
  const out: Placed[] = [];
  const seen = new Set<string>();
  const pad = 3 / scale;
  for (const [area, strong] of [...wanted.map((a) => [a, true] as const), ...LANDMARKS.map((a) => [a, false] as const)]) {
    const c = at.get(area);
    if (!c || seen.has(area)) continue;
    seen.add(area);
    const size = (strong ? LABEL_PX : LANDMARK_PX) / scale;
    const w = area.length * size * (strong ? 0.6 : 0.55);
    const h = size * 1.2;
    const x = Math.min(Math.max(c.x, w / 2 + pad), WIDTH - w / 2 - pad);
    const shifts = dotted.has(area) ? [-1, 1] : strong ? [0, -1, 1] : [0];
    for (const shift of shifts) {
      const y = c.y + shift * h;
      const box = [x - w / 2 - pad, y - h / 2 - pad, x + w / 2 + pad, y + h / 2 + pad];
      if (boxes.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
      boxes.push(box);
      out.push({ area, x, y, size, strong });
      break;
    }
  }
  return out;
}

function AreaMapInner({ shapes, tiers, selected, onSelect, markers = [], labels = [], dots, detail, label }: Props) {
  const cells = useProjected(shapes);
  const at = useMemo(() => new Map(cells.map((c) => [c.area, c])), [cells]);
  const wrap = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1); // screen pixels per map unit
  const [hover, setHover] = useState<{ area: string; x: number; y: number } | null>(null);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => setScale(el.clientWidth / WIDTH || 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const dotted = useMemo(() => new Set(dots ?? markers.map((m) => m.area)), [dots, markers]);
  const placed = useMemo(() => placeLabels(labels, at, scale, dotted), [labels, at, scale, dotted]);
  const hovered = hover ? tiers[hover.area] : undefined;
  const hoverDetail = hover && detail ? detail(hover.area) : null;

  return (
    <div className="map-wrap" ref={wrap} onPointerLeave={() => setHover(null)}>
      <svg className="map" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="group" aria-label={label}>
        <defs>
          <pattern id="map-none" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="8" height="8" className="hatch-ground" />
            <line x1="0" y1="0" x2="0" y2="8" className="hatch-line" />
          </pattern>
        </defs>
        <rect width={WIDTH} height={HEIGHT} rx="10" className="map-sea" />
        {cells.map((c) => {
          const tier = tiers[c.area];
          const name = `${c.area}: ${tier ?? "no rating"}`;
          return (
            <path
              key={c.area}
              d={c.d}
              className="cell"
              data-tier={tier ?? "none"}
              data-selected={c.area === selected || undefined}
              tabIndex={0}
              role="button"
              aria-label={name}
              aria-pressed={c.area === selected}
              onClick={() => onSelect(c.area)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(c.area);
                }
              }}
              onPointerMove={(e) => {
                if (e.pointerType !== "mouse" || !wrap.current) return;
                const box = wrap.current.getBoundingClientRect();
                setHover({ area: c.area, x: e.clientX - box.left, y: e.clientY - box.top });
              }}
              onFocus={() => setHover({ area: c.area, x: c.x * scale, y: c.y * scale })}
              onBlur={() => setHover(null)}
            />
          );
        })}
        {selected && at.get(selected) ? <path d={at.get(selected)!.d} className="cell-outline" aria-hidden="true" /> : null}
        {markers.map((m, i) => {
          const c = at.get(m.area);
          return c ? (
            <circle
              key={`${m.area}-${i}`}
              cx={c.x}
              cy={c.y}
              r={(m.kind === "flash_flood" ? 7 : 5.5) / Math.max(scale, 0.6)}
              className="alert-dot"
              data-kind={m.kind}
              data-recent={m.recent || undefined}
              style={{ strokeWidth: 2.5 / Math.max(scale, 0.6) }}
              aria-hidden="true"
            />
          ) : null;
        })}
        <g aria-hidden="true">
          {placed.map((p) => (
            <text
              key={p.area}
              x={p.x}
              y={p.y}
              className={p.strong ? "map-label" : "map-label map-label-quiet"}
              style={{ fontSize: p.size, strokeWidth: 4 / scale }}
            >
              {p.area}
            </text>
          ))}
        </g>
      </svg>
      {hover ? (
        <div
          className="map-tip"
          data-flip={hover.x > (wrap.current?.clientWidth ?? 0) - 200 || undefined}
          style={{ left: hover.x, top: hover.y }}
          aria-hidden="true"
        >
          <strong>{hover.area}</strong>
          {hovered ? (
            <span>
              <i className="swatch" data-tier={hovered} /> {TIER_LABEL[hovered]}: {TIER_SHORT[hovered]}
            </span>
          ) : null}
          {hoverDetail ? <span>{hoverDetail}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

export const AreaMap = memo(AreaMapInner);

/** The tiers with what each means, so the colours never have to be read alone. */
export function MapLegend() {
  return (
    <ul className="legend" aria-label="Legend">
      {(["Low", "Medium", "High"] as const).map((t) => (
        <li key={t}>
          <i className="swatch" data-tier={t} />
          <strong>{TIER_LABEL[t]}</strong>: {TIER_SHORT[t]}
        </li>
      ))}
    </ul>
  );
}
