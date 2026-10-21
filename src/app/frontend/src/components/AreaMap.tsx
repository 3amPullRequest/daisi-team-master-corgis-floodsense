import { geoMercator, geoPath } from "d3-geo";
import { memo, useMemo } from "react";
import type { AreaCollection, Tier } from "../lib/api";

const WIDTH = 800;
const HEIGHT = 470;

export type Marker = { area: string; kind: "warning" | "flash_flood"; recent: boolean };

type Props = {
  shapes: AreaCollection;
  tiers: Record<string, Tier | undefined>;
  selected: string | null;
  onSelect: (area: string) => void;
  markers?: Marker[];
  label: string;
};

/** The projected outlines, computed once per shape set: they don't change between ticks. */
function useProjected(shapes: AreaCollection) {
  return useMemo(() => {
    const projection = geoMercator().fitExtent(
      [
        [8, 8],
        [WIDTH - 8, HEIGHT - 8],
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

function AreaMapInner({ shapes, tiers, selected, onSelect, markers = [], label }: Props) {
  const cells = useProjected(shapes);
  const at = useMemo(() => new Map(cells.map((c) => [c.area, c])), [cells]);
  return (
    <svg className="map" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="group" aria-label={label}>
      {cells.map((c) => {
        const tier = tiers[c.area];
        return (
          <path
            key={c.area}
            d={c.d}
            className="cell"
            data-tier={tier ?? "none"}
            data-selected={c.area === selected || undefined}
            tabIndex={0}
            role="button"
            aria-label={`${c.area}: ${tier ?? "no rating"}`}
            aria-pressed={c.area === selected}
            onClick={() => onSelect(c.area)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(c.area);
              }
            }}
          >
            <title>{`${c.area}: ${tier ?? "no rating"}`}</title>
          </path>
        );
      })}
      {selected && at.get(selected) ? (
        <path d={at.get(selected)!.d} className="cell-outline" aria-hidden="true" />
      ) : null}
      {markers.map((m, i) => {
        const c = at.get(m.area);
        return c ? (
          <circle
            key={`${m.area}-${i}`}
            cx={c.x}
            cy={c.y}
            r={m.kind === "flash_flood" ? 9 : 6}
            className="alert-dot"
            data-kind={m.kind}
            data-recent={m.recent || undefined}
            aria-hidden="true"
          />
        ) : null;
      })}
    </svg>
  );
}

export const AreaMap = memo(AreaMapInner);
