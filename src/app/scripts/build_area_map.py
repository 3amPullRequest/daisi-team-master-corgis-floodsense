"""Build the app's map of NEA's 47 forecast areas: a Voronoi cell around each area's label point (an alert belongs to
the area whose point is nearest it, so the cell is exactly where an alert counts for that area), clipped to Singapore's
land (the union of the URA Master Plan 2019 land-use polygons) and simplified for the browser.

Run once, when the areas change; the output is checked in as frontend/public/areas.geojson.

    databricks experimental aitools tools query "SELECT area, lat, lon FROM flood_dev.dev_<you>_silver.forecast_areas" \
        -p flood_dev -o json > areas.json
    databricks fs cp dbfs:/Volumes/flood_dev/dev_<you>_bronze/landing/static_mp2019_land_use/version=<v>/<id>.geojson.gz \
        mp2019.geojson.gz -p flood_dev
    python src/app/scripts/build_area_map.py areas.json mp2019.geojson.gz src/app/frontend/public/areas.geojson

Needs shapely (not an app dependency).
"""

import gzip
import json
import sys
from pathlib import Path

from shapely import force_2d, set_precision, voronoi_polygons
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import unary_union

SIMPLIFY_DEG = 0.0004  # about 45 m: invisible at city scale, a few kB per area
MIN_ISLAND_KM2 = 0.3  # keep Sentosa, Ubin, Tekong and Jurong Island; drop reef specks


def land(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        features = json.load(f)["features"]
    # a small buffer closes the gaps between parcels (roads aren't land-use polygons in every release)
    parts = [
        force_2d(shape(f["geometry"])).buffer(0.0002)
        for f in features
        if f.get("geometry")
    ]
    whole = unary_union(parts).buffer(-0.0002)
    polygons = getattr(whole, "geoms", [whole])
    km2 = 111.32**2  # square degrees to km2, near enough at 1.3 N
    return unary_union([p for p in polygons if p.area * km2 >= MIN_ISLAND_KM2])


def main(areas_path: str, land_path: str, out_path: str) -> None:
    text = Path(areas_path).read_text(encoding="utf-8")
    rows = json.loads(text[text.index("[") :])
    points = {r["area"]: Point(float(r["lon"]), float(r["lat"])) for r in rows}
    coast = land(land_path)
    cells = voronoi_polygons(
        MultiPoint(list(points.values())), extend_to=coast.envelope.buffer(0.1)
    )
    features = []
    for area, point in sorted(points.items()):
        cell = next(c for c in cells.geoms if c.contains(point))
        clipped = cell.intersection(coast).simplify(
            SIMPLIFY_DEG, preserve_topology=True
        )
        clipped = set_precision(clipped, 1e-5)
        features.append(
            {
                "type": "Feature",
                "properties": {"area": area, "lat": point.y, "lon": point.x},
                "geometry": mapping(clipped),
            }
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": features},
            f,
            separators=(",", ":"),
        )
    print(f"{len(features)} areas -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
