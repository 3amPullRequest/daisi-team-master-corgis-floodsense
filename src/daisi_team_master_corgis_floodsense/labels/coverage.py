"""How well the alert patterns cover the landed Telegram posts: posts per format by year, and every post that needs a
look (no format matched, or one matched with a problem). Run it after adding a pattern:

    uv run python -m daisi_team_master_corgis_floodsense.labels.coverage .landing-dev/pub_telegram_posts

The folder is a local copy of landing/pub_telegram_posts (`databricks fs cp -r dbfs:/Volumes/.../pub_telegram_posts
.landing-dev/pub_telegram_posts -p flood_dev`). Silver applies the same rules, so the counts match
silver.telegram_posts.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from . import alert_text as T


def landed_posts(folder: Path) -> Iterable[dict]:
    for path in sorted(folder.rglob("*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    yield json.loads(line)


def latest_versions(posts: Iterable[dict]) -> dict[int, dict]:
    """One version per post id: the latest seen. A post recovered from the web archive was seen when it was captured,
    not when it was fetched."""
    latest: dict[int, dict] = {}
    for post in posts:
        seen = post.get("captured_at") or post["fetched_at"]
        current = latest.get(post["message_id"])
        if current is None or seen > (
            current.get("captured_at") or current["fetched_at"]
        ):
            latest[post["message_id"]] = post
    return latest


def report(posts: dict[int, dict]) -> dict:
    by_format: Counter[tuple[str, str]] = Counter()
    first_seen: dict[str, str] = {}
    review = []
    for message_id, post in sorted(posts.items()):
        parsed = T.parse(post.get("text"), service=bool(post.get("service")))
        posted = (
            datetime.fromisoformat(post["posted_at"]).astimezone(T.SGT)
            if post.get("posted_at")
            else None
        )
        year = str(posted.year) if posted else "????"
        fmt = parsed.pattern or f"({parsed.category})"
        by_format[(fmt, year)] += 1
        if posted:
            first_seen[fmt] = min(
                first_seen.get(fmt, "9999"), posted.date().isoformat()
            )
        if parsed.category == T.UNMATCHED or parsed.problem:
            review.append(
                {
                    "message_id": message_id,
                    "posted_at": posted.isoformat() if posted else None,
                    "category": parsed.category,
                    "problem": parsed.problem,
                    "hint": parsed.hint,
                    "text": T.normalize(post.get("text"))[:300],
                }
            )
    years = sorted({year for _, year in by_format})
    formats = sorted(
        {fmt for fmt, _ in by_format}, key=lambda f: first_seen.get(f, "9999")
    )
    return {
        "posts": len(posts),
        "years": years,
        "table": [
            {
                "format": fmt,
                "first_seen": first_seen.get(fmt),
                **{y: by_format[(fmt, y)] for y in years},
            }
            for fmt in formats
        ],
        "review": review,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "folder", type=Path, help="a local copy of landing/pub_telegram_posts"
    )
    args = parser.parse_args(argv)
    result = report(latest_versions(landed_posts(args.folder)))
    print(f"{result['posts']} posts")
    header = ["format", "first_seen", *result["years"]]
    print("  ".join(f"{h:>22}" if i == 0 else f"{h:>10}" for i, h in enumerate(header)))
    for row in result["table"]:
        cells = [
            row["format"],
            row["first_seen"] or "",
            *(row[y] for y in result["years"]),
        ]
        print(
            "  ".join(
                f"{c:>22}" if i == 0 else f"{c!s:>10}" for i, c in enumerate(cells)
            )
        )
    print(f"\n{len(result['review'])} posts to review:")
    for item in result["review"]:
        print(f"- {item['message_id']} {item['posted_at']} {item['category']}", end="")
        print(f" ({item['problem'] or 'hint: ' + str(item['hint'])})")
        print("    " + item["text"].replace("\n", " | ")[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
