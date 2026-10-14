from pathlib import Path

from daisi_team_master_corgis_floodsense.ingest.scrapers import (
    parse_nea_tides,
    parse_telegram_listing,
    parse_telegram_page,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_telegram_posts():
    posts = parse_telegram_page(
        (FIXTURES / "telegram_page.html").read_text(encoding="utf-8")
    )
    assert [p["message_id"] for p in posts] == [1, 4, 1630, 1631]
    service, edited, _, alert = posts
    assert service["service"] and service["text"] == "Channel created"
    assert edited["edited"] and edited["views"] == "102K"
    assert alert["posted_at"] == "2026-10-01T06:20:40+00:00"
    assert alert["text"].startswith("[Risk of Flash Floods]\n\nDue to heavy rain")
    assert alert["text"].endswith(
        "Jalan Boonlay (from International Road to Enterprise Road) [14:14hours]"
    )
    assert alert["html"].startswith("[Risk of Flash Floods]<br/><br/>")
    assert not alert["edited"] and not alert["service"]
    assert {p["channel_title"] for p in posts} == {"PUB Flood Alerts"}


def test_telegram_listing_says_whether_newer_posts_exist():
    html = (FIXTURES / "telegram_page.html").read_text(encoding="utf-8")

    def link(direction: str) -> str:
        return (
            f'<a href="/s/pubfloodalerts?{direction}=1631" class="tme_messages_more js-messages_more" '
            f'data-{direction}="1631"></a></section>'
        )

    assert parse_telegram_listing(html)[1] is False
    assert (
        parse_telegram_listing(html.replace("</section>", link("before")))[1] is False
    )
    assert parse_telegram_listing(html.replace("</section>", link("after")))[1] is True


def test_nea_tide_rows():
    rows = parse_nea_tides((FIXTURES / "nea_tides.html").read_text(encoding="utf-8"))
    assert len(rows) == 16  # 2 months x 2 days x 4 tides
    assert rows[0] == {
        "table_index": 0,
        "month_label": "Oct 2026",
        "month": "Oct",
        "day": "1",
        "time": "0156",
        "height_m": "2.8",
        "hl": "H",
    }
    # rows after the first of a day inherit its day cell (rowspan)
    assert [(r["day"], r["time"], r["hl"]) for r in rows[1:4]] == [
        ("1", "0745", "L"),
        ("1", "1320", "H"),
        ("1", "2005", "L"),
    ]
    assert rows[8]["month_label"] == "Nov 2026" and rows[8]["day"] == "1"
