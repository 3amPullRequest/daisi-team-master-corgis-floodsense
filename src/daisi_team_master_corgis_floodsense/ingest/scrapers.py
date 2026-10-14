"""HTML sources: the PUB Flood Alerts Telegram channel preview (t.me/s/...) and NEA's tide-timings page.

The parsers only lift structure out of the markup (post id, timestamp, text; table cells). Interpreting the text
(alert type, location, issue time) is silver's job, so the raw post HTML is kept alongside.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser


class _TelegramPageParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self._source = source
        self._line_offsets = [0] + [m.end() for m in re.finditer("\n", source)]
        self.posts: list[dict] = []
        self.has_newer = False  # the page links to newer posts
        self._post: dict | None = None
        self._text_depth = 0  # >0 while inside the post's text div
        self._text_start = 0
        self._text_parts: list[str] = []
        self._span = ""  # innermost tracked element: "views", "meta" or "owner"

    def _offset(self) -> int:
        line, col = self.getpos()
        return self._line_offsets[line - 1] + col

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if tag == "div" and "tgme_widget_message" in classes and attr.get("data-post"):
            channel, _, message_id = attr["data-post"].rpartition("/")
            self._post = {
                "channel": channel,
                "channel_title": None,
                "message_id": int(message_id),
                "posted_at": None,
                "text": "",
                "html": "",
                "views": None,
                "edited": False,
                "service": "service_message" in classes,
            }
            self.posts.append(self._post)
            return
        if tag == "a" and "tme_messages_more" in classes:
            self.has_newer |= bool(attr.get("data-after"))
            return
        if self._post is None:
            return
        if self._text_depth:
            if tag == "div":
                self._text_depth += 1
            elif tag == "br":
                self._text_parts.append("\n")
        elif tag == "div" and "js-message_text" in classes and not self._post["html"]:
            self._text_depth = 1
            self._text_start = self._offset() + len(self.get_starttag_text() or "")
            self._text_parts = []
        elif tag == "time" and attr.get("datetime") and self._post["posted_at"] is None:
            self._post["posted_at"] = attr["datetime"]
        elif tag == "span" and "tgme_widget_message_views" in classes:
            self._span = "views"
        elif tag == "span" and "tgme_widget_message_meta" in classes:
            self._span = "meta"
        elif (
            tag == "a"
            and "tgme_widget_message_owner_name" in classes
            and self._post["channel_title"] is None
        ):
            self._span = "owner"
            self._post["channel_title"] = ""

    def handle_endtag(self, tag: str) -> None:
        if self._text_depth and tag == "div":
            self._text_depth -= 1
            if not self._text_depth:
                self._post["html"] = self._source[self._text_start : self._offset()]
                self._post["text"] = "".join(self._text_parts).strip()
        elif tag == "span" and self._span == "views":
            self._span = ""
        elif tag == "a" and self._span == "meta":
            self._span = ""  # "edited", if present, comes before the date link inside the meta span
        elif tag == "a" and self._span == "owner":
            self._span = ""
            self._post["channel_title"] = self._post["channel_title"].strip()

    def handle_data(self, data: str) -> None:
        if self._text_depth:
            self._text_parts.append(data)
        elif self._post is not None and self._span == "views":
            self._post["views"] = data.strip()
        elif self._post is not None and self._span == "meta" and "edited" in data:
            self._post["edited"] = True
        elif self._post is not None and self._span == "owner":
            self._post["channel_title"] += data


def parse_telegram_page(html: str) -> list[dict]:
    """Posts on one t.me/s/<channel> page (or one embedded post), oldest first, as dicts with message_id, posted_at
    (UTC), text, html, and channel_title, the channel's name when the page was fetched."""
    return parse_telegram_listing(html)[0]


def parse_telegram_listing(html: str) -> tuple[list[dict], bool]:
    """parse_telegram_page, plus whether the page links to newer posts: a channel page without that link ends at the
    channel's newest post."""
    parser = _TelegramPageParser(html)
    parser.feed(html)
    parser.close()
    return parser.posts, parser.has_newer


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._in_table = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table += 1
            if self._in_table == 1:
                self.tables.append([])
        elif self._in_table == 1 and tag == "tr":
            self._row = []
        elif self._row is not None and tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "table":
            self._in_table -= 1
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_MONTH_LABEL = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (20\d\d)\b"
)


def parse_nea_tides(html: str) -> list[dict]:
    """Tide rows from NEA's page: one table per month, a day cell spanning that day's 3-4 high/low waters.

    Values stay as printed: time 'HHMM' (SGT), height in metres, 'H' or 'L'. `month_label` ('Oct 2026') comes from
    the page's month tabs, which appear in the same order as the tables."""
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    first_table = html.find("<table")
    labels = list(
        dict.fromkeys(
            m.group(0)
            for m in _MONTH_LABEL.finditer(re.sub(r"<[^>]+>", " ", html[:first_table]))
        )
    )
    rows: list[dict] = []
    for index, table in enumerate(parser.tables):
        month, day = None, None
        for cells in table:
            if len(cells) == 2 and cells[1].lower() == "tide":
                month = cells[0]
            elif len(cells) == 4 and cells[0].isdigit():
                day, time_, height, hl = cells
                rows.append(_tide_row(index, labels, month, day, time_, height, hl))
            elif len(cells) == 3 and day is not None and cells[0].isdigit():
                rows.append(_tide_row(index, labels, month, day, *cells))
    return rows


def _tide_row(index, labels, month, day, time_, height, hl) -> dict:
    return {
        "table_index": index,
        "month_label": labels[index] if index < len(labels) else None,
        "month": month,
        "day": day,
        "time": time_,
        "height_m": height,
        "hl": hl,
    }
