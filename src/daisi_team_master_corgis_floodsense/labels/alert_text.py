"""PUB alert text: what kind of alert a PUB Flood Alerts post is, and the locations and times it names.

PUB's Telegram posts and the `description` of its flood-alerts API carry the same text, so silver reads both with
`parse`, and a fetcher can use it to count posts that no pattern knows.

PUB has reworded its posts several times (DEVELOPER_NOTES "Things that bit us"), so every format it has used has its
own pattern, named after the format's marker. A post matches one of them or it is `unmatched`: nothing is guessed. An
unmatched post never becomes a label. Silver lists it for review (silver.telegram_review) with a hint of what it looks
like, so a new format shows up the first time PUB uses it.

A post is read in four steps:

1. Classify. `parse` normalises the text and tries FORMATS in order. A Format has a headline regex that must match at
   the start of the text, and a reader for the rest. The first headline that matches decides the category and the
   pattern. A post that matches none is `unmatched`, with a `hint` (from loose keyword clues) of the category it
   resembles, to steer whoever reviews it. Service messages and empty posts are `not_relevant`.
2. Read the places. The format's reader lifts out the locations. A list format (the 2021-2023 warnings and the older
   flash-flood list) has one bullet line per location, each ending in its own stamp ("• Harper Rd [10:10 hours]"),
   and `_segments` splits the text at the stamps so that each stamp ends the location before it. Other formats name
   one place in a sentence, and the reader cuts it from the lead phrase to the end of that sentence. NEA's heavy-rain
   posts have regions, a valid window and an issue stamp in place of locations. The reader never guesses: with no
   location it records a `problem` (NO_LOCATION), and with a location that has no time, another (NO_STAMP), and
   silver sends a post with a problem to review.
3. Date the stamps. A stamp is only 'HH:MM' in SGT. `stamp_time` turns it into a moment from the posting time: the day
   of posting, or the day before when that would be later than the post by more than CLOCK_SKEW, as when a post just
   after midnight is about an alert just before it. `window_times` does the same for a heavy-rain window, placing it so
   that it starts within 12 hours of being issued and ends after it starts.
4. Hand it to Spark. `parse_row` is `parse` plus those conversions as a plain dict matching ROW_SCHEMA, which silver
   wraps as a UDF. This module uses only the standard library and the contract, so it can travel to executors by value
   and the fetchers and tests read posts exactly as silver does.

To support a new format, add a Format to FORMATS, in the order of the others, with a real post as a test fixture.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..ml.contract import FLASH_FLOOD, WARNING

SGT = timezone(timedelta(hours=8), "SGT")  # no daylight saving since 1982

FLOOD_SUBSIDED = "flood_subsided"
HEAVY_RAIN = "heavy_rain"  # NEA: heavy rain expected over parts of Singapore in the next hour or so
HEAVY_RAIN_EASED = "heavy_rain_eased"
NOT_RELEVANT = "not_relevant"  # channel service messages, posts without text, and announcements once reviewed
UNMATCHED = "unmatched"  # no pattern fits: a new format, or a post that isn't an alert
CATEGORIES = (
    WARNING,
    FLASH_FLOOD,
    FLOOD_SUBSIDED,
    HEAVY_RAIN,
    HEAVY_RAIN_EASED,
    NOT_RELEVANT,
    UNMATCHED,
)

# Problems with a post whose format matched. Each sends the post to review.
NO_LOCATION = "no location"  # nothing to place, so the post can't become an alert
NO_STAMP = (
    "no stamp"  # a location without a time: its alert is timed at the posting instead
)

# A stamp can run a few minutes ahead of Telegram's clock. Later than this after the posting, it was stamped the day
# before: a post just after midnight about an alert just before it.
CLOCK_SKEW = timedelta(minutes=10)

# A time PUB or NEA stamped on a post, in every spelling seen so far: [14:13 hours], [14:13hours], [14:13hrs],
# [14:13], [1617 hours], [Issued 14:13 hours], (Issued 11:00 hrs), Issued 1510 hours, [Issued by NEA, 15: 30 hours],
# [Issued by NEA, [16:11 hours]. It needs an opening bracket or "Issued", so a bus stop number isn't read as a time.
_CLOCK = r"(?P<hour>[01]?\d|2[0-3])\s?:?\s?(?P<minute>[0-5]\d)(?!\d)"
_STAMP = re.compile(
    r"(?:[\[(]\s*(?:Issued\s*)?(?:by\s+NEA\s*,\s*)?\[?\s*|\bIssued\s+(?:at\s+)?)"
    + _CLOCK
    + r"(?:\s*h(?:ou)?rs?\b)*\s*[\])]?",
    re.IGNORECASE,
)
_BULLET = re.compile(r"\s*(?:[•●▪◦‣·\-*]|🔹|🔸|▪️)\s*")
_HASHTAG = re.compile(r"#\w+")


@dataclass(frozen=True)
class Located:
    """A place a post names, and the time PUB stamped against it ('HH:MM', SGT), as written."""

    location: str
    stamp: str | None = None


@dataclass(frozen=True)
class Parsed:
    category: str
    pattern: str | None = None  # the format that matched, e.g. "warning_bracket"
    locations: tuple[Located, ...] = ()  # alerts and subsided posts
    problem: str | None = None
    hint: str | None = None  # unmatched posts: the category the text resembles, if any
    regions: tuple[str, ...] = ()  # heavy rain: "northern", "central", ... or "many"
    valid: tuple[str, str] | None = None  # heavy rain: expected from and to, 'HH:MM'
    issued: str | None = None  # heavy rain and eased: NEA's issue stamp, 'HH:MM'

    @property
    def is_alert(self) -> bool:
        return self.category in (WARNING, FLASH_FLOOD)


@dataclass(frozen=True)
class Format:
    """One way PUB has written one kind of post: a headline at the start of the text and a reader for the rest."""

    pattern: str
    category: str
    headline: re.Pattern[str]
    read: Callable[[str], Parsed]


def _headline(regex: str) -> re.Pattern[str]:
    return re.compile(regex, re.IGNORECASE)


def _clean(text: str) -> str:
    text = _HASHTAG.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .,;:-")


def _stamp(match: re.Match[str]) -> str:
    return f"{int(match['hour']):02d}:{match['minute']}"


_BOILERPLATE = re.compile(
    r"(?:Due to\b|Please\b|PUB officers\b|Risk of\b|#)", re.IGNORECASE
)


def _segments(text: str) -> list[Located]:
    """Locations, each ended by its stamp: 'A [10:10 hours] B [10:12 hours]' -> A at 10:10, B at 10:12. A stamp with
    no text before it adds nothing (id 89: '[19:15 hours [19:15 hours]'), and text after the last stamp is one more
    location without a stamp, unless it is one of the format's fixed sentences."""
    located, start = [], 0
    for match in _STAMP.finditer(text):
        located.append(Located(_clean(text[start : match.start()]), _stamp(match)))
        start = match.end()
    tail = _clean(text[start:])
    if tail and not _BOILERPLATE.match(tail):
        located.append(Located(tail))
    return [loc for loc in located if loc.location]


def _bullets(body: str) -> list[Located]:
    """The locations on bullet lines: '• Harper Rd [10:10 hours]', '🔹Siang Kuang Ave [18:58 hours]'."""
    located = []
    for line in body.splitlines():
        bullet = _BULLET.match(line)
        if bullet:
            located += _segments(line[bullet.end() :])
    return located


def _listed(body: str) -> list[Located]:
    """The locations of a list format: its bullet lines, or, when PUB left the bullets out, the lines below the
    headline that aren't the format's fixed sentences (id 424: 'KPE (towards ECP) near Buangkok East Drive
    [10:11 hours].')."""
    located = _bullets(body)
    if located:
        return located
    lines = [line.strip() for line in body.splitlines()[1:]]
    return _segments("\n".join(line for line in lines if not _BOILERPLATE.match(line)))


def _alert(category: str, pattern: str, located: list[Located]) -> Parsed:
    located = [loc for loc in located if loc.location]
    if not located:
        return Parsed(category, pattern, problem=NO_LOCATION)
    no_stamp = any(loc.stamp is None for loc in located)
    return Parsed(
        category, pattern, tuple(located), problem=NO_STAMP if no_stamp else None
    )


def _warning(pattern: str) -> Callable[[str], Parsed]:
    """Bullet lines (2021-2023), or what follows '... for the next 1 hour:' (Nov 2023 on: one location)."""

    def read(body: str) -> Parsed:
        rest = re.search(
            r"\bhours?\s*:\s*(?P<rest>.+)", body, re.IGNORECASE | re.DOTALL
        )
        if rest and not _bullets(body):
            return _alert(WARNING, pattern, _segments(rest["rest"]))
        return _alert(WARNING, pattern, _listed(body))

    return read


def _flood_list(body: str) -> Parsed:
    return _alert(FLASH_FLOOD, "flood_siren", _listed(body))


# "... at <location>. Please avoid the area. ... Issued 1510 hours." The location ends at the first full stop that a
# known next sentence follows, else at the first full stop; a full stop inside a name ("St. John's") is rare.
_NEXT_SENTENCE = r"\.\s*(?=Please\b|PUB\b|Issued\b|Roads?\b|Traffic\b|Motorists\b|\[|$)"


def _one_place(category: str, pattern: str, place: str) -> Callable[[str], Parsed]:
    """Posts about one place: `place` matches up to it, then the location runs to the end of its sentence."""
    lead = re.compile(place, re.IGNORECASE)

    def read(body: str) -> Parsed:
        start = lead.search(body)
        if not start:
            return Parsed(category, pattern, problem=NO_LOCATION)
        rest = body[start.end() :]
        end = re.search(_NEXT_SENTENCE, rest) or re.search(r"\.(?=\s|$)", rest)
        location = rest[: end.start()] if end else rest
        stamp = _STAMP.search(rest)
        if stamp and stamp.start() < len(location):
            location = location[: stamp.start()]  # no full stop before the stamp
        return _alert(
            category,
            pattern,
            [Located(_clean(location), _stamp(stamp) if stamp else None)],
        )

    return read


def _subsided_arrow(body: str) -> Parsed:
    """'🔽 Flash flood at <location> has subsided. [14:25 hours]'"""
    match = re.search(
        r"Flash flood at\s+(?P<loc>.+?)\s+has subsided", body, re.IGNORECASE | re.DOTALL
    )
    if not match:
        return Parsed(FLOOD_SUBSIDED, "subsided_arrow", problem=NO_LOCATION)
    stamp = _STAMP.search(body, match.end())
    located = Located(_clean(match["loc"]), _stamp(stamp) if stamp else None)
    return _alert(FLOOD_SUBSIDED, "subsided_arrow", [located])


_HEAVY_RAIN = re.compile(
    r"Heavy rain expected over\s+(?P<regions>[a-z ,&]+?)\s+(?:areas?\s+)?of Singapore\s+"
    r"from\s+(?P<start>\d{1,2}\s?[:.]?\s?\d{2})\s*(?:h(?:ou)?rs?)?\s+to\s+(?P<end>\d{1,2}\s?[:.]?\s?\d{2})",
    re.IGNORECASE,
)


def _hhmm(text: str) -> str | None:
    """'14:00', '1400', '14.00' -> '14:00'; NEA writes the midnight that ends a window as 24:00."""
    digits = re.sub(r"\D", "", text)
    hour, minute = int(digits[:-2]), int(digits[-2:])
    ok = minute < 60 and (hour < 24 or (hour, minute) == (24, 0))
    return f"{hour:02d}:{minute:02d}" if ok else None


def _heavy_rain(body: str) -> Parsed:
    """'⛈ Heavy rain expected over northern and central areas of Singapore from 14:00 to 15:00 hours.
    [Issued by NEA, 13:23 hours]'"""
    match = _HEAVY_RAIN.search(body)
    start, end = (_hhmm(match["start"]), _hhmm(match["end"])) if match else (None, None)
    if not (start and end):
        return Parsed(HEAVY_RAIN, "heavy_rain", problem="no time window")
    regions = tuple(
        r.strip().lower()
        for r in re.split(r",|&|\band\b", match["regions"])
        if r.strip()
    )
    issued = _STAMP.search(body, match.end())
    return Parsed(
        HEAVY_RAIN,
        "heavy_rain",
        regions=regions,
        valid=(start, end),
        issued=_stamp(issued) if issued else None,
        problem=None if issued else NO_STAMP,
    )


def _eased(body: str) -> Parsed:
    """'The heavy rain has eased. [Issued by NEA, 14:41 hours]'"""
    issued = _STAMP.search(body)
    return Parsed(
        HEAVY_RAIN_EASED,
        "heavy_rain_eased",
        issued=_stamp(issued) if issued else None,
        problem=None if issued else NO_STAMP,
    )


# Every format PUB has used, oldest first within each kind. First seen (SGT) in the archive:
FORMATS: tuple[Format, ...] = (
    # Nov 2021: "RISING WATERS / POTENTIAL FLASH FLOODS", bullets with "(Issued 11:00 hrs)"
    Format(
        "warning_rising_waters",
        WARNING,
        _headline(r"RISING WATERS\s*/\s*POTENTIAL FLASH FLOODS?"),
        _warning("warning_rising_waters"),
    ),
    # Dec 2021: "🌊 Risk of Flash Floods", "water levels in drains have reached 90% in these areas", bullets
    Format(
        "warning_wave",
        WARNING,
        _headline(r"🌊\s*Risk of Flash Floods?\b"),
        _warning("warning_wave"),
    ),
    # Feb 2023: "⚠️ Risk of Flash Floods", "please avoid the following location(s)", bullets
    Format(
        "warning_caution",
        WARNING,
        _headline(r"⚠️?\s*Risk of Flash Floods?\b"),
        _warning("warning_caution"),
    ),
    # Nov 2023: "[Risk of Flash Floods]" ("[Risk of Flash Flood]" once), one location after "for the next 1 hour:"
    Format(
        "warning_bracket",
        WARNING,
        _headline(r"\[\s*Risk of Flash Floods?\s*\]"),
        _warning("warning_bracket"),
    ),
    # Feb 2022: "🚨 Flash flood at the following location(s):", bullets, stamps sometimes missing
    Format(
        "flood_siren",
        FLASH_FLOOD,
        _headline(r"🚨\s*Flash floods? at the following locations?(?:\(s\))?\s*:"),
        _flood_list,
    ),
    # Nov 2024: "[FLASH FLOOD OCCURRED]", "Flash flood at <location>. Please avoid the area. ... Issued 1510 hours."
    Format(
        "flood_bracket",
        FLASH_FLOOD,
        _headline(r"\[\s*FLASH FLOODS? OCCURRED\s*\]"),
        _one_place(FLASH_FLOOD, "flood_bracket", r"Flash floods? at\s+"),
    ),
    # Feb 2022: "🔽 Flash flood at <location> has subsided. [14:25 hours]"
    Format(
        "subsided_arrow",
        FLOOD_SUBSIDED,
        _headline(r"🔽\s*Flash flood at\s+.+?\s+has subsided"),
        _subsided_arrow,
    ),
    # Nov 2023: "Flash flood subsided at <location>. Issued 1740 hours." (or "[17:40 hours]")
    Format(
        "subsided_plain",
        FLOOD_SUBSIDED,
        _headline(r"Flash flood subsided at\s+"),
        _one_place(FLOOD_SUBSIDED, "subsided_plain", r"Flash flood subsided at\s+"),
    ),
    # Nov 2021 on, from NEA: "⛈ Heavy rain expected over <regions> of Singapore from HH:MM to HH:MM hours."
    Format(
        "heavy_rain",
        HEAVY_RAIN,
        _headline(r"(?:Heavy Rain Warning\s+)?(?:⛈️?\s*)?Heavy rain expected over\b"),
        _heavy_rain,
    ),
    Format(
        "heavy_rain_eased",
        HEAVY_RAIN_EASED,
        _headline(r"The heavy rain has eased\b"),
        _eased,
    ),
)

# Loose clues for posts no format matches, most specific first; they only steer review.
_HINTS = (
    (
        WARNING,
        re.compile(
            r"risk of flash flood|90\s?%|water levels? in (?:the )?drains",
            re.IGNORECASE,
        ),
    ),
    (FLOOD_SUBSIDED, re.compile(r"\bsubsided\b", re.IGNORECASE)),
    (FLASH_FLOOD, re.compile(r"\bflash flood", re.IGNORECASE)),
    (HEAVY_RAIN_EASED, re.compile(r"\brain\b.*\beased\b", re.IGNORECASE | re.DOTALL)),
    (HEAVY_RAIN, re.compile(r"\bheavy rain\b", re.IGNORECASE)),
)


def normalize(text: str | None) -> str:
    """Line breaks and spaces as plain characters, and no blank space at either end."""
    return (
        (text or "")
        .replace("\r\n", "\n")
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .strip()
    )


def parse(text: str | None, *, service: bool = False) -> Parsed:
    """Classify one post's text (or one API description) and read its locations and stamps."""
    text = normalize(text)
    if service:
        return Parsed(NOT_RELEVANT, "service")
    if not text:
        return Parsed(NOT_RELEVANT, "empty")
    for fmt in FORMATS:
        if fmt.headline.match(text):
            return fmt.read(text)
    hint = next((category for category, clue in _HINTS if clue.search(text)), None)
    return Parsed(UNMATCHED, hint=hint)


def stamp_time(stamp: str, posted_at: datetime) -> datetime:
    """The moment a stamp ('HH:MM', SGT) names: the day of posting, or the day before when that would come later than
    the posting (beyond CLOCK_SKEW)."""
    posted = posted_at.astimezone(SGT)
    hour, minute = (int(part) for part in stamp.split(":"))
    at = posted.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return at - timedelta(days=1) if at > posted + CLOCK_SKEW else at


def _on_day(day: datetime, hhmm: str) -> datetime:
    """`day` at 'HH:MM'; '24:00' is the midnight that ends it."""
    hour, minute = int(hhmm[:2]), int(hhmm[3:])
    midnight = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(hours=hour, minutes=minute)


def window_times(
    valid: tuple[str, str], issued_at: datetime
) -> tuple[datetime, datetime]:
    """A heavy-rain window ('14:00', '15:00') as moments: it starts within 12 hours of being issued, and ends after it
    starts."""
    start = _on_day(issued_at.astimezone(SGT), valid[0])
    if start - issued_at > timedelta(hours=12):
        start -= timedelta(days=1)
    elif issued_at - start > timedelta(hours=12):
        start += timedelta(days=1)
    end = _on_day(start, valid[1])
    return start, end + timedelta(days=1) if end <= start else end


# What `parse_row` returns, as Spark DDL, for the silver UDF. Times are ISO strings with their offset (+08:00).
ROW_SCHEMA = (
    "category STRING, pattern STRING, problem STRING, hint STRING, "
    "locations ARRAY<STRUCT<location: STRING, stamp: STRING, stamp_ts: STRING>>, "
    "regions ARRAY<STRING>, issued_ts STRING, valid_from STRING, valid_to STRING"
)


def parse_row(
    text: str | None, posted_at: str | None, service: bool | None = False
) -> dict:
    """`parse` for Spark: a dict matching ROW_SCHEMA. `posted_at` is an ISO timestamp with an offset; each stamp is
    turned into a moment relative to it."""
    parsed = parse(text, service=bool(service))
    posted = datetime.fromisoformat(posted_at) if posted_at else None

    def moment(stamp: str | None) -> str | None:
        return stamp_time(stamp, posted).isoformat() if stamp and posted else None

    issued = moment(parsed.issued)
    valid_from = valid_to = None
    if parsed.valid and posted:
        start, end = window_times(
            parsed.valid, datetime.fromisoformat(issued) if issued else posted
        )
        valid_from, valid_to = start.isoformat(), end.isoformat()
    return {
        "category": parsed.category,
        "pattern": parsed.pattern,
        "problem": parsed.problem,
        "hint": parsed.hint,
        "locations": [
            {
                "location": loc.location,
                "stamp": loc.stamp,
                "stamp_ts": moment(loc.stamp),
            }
            for loc in parsed.locations
        ],
        "regions": list(parsed.regions),
        "issued_ts": issued,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }
