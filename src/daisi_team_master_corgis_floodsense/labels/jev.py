"""Jev, the fallback reader for PUB posts no pattern can read: TypeSafe AI's Jev 1.14.0, served in Unity Catalog as
system.ai.typesafeai-jev-1-14-0 and called through Unity Gateway's System One API.

Jev is a decision model. It answers typed questions about a text with probabilities (a `choice` among named options,
or a `noul`, the probability of yes) and writes no text, so it can't copy a location out of a post. It can pick one,
though, and a post is read in three steps:

1. Candidates. `candidates` cuts the post into the phrases that could be a place without knowing its format: the
   pieces between line breaks, bullets, stamps, semicolons, colons and full stops, and each piece again with its lead
   ("Flash flood at ...") and its tail ("... has subsided") cut off. Each keeps its span in the text, so a place can be
   timed by the stamp after it.
2. One request. `request` asks, in one call, which kind of post it is (a choice over the six categories) and, for each
   candidate, whether it is a place the post reports on (a noul). Jev answers every question at once.
3. The reading. `read` turns the answers into an alert_text.Parsed, as if a pattern named 'jev' had matched: the
   category when Jev is at least CATEGORY_MIN sure of it (else the post stays `unmatched`, with Jev's guess as its
   hint), the candidates Jev gives at least PLACE_MIN (overlapping ones resolved to the likelier), and for each place
   the first stamp after it that comes before the next place, found by the patterns' own stamp regex. A post with one
   stamp gives it to every place. Heavy-rain posts are read by the heavy-rain readers, which search the whole text.

The answers are landed as they came (bronze.telegram_jev_parses, one row per call), and silver reads them again with
`read_row`, so changing a threshold here re-reads every past call without asking Jev again. Like alert_text, this
module uses only the standard library; the HTTP call is the caller's (ingest/fallback.py).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

from . import alert_text as T

MODEL = "system.ai.typesafeai-jev-1-14-0"
PATH = "/ai-gateway/typesafe/v1/systemone"
PATTERN = "jev"  # silver.telegram_posts.pattern (and method) of a post Jev read

# Below these, Jev's answer isn't used: the post stays unmatched (category) or the phrase isn't a place.
CATEGORY_MIN = 0.6
PLACE_MIN = 0.5
MAX_CANDIDATES = (
    16  # each is one question; the 2021-2023 lists name up to about 8 places
)

CATEGORY_QUESTION = {
    "type": "choice",
    "instructions": "Which kind of post from PUB, Singapore's national water agency, is this?",
    "criteria": {
        T.WARNING: "Risk of flash floods: water in drains or canals has risen high (for example to 90%) at named "
        "places, and people should avoid them for the next hour.",
        T.FLASH_FLOOD: "A flash flood has occurred, or is happening now, at named places.",
        T.FLOOD_SUBSIDED: "A flash flood at a named place has subsided or ended.",
        T.HEAVY_RAIN: "NEA expects heavy rain over areas of Singapore during a time window.",
        T.HEAVY_RAIN_EASED: "NEA says the heavy rain has eased.",
        T.NOT_RELEVANT: "Anything else: announcements, advisories, tips, scam warnings, channel messages.",
    },
}

# Splits a post into pieces: line breaks, bullets, semicolons, a colon or a full stop that ends a phrase.
_SEPARATOR = re.compile(r"\n|[•●▪◦‣·🔹🔸]|;|:(?=\s|$)|[.!?](?=\s|$)")
_LEAD = re.compile(r"^.*?\b(?:at|along)\s+", re.IGNORECASE)
_TAIL = re.compile(
    r"\s+(?:(?:has|have)\s+(?:now\s+)?(?:subsided|eased)|for the next|as of|since|until|(?:is|are)\s+"
    r"(?:flooded|flooding|closed|impassable))\b.*$",
    re.IGNORECASE,
)
# What a candidate can't start or end with: anything but a letter, a digit or a bracket.
_EDGES = re.compile(r"^[^\w(\[]+|[^\w)\]]+$")
# Pieces that are the post's fixed sentences, not places.
_NOT_PLACE = re.compile(
    r"^(?:please|pub\b|due to|risk of|roads? (?:are|is)|motorists|issued|the following|this location|these "
    r"(?:areas|locations)|the above|the area|heavy rain|the heavy rain|http)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    """A phrase of the post that could be a place, and where it sits in the normalised text."""

    text: str
    start: int
    end: int


def text_sha256(text: str | None) -> str:
    """The key of a post's version: Jev is asked once per id and text (silver's sha2(text, 256) of the same text)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def needs_fallback(parsed: T.Parsed) -> bool:
    """A post the patterns can't read: no format matched, or one did but its reader found no place."""
    return parsed.category == T.UNMATCHED or parsed.problem == T.NO_LOCATION


def _pieces(text: str) -> list[tuple[int, int]]:
    """Spans between separators and stamps."""
    cuts = sorted(
        [(m.start(), m.end()) for m in T._STAMP.finditer(text)]
        + [(m.start(), m.end()) for m in _SEPARATOR.finditer(text)]
    )
    spans, pos = [], 0
    for start, end in cuts:
        if start >= pos:
            spans.append((pos, start))
            pos = end
    spans.append((pos, len(text)))
    return spans


def _trimmed(text: str, start: int, end: int) -> Candidate | None:
    """The span without hashtags, and without the spaces, punctuation and symbols (bullets of any kind, emoji) at its
    ends, if it could name a place."""
    piece = text[start:end]
    cleaned = _EDGES.sub("", T._clean(piece))
    letters = sum(c.isalpha() for c in cleaned)
    if letters < 3 or len(cleaned) > 120 or _NOT_PLACE.match(cleaned):
        return None
    lead = re.match(r"[^\w(\[]*", piece).end()
    trail = re.search(r"[^\w)\]]*$", piece).start()
    return Candidate(cleaned, start + lead, start + trail)


def candidates(text: str | None) -> list[Candidate]:
    """The phrases of a post that could be a place, in the order they appear, at most MAX_CANDIDATES."""
    text = T.normalize(text)
    found: dict[str, Candidate] = {}
    for start, end in _pieces(text):
        piece = text[start:end]
        lead = _LEAD.match(piece)
        forms = [(start, end)]
        if lead:
            forms.append((start + lead.end(), end))
        for s, e in list(forms):
            tail = _TAIL.search(text[s:e])
            if tail:
                forms.append((s, s + tail.start()))
        for s, e in forms:
            candidate = _trimmed(text, s, e)
            if candidate and candidate.text.lower() not in found:
                found[candidate.text.lower()] = candidate
    ordered = sorted(found.values(), key=lambda c: (c.start, -c.end))
    return ordered[:MAX_CANDIDATES]


def _place_question(phrase: str) -> dict:
    return {
        "type": "noul",
        "instructions": f'Is "{phrase}" the name of a place in Singapore (a road, junction, canal, block or '
        "landmark) that this post reports a flood risk, a flash flood or a subsided flood at?",
        "criteria": {
            "true": "The phrase is that place's name as the post writes it, with any qualifier that is part of the "
            "location (such as 'towards ECP' or 'between X and Y'), and no other words.",
            "false": "The phrase is not such a place, or it also holds other words: the alert's wording, a time, "
            "or an instruction.",
        },
    }


def request(text: str | None, found: list[Candidate], model: str = MODEL) -> dict:
    """The System One request body: the category, and one place question per candidate."""
    questions = {"category": CATEGORY_QUESTION}
    questions |= {f"place_{i}": _place_question(c.text) for i, c in enumerate(found)}
    return {
        "model": model,
        "state": {
            "channel": "PUB Flood Alerts (Singapore) on Telegram",
            "post": T.normalize(text),
        },
        "questions": questions,
    }


def _stamps(text: str) -> list[tuple[int, str]]:
    return [(m.start(), T._stamp(m)) for m in T._STAMP.finditer(text)]


def _places(found: list[Candidate], answers: dict) -> list[tuple[Candidate, float]]:
    """The candidates Jev gives at least PLACE_MIN, likeliest first among overlapping ones, in text order."""
    scored = []
    for i, candidate in enumerate(found):
        p = (answers.get(f"place_{i}") or {}).get("noul")
        if isinstance(p, (int, float)) and p >= PLACE_MIN:
            scored.append((candidate, float(p)))
    kept: list[tuple[Candidate, float]] = []
    for candidate, p in sorted(scored, key=lambda cp: (-cp[1], len(cp[0].text))):
        if all(candidate.end <= k.start or candidate.start >= k.end for k, _ in kept):
            kept.append((candidate, p))
    return sorted(kept, key=lambda cp: cp[0].start)


def _timed(text: str, places: list[Candidate]) -> list[T.Located]:
    """Each place with the first stamp after it and before the next place; one stamp in the post times them all."""
    stamps = _stamps(text)
    distinct = {s for _, s in stamps}
    located = []
    for i, place in enumerate(places):
        until = places[i + 1].start if i + 1 < len(places) else len(text) + 1
        after = [s for pos, s in stamps if place.end <= pos < until]
        stamp = (
            after[0]
            if after
            else (next(iter(distinct)) if len(distinct) == 1 else None)
        )
        located.append(T.Located(place.text, stamp))
    return located


def read(text: str | None, found: list[Candidate], answers: dict) -> T.Parsed:
    """Jev's answers as a Parsed (pattern 'jev'), with the thresholds above."""
    text = T.normalize(text)
    category = answers.get("category") or {}
    choice, confidence = category.get("choice"), category.get("confidence") or 0.0
    if choice not in CATEGORY_QUESTION["criteria"] or confidence < CATEGORY_MIN:
        return T.Parsed(T.UNMATCHED, hint=choice)
    if choice == T.HEAVY_RAIN:
        return replace(T._heavy_rain(text), pattern=PATTERN)
    if choice == T.HEAVY_RAIN_EASED:
        return replace(T._eased(text), pattern=PATTERN)
    if choice == T.NOT_RELEVANT:
        return T.Parsed(T.NOT_RELEVANT, PATTERN)
    places = [c for c, _ in _places(found, answers)]
    return T._alert(choice, PATTERN, _timed(text, places))


def summary(found: list[Candidate], answers: dict) -> dict:
    """What Jev answered, for people: its category and runner-up with their probabilities, and each place's."""
    category = answers.get("category") or {}
    ranked = sorted(
        (category.get("probabilities") or {}).items(), key=lambda kv: -kv[1]
    )
    return {
        "category": category.get("choice"),
        "confidence": category.get("confidence"),
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "places": [
            (c.text, (answers.get(f"place_{i}") or {}).get("noul"))
            for i, c in enumerate(found)
        ],
    }


def read_row(
    text: str | None,
    posted_at: str | None,
    candidates_json: str | None,
    answers_json: str | None,
) -> dict | None:
    """`read` for Spark, from a landed call's candidates and answers (JSON): a dict matching alert_text.ROW_SCHEMA,
    or None when the call holds no answers."""
    if not answers_json:
        return None
    found = [
        Candidate(c["text"], c["start"], c["end"])
        for c in json.loads(candidates_json or "[]")
    ]
    return T.as_row(read(text, found, json.loads(answers_json)), posted_at)
