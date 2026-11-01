"""Jev as the fallback reader (labels/jev.py): the candidates it is shown, the request, and its answers read back.

No test calls Jev. `answers` plays a Jev that knows the post: it picks `category` and says yes to exactly the places
named. Over the whole archive (778 posts naming 899 places, checked 23 Oct 2026), every place the patterns read is a
candidate, and timing those candidates gives every one the stamp the patterns gave it; the fixtures below hold one post
per format.
"""

import json

import pytest

from daisi_team_master_corgis_floodsense.labels import alert_text as T
from daisi_team_master_corgis_floodsense.labels import jev
from daisi_team_master_corgis_floodsense.ml import contract as C
from tests.test_labels_alert_text import POSTS

# A format PUB has never used, for the stand-in tests.
NEW_FORMAT = (
    "⚠️ FLASH FLOOD ALERT ⚠️\n\nFlooding reported along Bukit Timah Rd (towards Clementi Rd). "
    "Please avoid the area. [15:21 hours]"
)


def answers(
    found: list[jev.Candidate],
    category: str,
    places: set[str],
    confidence: float = 0.95,
) -> dict:
    """What a Jev that knows the post would answer."""
    out = {
        "category": {
            "type": "choice",
            "choice": category,
            "confidence": confidence,
            "probabilities": {
                category: confidence,
                C.WARNING if category != C.WARNING else C.FLASH_FLOOD: 0.03,
            },
        }
    }
    for i, c in enumerate(found):
        out[f"place_{i}"] = {"type": "noul", "noul": 0.97 if c.text in places else 0.04}
    return out


def split(location: str) -> list[str]:
    """The places a pattern's location names (the bracket format lumps 'A; B' into one)."""
    return [T._clean(part) for part in location.split(";")]


ALERT_POSTS = [
    mid
    for mid, post in POSTS.items()
    if T.parse(post["text"], service=post["service"]).locations
]


@pytest.mark.parametrize("message_id", ALERT_POSTS)
def test_every_place_a_pattern_reads_is_a_candidate(message_id):
    text = POSTS[message_id]["text"]
    texts = {c.text for c in jev.candidates(text)}
    for loc in T.parse(text).locations:
        assert set(split(loc.location)) <= texts, (loc.location, texts)


@pytest.mark.parametrize("message_id", ALERT_POSTS)
def test_a_jev_that_knows_the_post_reads_what_the_pattern_reads(message_id):
    text = POSTS[message_id]["text"]
    by_pattern = T.parse(text)
    found = jev.candidates(text)
    named = {part for loc in by_pattern.locations for part in split(loc.location)}
    read = jev.read(text, found, answers(found, by_pattern.category, named))
    assert read.category == by_pattern.category and read.pattern == jev.PATTERN
    expected = [
        (part, loc.stamp)
        for loc in by_pattern.locations
        for part in split(loc.location)
    ]
    assert [(loc.location, loc.stamp) for loc in read.locations] == expected


def test_jev_stands_in_for_a_format_no_pattern_knows():
    assert T.parse(NEW_FORMAT).category == T.UNMATCHED
    found = jev.candidates(NEW_FORMAT)
    assert "Bukit Timah Rd (towards Clementi Rd)" in {c.text for c in found}
    read = jev.read(
        NEW_FORMAT,
        found,
        answers(found, C.FLASH_FLOOD, {"Bukit Timah Rd (towards Clementi Rd)"}),
    )
    assert read.category == C.FLASH_FLOOD and read.problem is None
    assert read.locations == (
        T.Located("Bukit Timah Rd (towards Clementi Rd)", "15:21"),
    )


def test_of_two_overlapping_places_the_likelier_is_kept():
    text = "🔽 Flash flood at Bedok North Ave 3 has subsided. [14:25 hours]"
    found = jev.candidates(text)
    said = answers(found, T.FLOOD_SUBSIDED, {"Bedok North Ave 3"})
    longer = next(
        i for i, c in enumerate(found) if c.text == "Bedok North Ave 3 has subsided"
    )
    said[f"place_{longer}"]["noul"] = 0.6  # also a yes, but less sure
    read = jev.read(text, found, said)
    assert read.locations == (T.Located("Bedok North Ave 3", "14:25"),)


def test_an_unsure_jev_leaves_the_post_unmatched_with_its_guess():
    found = jev.candidates(NEW_FORMAT)
    read = jev.read(
        NEW_FORMAT, found, answers(found, C.FLASH_FLOOD, set(), confidence=0.4)
    )
    assert (
        read.category == T.UNMATCHED
        and read.hint == C.FLASH_FLOOD
        and not read.is_alert
    )


def test_an_alert_with_no_likely_place_has_no_location():
    found = jev.candidates(NEW_FORMAT)
    read = jev.read(NEW_FORMAT, found, answers(found, C.FLASH_FLOOD, set()))
    assert read.category == C.FLASH_FLOOD and read.problem == T.NO_LOCATION


def test_heavy_rain_is_read_by_the_heavy_rain_reader():
    text = "NEA: heavy rain expected over western areas of Singapore from 14:00 to 15:00 hours. [Issued by NEA, 13:40]"
    assert T.parse(text).category == T.UNMATCHED
    read = jev.read(text, jev.candidates(text), answers([], T.HEAVY_RAIN, set()))
    assert (read.category, read.regions, read.valid, read.issued) == (
        T.HEAVY_RAIN,
        ("western",),
        ("14:00", "15:00"),
        "13:40",
    )


def test_needs_fallback_only_when_the_patterns_cannot_read_the_post():
    assert jev.needs_fallback(T.parse(NEW_FORMAT))
    assert jev.needs_fallback(
        T.parse("[FLASH FLOOD OCCURRED]\n\nPlease avoid the area.")
    )  # matched, no place
    assert not jev.needs_fallback(
        T.parse(POSTS[1268]["text"])
    )  # no stamp, but readable
    assert not jev.needs_fallback(T.parse(POSTS[1630]["text"]))


def test_one_request_asks_the_category_and_each_candidate():
    found = jev.candidates(NEW_FORMAT)
    body = jev.request(NEW_FORMAT, found)
    assert body["model"] == "system.ai.typesafeai-jev-1-14-0"
    assert set(body["questions"]) == {"category"} | {
        f"place_{i}" for i in range(len(found))
    }
    assert set(body["questions"]["category"]["criteria"]) == {
        C.WARNING, C.FLASH_FLOOD, T.FLOOD_SUBSIDED, T.HEAVY_RAIN, T.HEAVY_RAIN_EASED, T.NOT_RELEVANT
    }  # fmt: skip
    assert all(
        q["type"] == "noul" for k, q in body["questions"].items() if k != "category"
    )
    assert body["state"]["post"] == T.normalize(NEW_FORMAT)


def test_silver_reads_a_landed_call_as_the_fetch_did():
    found = jev.candidates(NEW_FORMAT)
    said = answers(found, C.FLASH_FLOOD, {"Bukit Timah Rd (towards Clementi Rd)"})
    landed = json.dumps([c.__dict__ for c in found]), json.dumps(said)
    row = jev.read_row(NEW_FORMAT, "2026-10-23T07:22:05+00:00", *landed)
    assert row == T.as_row(
        jev.read(NEW_FORMAT, found, said), "2026-10-23T07:22:05+00:00"
    )
    assert row["locations"][0]["stamp_ts"] == "2026-10-23T15:21:00+08:00"
    assert jev.read_row(NEW_FORMAT, None, landed[0], None) is None


def test_the_text_hash_matches_spark_sha2():
    # Spark's sha2(text, 256) hashes the UTF-8 bytes, as hashlib does here
    assert (
        jev.text_sha256("abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# --- a foundation model in Jev's place -----------------------------------------------------------------------------


def chat_reply(
    found: list[jev.Candidate], category: str, places: set[str], confidence: float = 0.9
) -> str:
    """What a chat model that knows the post replies, in the shape `prompt` asks for."""
    reply = {"category": {"choice": category, "confidence": confidence}}
    for i, c in enumerate(found):
        reply[f"place_{i}"] = {"answer": c.text in places, "confidence": 0.95}
    return json.dumps(reply)


def test_the_prompt_puts_the_post_and_every_question_to_the_chat_model():
    found = jev.candidates(NEW_FORMAT)
    body = jev.request(NEW_FORMAT, found, jev.FOUNDATION_MODEL)
    chat = jev.prompt(body)
    assert [m["role"] for m in chat["messages"]] == ["system", "user"]
    asked = chat["messages"][1]["content"]
    assert T.normalize(NEW_FORMAT) in asked and "json" in asked
    for option, meaning in jev.CATEGORY_QUESTION["criteria"].items():
        assert f"- {option}: {meaning}" in asked
    for i, c in enumerate(found):
        assert f'place_{i}: Is "{c.text}"' in asked
    assert (
        asked.count(jev.PLACE_CRITERIA["true"]) == 1
    )  # the shared rules are stated once, not per phrase
    assert chat["temperature"] == 0 and chat["response_format"] == {
        "type": "json_object"
    }
    assert jev.is_foundation(jev.FOUNDATION_MODEL) and not jev.is_foundation(jev.MODEL)


def test_a_chat_reply_maps_to_jev_answers_and_reads_the_same():
    found = jev.candidates(NEW_FORMAT)
    questions = jev.request(NEW_FORMAT, found)["questions"]
    said = jev.answers_from_chat(
        chat_reply(found, C.FLASH_FLOOD, {"Bukit Timah Rd (towards Clementi Rd)"}),
        questions,
    )
    assert said["category"] == {
        "type": "choice",
        "choice": C.FLASH_FLOOD,
        "confidence": 0.9,
        "probabilities": {C.FLASH_FLOOD: 0.9},
    }
    place = next(
        i
        for i, c in enumerate(found)
        if c.text == "Bukit Timah Rd (towards Clementi Rd)"
    )
    assert said[f"place_{place}"] == {"type": "noul", "noul": 0.95}
    others = [
        v["noul"] for k, v in said.items() if k not in ("category", f"place_{place}")
    ]
    assert others and all(
        abs(p - 0.05) < 1e-9 for p in others
    )  # false at 0.95 sure is 5% yes
    read = jev.read(NEW_FORMAT, found, said)
    assert read.category == C.FLASH_FLOOD and read.pattern == jev.PATTERN
    assert read.locations == (
        T.Located("Bukit Timah Rd (towards Clementi Rd)", "15:21"),
    )
    assert jev.summary(found, said)["runner_up"] is None


def test_a_sloppy_chat_reply_is_read_leniently_and_an_empty_one_is_an_error():
    found = jev.candidates(NEW_FORMAT)
    questions = jev.request(NEW_FORMAT, found)["questions"]
    sloppy = (
        '```json\n{"category": {"choice": "warning"}, "place_0": {"answer": "true", "confidence": "0.7"}, '
        '"place_1": {"answer": "maybe", "confidence": 0.5}, "place_2": "no", "place_9": {"answer": true}}\n```'
    )
    said = jev.answers_from_chat(sloppy, questions)
    assert (
        said["category"]["choice"] == C.WARNING
        and said["category"]["confidence"] == 1.0
    )  # not given: taken as sure
    assert said["place_0"] == {"type": "noul", "noul": 0.7}
    assert "place_1" not in said and "place_2" not in said and "place_9" not in said
    unsure = jev.answers_from_chat(
        '{"category": {"choice": "flash_flood", "confidence": 0.3}}', questions
    )
    assert jev.read(NEW_FORMAT, found, unsure).category == T.UNMATCHED
    with pytest.raises(ValueError, match="no JSON object"):
        jev.answers_from_chat("I cannot help with that.", questions)
    assert (
        jev.answers_from_chat(
            '{"category": {"choice": "something else", "confidence": 2}}', questions
        )
        == {}
    )
