"""The cross-language agent-response corpus from `prism-parity`.

The response body is UNTRUSTED input -- it is whatever the provider sent -- and
three things a consumer cannot decide for itself ride on how it is read:

- whether the run is FINISHED, which is what ``wait()`` turns on. Call a queued
  run terminal and an empty answer is returned as the final one; call a
  cancelled run live and the loop polls to timeout for a run that ended
  promptly.
- which CITATIONS it carries and in WHAT ORDER, because a UI numbers them and
  the answer text refers to them by that number.
- whether the body is refused, and under which identifier -- a consumer
  switches on that to tell a bad request from a rate limit.

The first two agree in all three languages on every row, which is the part
worth knowing. The identifiers do not, and those are pinned in the negative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prism_perplexity import AgentClient, HttpResponse, PerplexityError

_CORPUS_PATH = Path(__file__).parent / "fixtures" / "perplexity-agent-response.json"

with _CORPUS_PATH.open(encoding="utf-8") as handle:
    CORPUS = json.load(handle)


def parse(entry: dict[str, Any]) -> dict[str, Any]:
    """`retrieve` rather than `create`: it is the call the polling loop makes,
    it sends no body of its own, and so each row records the PARSE and nothing
    about the request that provoked it.
    """
    client = AgentClient(
        lambda _request: HttpResponse(status=entry["http_status"], json=entry["body"])
    )

    try:
        response = client.retrieve("resp_probe")
    except PerplexityError as error:
        return {
            "refused": True,
            "error_code": error.code,
            "error_type": error.provider_type,
        }

    return {
        "refused": False,
        "id": response.id,
        "status": response.status.value,
        "terminal": response.is_terminal(),
        "successful": response.is_successful(),
        "model": response.model,
        "created_at": response.created_at,
        "output_count": len(response.output),
        "annotations": response.annotations,
        "usage": response.usage,
        "error": None
        if response.error is None
        else {
            "message": response.error.message,
            "code": response.error.code,
            "type": response.error.type,
        },
        "text": response.text(),
    }


def case_of(case_id: str) -> dict[str, Any]:
    return next(entry for entry in CORPUS["cases"] if entry["id"] == case_id)


def test_is_the_whole_suite_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CORPUS["cases"]) == 17


@pytest.mark.parametrize("entry", CORPUS["cases"], ids=lambda e: e["id"])
def test_parses_the_way_the_corpus_recorded(entry: dict[str, Any]) -> None:
    assert parse(entry) == entry["parsed"]["py"]


def test_agrees_with_the_reference_on_whether_a_run_is_finished() -> None:
    """The load-bearing value, and the good news.

    ``wait()`` has exactly two ways to be wrong and both are silent, so this is
    asserted against live output rather than read off the corpus.
    """
    for entry in CORPUS["cases"]:
        if entry["parsed"]["php"]["refused"]:
            continue

        assert parse(entry)["terminal"] == entry["parsed"]["php"]["terminal"], entry["id"]


def test_agrees_with_the_reference_on_the_citations_and_their_order() -> None:
    """Holds across the malformed rows too: a bare string in an annotation list
    is dropped by all three, and the sibling that follows it still arrives.
    """
    for entry in CORPUS["cases"]:
        if entry["parsed"]["php"]["refused"]:
            continue

        assert parse(entry)["annotations"] == entry["parsed"]["php"]["annotations"], entry["id"]


def test_renders_an_absent_usage_as_a_mapping_where_the_reference_renders_a_list() -> None:
    """G-29, and a second instance of G-20.

    An empty PHP array encodes as ``[]``, never ``{}``. A consumer reading
    ``usage.input_tokens`` off serialised output gets a list where it expected
    an object, and which one depends on the language that served the request.
    """
    entry = case_of("agent-0002")

    assert parse(entry)["usage"] == {}
    assert isinstance(parse(entry)["usage"], dict)
    assert isinstance(entry["parsed"]["php"]["usage"], list)


def test_separates_its_own_refusal_code_from_the_provider_error_type() -> None:
    """G-30, and a difference in SHAPE rather than in naming.

    This port has a ``code`` it owns and reserves the provider type for what
    Perplexity actually said. The reference has no code at all and puts its own
    client identifier into the slot the provider's type would occupy -- so on a
    client-side refusal a consumer cannot tell "the provider called this
    invalid" from "this library did".
    """
    entry = case_of("agent-0008")
    parsed = parse(entry)

    assert parsed["error_code"] == "invalid_response"
    assert parsed["error_type"] is None
    assert entry["parsed"]["php"]["error_code"] is None
    assert entry["parsed"]["php"]["error_type"] == "invalid_response"


def test_calls_a_json_array_body_unreadable_which_is_what_it_is() -> None:
    """G-31, and one of two rows where this port is plainly right.

    PHP's ``is_array`` is true for a decoded JSON LIST as well as a map, so a
    body that was never a response passes the reference's readability check and
    fails its status check instead -- telling the caller the provider sent a
    response missing a status, when a proxy or a captive portal sent something
    that was never a response at all.

    Asserted in the POSITIVE: this is the behaviour to keep.
    """
    entry = case_of("agent-0012")

    assert parse(entry)["error_code"] == "unreadable_response"
    assert entry["parsed"]["php"]["error_type"] == "invalid_response"


def test_nulls_a_numeric_string_created_at_where_the_reference_coerces_it() -> None:
    """G-32.

    JSON has one number type and providers still send timestamps as strings.
    Neither answer is obviously right -- the reference is forgiving, this is
    predictable -- but a caller cannot have both, and a timestamp that exists in
    one language and is null in another is discovered by rendering a blank date.
    """
    entry = case_of("agent-0016")

    assert parse(entry)["created_at"] is None
    assert entry["parsed"]["php"]["created_at"] == 1730000000


def test_agrees_with_the_other_port_on_every_row() -> None:
    """Recorded because it is the useful half of the finding.

    This is a reference-versus-ports split, not a three-way scatter, so every
    divergence above has exactly one side that has to move.
    """
    assert [entry["id"] for entry in CORPUS["cases"] if not entry["ports_agree"]] == []
