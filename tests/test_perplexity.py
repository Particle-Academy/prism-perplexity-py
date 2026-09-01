"""Mirrors prism-perplexity-ts/test/perplexity.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from prism_perplexity import (
    AgentClient,
    AgentStatus,
    HttpRequest,
    HttpResponse,
    PerplexityError,
    embeddings,
    search,
)


def transport(replies: list[HttpResponse]) -> tuple[Any, list[dict[str, Any]]]:
    """Records what was sent, and replies with whatever the test scripted."""
    sent: list[dict[str, Any]] = []
    state = {"call": 0}

    def http(request: HttpRequest) -> HttpResponse:
        sent.append({"method": request.method, "path": request.path, "body": request.body})
        reply = replies[min(state["call"], len(replies) - 1)]
        state["call"] += 1
        return reply

    return http, sent


def ok(payload: Any) -> HttpResponse:
    return HttpResponse(status=200, json=payload)


# -- search ------------------------------------------------------------------


def test_posts_the_query_and_returns_the_results() -> None:
    http, sent = transport([ok({"results": [{"url": "https://example.test"}]})])

    assert search(http, "prism ecosystem") == [{"url": "https://example.test"}]
    assert sent[0]["method"] == "POST"
    assert sent[0]["path"] == "/search"


def test_takes_several_queries_at_once() -> None:
    http, sent = transport([ok({"results": []})])
    search(http, ["one", "two"])

    assert sent[0]["body"]["query"] == ["one", "two"]


def test_no_results_is_an_answer_not_a_failure() -> None:
    # The same rule the Agent API sets for a completed run with no sources.
    http, _ = transport([ok({"results": []})])

    assert search(http, "nothing at all") == []


def test_drops_an_unset_option_rather_than_sending_null() -> None:
    # Perplexity rejects some fields sent as null that it accepts as absent.
    http, sent = transport([ok({"results": []})])
    search(http, "q", {"max_results": None, "recency": "week"})

    assert sent[0]["body"] == {"query": "q", "recency": "week"}


def test_raises_the_provider_error_with_its_type() -> None:
    http, _ = transport([ok({"error": {"type": "rate_limited", "message": "Slow down."}})])

    with pytest.raises(PerplexityError) as caught:
        search(http, "q")

    assert caught.value.code == "provider_error"
    assert caught.value.provider_type == "rate_limited"


def test_names_a_non_json_body_rather_than_a_parse_error() -> None:
    http, _ = transport([HttpResponse(status=502, json=None)])

    with pytest.raises(PerplexityError) as caught:
        search(http, "q")

    assert caught.value.code == "unreadable_response"
    assert caught.value.status == 502


# -- embeddings --------------------------------------------------------------


def test_posts_to_embeddings_and_returns_the_vectors() -> None:
    http, sent = transport(
        [
            ok(
                {
                    "model": "pplx-embed",
                    "data": [{"embedding": [0.1, 0.2]}],
                    "usage": {"total_tokens": 4},
                }
            )
        ]
    )

    response = embeddings(http, "pplx-embed", ["hello"])

    assert sent[0]["path"] == "/embeddings"
    assert response.embeddings == [[0.1, 0.2]]
    assert response.usage == {"total_tokens": 4}
    assert response.model == "pplx-embed"


def test_contextualized_uses_a_different_endpoint() -> None:
    # Not a flag on the same call. Prism's embeddings abstraction has no concept
    # of "these inputs belong together", so this is reached deliberately.
    http, sent = transport([ok({"data": []})])
    embeddings(http, "m", ["a", "b"], contextualized=True, document_context="a report")

    assert sent[0]["path"] == "/contextualized-embeddings"
    assert sent[0]["body"]["document_context"] == "a report"


def test_omits_document_context_when_there_is_none() -> None:
    http, sent = transport([ok({"data": []})])
    embeddings(http, "m", ["a"])

    assert "document_context" not in sent[0]["body"]


def test_embeddings_raises_the_provider_error() -> None:
    http, _ = transport([ok({"error": {"message": "bad model"}})])

    with pytest.raises(PerplexityError) as caught:
        embeddings(http, "nope", ["a"])

    assert caught.value.provider_type == "embeddings_error"


# -- the Agent API -----------------------------------------------------------

QUEUED = ok({"id": "run-1", "status": "queued"})
DONE = ok(
    {
        "id": "run-1",
        "status": "completed",
        "model": "sonar-deep-research",
        "created_at": 1700000000,
        "usage": {"total_tokens": 100},
        "output": [
            {
                "content": [
                    {"text": "First part.", "annotations": [{"url": "https://a.test"}]},
                    {"text": "Second part."},
                ]
            }
        ],
    }
)


def test_creates_a_run_in_the_background_by_default() -> None:
    # A research run can take minutes, and a request held open that long is one
    # a proxy will cut.
    http, sent = transport([QUEUED])
    AgentClient(http).create("research prism")

    assert sent[0]["path"] == "/v1/agent"
    assert sent[0]["body"]["background"] is True


def test_the_caller_can_turn_background_off() -> None:
    http, sent = transport([QUEUED])
    AgentClient(http).create("x", {"background": False})

    assert sent[0]["body"]["background"] is False


def test_escapes_the_id_in_the_path() -> None:
    # An id is provider-supplied; interpolating one containing a slash would
    # reach a different endpoint entirely.
    http, sent = transport([DONE])
    AgentClient(http).retrieve("run/../admin")

    assert sent[0]["path"] == "/v1/agent/run%2F..%2Fadmin"


def test_collects_the_text_and_annotations_out_of_the_output() -> None:
    http, _ = transport([DONE])
    response = AgentClient(http).retrieve("run-1")

    assert response.text() == "First part.\nSecond part."
    assert response.annotations == [{"url": "https://a.test"}]
    assert response.is_successful() is True
    assert response.created_at == 1700000000


def test_knows_which_states_are_terminal() -> None:
    assert all(
        AgentStatus(value).is_terminal()
        for value in ("completed", "failed", "incomplete", "cancelled")
    )
    assert AgentStatus.QUEUED.is_terminal() is False
    assert AgentStatus.IN_PROGRESS.is_terminal() is False


def test_polls_until_the_run_is_terminal() -> None:
    slept: list[float] = []
    http, _ = transport([QUEUED, QUEUED, DONE])
    client = AgentClient(http, sleeper=slept.append)

    response = client.wait("run-1", 5, 0.25)

    assert response.is_successful() is True
    # Two waits for three attempts: none after the one that succeeded.
    assert slept == [0.25, 0.25]


def test_does_not_sleep_after_the_last_attempt() -> None:
    # Waiting out an interval only to raise makes every timeout one interval
    # slower than it needs to be.
    slept: list[float] = []
    http, _ = transport([QUEUED])
    client = AgentClient(http, sleeper=slept.append)

    with pytest.raises(PerplexityError) as caught:
        client.wait("run-1", 2, 0.1)

    assert caught.value.code == "agent_wait_timed_out"
    assert slept == [0.1]


def test_raises_on_timeout_rather_than_returning_a_non_terminal_response() -> None:
    # A caller handed back a `queued` would have to re-check the status to know
    # the wait failed, and the ones who forget treat an unfinished run as an
    # empty answer.
    http, _ = transport([QUEUED])

    with pytest.raises(PerplexityError):
        AgentClient(http, sleeper=lambda _s: None).wait("run-1", 1)


def test_refuses_a_nonsensical_wait_rather_than_looping_forever() -> None:
    http, _ = transport([QUEUED])

    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).wait("run-1", 0)

    assert caught.value.code == "invalid_argument"


def test_carries_a_failed_runs_error_without_raising() -> None:
    # A run that FAILED is a terminal answer, not a transport failure: the
    # caller asked what happened and this is what happened.
    http, _ = transport(
        [
            ok(
                {
                    "id": "run-1",
                    "status": "failed",
                    "error": {"message": "backend down", "code": "e50"},
                }
            )
        ]
    )

    response = AgentClient(http).retrieve("run-1")

    assert response.is_terminal() is True
    assert response.is_successful() is False
    assert response.error is not None
    assert response.error.message == "backend down"
    assert response.error.code == "e50"


def test_rejects_a_response_with_an_unrecognised_status() -> None:
    http, _ = transport([ok({"id": "run-1", "status": "vibing"})])

    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).retrieve("run-1")

    assert caught.value.code == "invalid_response"


def test_raises_an_http_failure_with_the_provider_type() -> None:
    http, _ = transport(
        [HttpResponse(status=429, json={"error": {"type": "rate_limit", "message": "Too many"}})]
    )

    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).retrieve("run-1")

    assert caught.value.code == "provider_error"
    assert caught.value.provider_type == "rate_limit"
    assert caught.value.status == 429
