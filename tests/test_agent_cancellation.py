"""Independent lifecycle expectations; cancellation is an acknowledgement, not completion."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prism_perplexity import AgentClient, HttpRequest, HttpResponse, PerplexityError


def transport(bodies: list[dict[str, Any]]) -> tuple[Any, list[HttpRequest]]:
    sent: list[HttpRequest] = []

    def http(request: HttpRequest) -> HttpResponse:
        sent.append(request)
        return HttpResponse(200, bodies[min(len(sent) - 1, len(bodies) - 1)])

    return http, sent


def test_cancel_acknowledges_cancelling_without_completion_or_success() -> None:
    body = {"id": "resp_probe", "status": "cancelling"}
    http, sent = transport([body])
    response = AgentClient(http).cancel("resp_probe")
    assert response.id == "resp_probe"
    assert response.status.value == "cancelling"
    assert response.is_terminal() is False
    assert response.is_successful() is False
    assert response.raw is body
    assert sent == [HttpRequest("POST", "/v1/agent/resp_probe/cancel")]


def test_retrieve_while_cancelling_is_not_terminal() -> None:
    http, _ = transport([{"id": "resp_probe", "status": "cancelling"}])
    response = AgentClient(http).retrieve("resp_probe")
    assert response.status.value == "cancelling"
    assert response.is_terminal() is False
    assert response.is_successful() is False


def test_wait_polls_through_cancelling_and_does_not_sleep_after_cancelled() -> None:
    http, sent = transport(
        [{"id": "resp_probe", "status": status} for status in ("queued", "cancelling", "cancelled")]
    )
    slept: list[float] = []
    response = AgentClient(http, sleeper=slept.append).wait("resp_probe", 3, 0.007)
    assert response.status.value == "cancelled"
    assert response.is_terminal() is True
    assert response.is_successful() is False
    assert len(sent) == 3
    assert slept == [0.007, 0.007]


def test_wait_times_out_cancelling_within_original_attempt_bound() -> None:
    http, sent = transport([{"id": "resp_probe", "status": "cancelling"}])
    slept: list[float] = []
    with pytest.raises(PerplexityError) as caught:
        AgentClient(http, sleeper=slept.append).wait("resp_probe", 2, 0.007)
    assert caught.value.code == "agent_wait_timed_out"
    assert len(sent) == 2
    assert slept == [0.007]


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"reason": "max_output_tokens"}, "max_output_tokens"),
        ({"reason": "future_reason"}, "future_reason"),
        ({"reason": ""}, ""),
        ({"reason": None}, None),
        ({"reason": 42}, None),
        ({"reason": False}, None),
        ({"reason": ["max_output_tokens"]}, None),
        ({"reason": {"value": "max_output_tokens"}}, None),
        ({}, None),
        (None, None),
        ("max_output_tokens", None),
    ],
)
def test_reads_only_string_incomplete_reason(details: Any, expected: str | None) -> None:
    body = {"id": "resp_probe", "status": "incomplete", "incomplete_details": details}
    http, _ = transport([body])
    response = AgentClient(http).retrieve("resp_probe")
    assert response.incomplete_reason() == expected
    assert response.is_terminal() is True
    assert response.is_successful() is False
    assert response.raw is body


def test_absent_incomplete_details_has_no_reason() -> None:
    http, _ = transport([{"id": "resp_probe", "status": "incomplete"}])
    assert AgentClient(http).retrieve("resp_probe").incomplete_reason() is None


def test_unknown_status_still_fails() -> None:
    http, _ = transport([{"id": "resp_probe", "status": "cancelling_forever"}])
    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).cancel("resp_probe")
    assert caught.value.code == "invalid_response"


def test_http_failure_is_not_a_cancellation_acknowledgement() -> None:
    client = AgentClient(
        lambda _: HttpResponse(
            400, {"error": {"type": "already_terminal", "message": "Already done"}}
        )
    )
    with pytest.raises(PerplexityError) as caught:
        client.cancel("resp_probe")
    assert caught.value.code == "provider_error"
    assert caught.value.provider_type == "already_terminal"
    assert caught.value.status == 400


def test_documented_cancel_accepts_only_response_id_and_cancelling() -> None:
    body = {"response_id": "resp_probe", "status": "cancelling"}
    http, sent = transport([body])
    response = AgentClient(http).cancel("resp_probe")
    assert response.id == "resp_probe"
    assert response.status.value == "cancelling"
    assert response.is_terminal() is False
    assert response.is_successful() is False
    assert response.model is None
    assert response.output == []
    assert response.annotations == []
    assert response.usage == {}
    assert response.created_at is None
    assert response.error is None
    assert response.raw is body
    assert response.raw == {"response_id": "resp_probe", "status": "cancelling"}
    assert sent == [HttpRequest("POST", "/v1/agent/resp_probe/cancel")]


def test_documented_cancel_prefers_response_id_when_both_keys_present() -> None:
    http, _ = transport([{"response_id": "resp_probe", "id": "resp_other", "status": "cancelling"}])
    assert AgentClient(http).cancel("resp_probe").id == "resp_probe"


@pytest.mark.parametrize("response_id", [None, "", 42, False, [], {}])
def test_documented_cancel_invalid_present_response_id_never_falls_back(response_id: Any) -> None:
    http, _ = transport([{"response_id": response_id, "id": "resp_probe", "status": "cancelling"}])
    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).cancel("resp_probe")
    assert caught.value.code == "invalid_response"


def test_documented_cancel_missing_both_identifiers_fails() -> None:
    http, _ = transport([{"status": "cancelling"}])
    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).cancel("resp_probe")
    assert caught.value.code == "invalid_response"


@pytest.mark.parametrize("operation", ["create", "retrieve"])
def test_documented_cancel_alias_does_not_apply_elsewhere(operation: str) -> None:
    http, _ = transport([{"response_id": "resp_probe", "status": "completed"}])
    client = AgentClient(http)
    with pytest.raises(PerplexityError) as caught:
        if operation == "create":
            client.create("synthetic")
        else:
            client.retrieve("resp_probe")
    assert caught.value.code == "invalid_response"


def test_documented_cancel_escapes_path_and_keeps_id_only_legacy_ack() -> None:
    http, sent = transport([{"id": "run/../admin", "status": "cancelling"}])
    assert AgentClient(http).cancel("run/../admin").id == "run/../admin"
    assert sent == [HttpRequest("POST", "/v1/agent/run%2F..%2Fadmin/cancel")]


@pytest.mark.parametrize(
    "body",
    [
        {"response_id": "resp_other", "status": "cancelling"},
        {"id": "resp_other", "status": "cancelling"},
        {"response_id": "resp_other", "id": "resp_probe", "status": "cancelling"},
        {"response_id": "resp_Probe", "status": "cancelling"},
    ],
)
def test_cancel_refuses_mismatched_ack_without_retry_or_raw_mutation(body: dict[str, Any]) -> None:
    original = json.loads(json.dumps(body))
    http, sent = transport([body])
    with pytest.raises(PerplexityError) as caught:
        AgentClient(http).cancel("resp_probe")
    assert caught.value.code == "invalid_response"
    assert caught.value.status == 200
    assert (
        "Cancellation state is uncertain. Use retrieve() with the requested ID to check its state."
        in str(caught.value)
    )
    assert "resp_probe" not in str(caught.value)
    assert "resp_other" not in str(caught.value)
    assert sent == [HttpRequest("POST", "/v1/agent/resp_probe/cancel")]
    assert body == original


@pytest.mark.parametrize("run_id", ["resp_Aa", "resp_0"])
def test_cancel_accepts_matching_opaque_identifier(run_id: str) -> None:
    body = {"response_id": run_id, "status": "cancelling"}
    http, sent = transport([body])
    response = AgentClient(http).cancel(run_id)
    assert response.id == run_id
    assert response.raw is body
    assert response.raw == {"response_id": run_id, "status": "cancelling"}
    assert len(sent) == 1
