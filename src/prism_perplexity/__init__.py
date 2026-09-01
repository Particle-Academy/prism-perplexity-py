"""Perplexity's search, embeddings and Agent API -- the parts Prism core does not carry."""

from __future__ import annotations

import json as _json
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "AgentClient",
    "AgentError",
    "AgentResponse",
    "AgentStatus",
    "EmbeddingsResponse",
    "ErrorCode",
    "HttpRequest",
    "HttpResponse",
    "PerplexityError",
    "embeddings",
    "search",
    "urllib_transport",
]


class ErrorCode(str, Enum):
    """The stable identity of a failure.

    The reference throws Prism's ``PrismException`` with an English message and
    a provider-supplied error type. A class name does not survive a port and a
    sentence is not a contract, so the code is what a consumer branches on --
    the same decision as ``prism-ai`` and ``prism-ai-harness``.
    """

    #: The endpoint did not return a JSON object at all.
    UNREADABLE_RESPONSE = "unreadable_response"
    #: The endpoint returned JSON, and it carried an error.
    PROVIDER_ERROR = "provider_error"
    #: The response is missing something this client requires to proceed.
    INVALID_RESPONSE = "invalid_response"
    #: A background run did not finish before the attempts ran out.
    AGENT_WAIT_TIMED_OUT = "agent_wait_timed_out"
    #: An argument this client refuses to send.
    INVALID_ARGUMENT = "invalid_argument"


class PerplexityError(Exception):
    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        provider_type: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code: str = code.value if isinstance(code, ErrorCode) else code
        self.message = message
        #: The provider's own error type, when it gave one.
        self.provider_type = provider_type
        self.status = status

    def __repr__(self) -> str:
        return f"PerplexityError(code={self.code!r}, message={self.message!r})"


def _as_dict(value: Any) -> dict[str, Any]:
    """A dict, or an empty one. mypy does not narrow the inline conditional
    form, and the inline form is also where a `None` slips through unnoticed.
    """
    return value if isinstance(value, dict) else {}


def _as_str(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) else fallback


def _as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


# -- transport ---------------------------------------------------------------


@dataclass(frozen=True)
class HttpRequest:
    method: str
    #: Relative to the base url the transport was built with.
    path: str
    body: dict[str, Any] | None = None


@dataclass(frozen=True)
class HttpResponse:
    status: int
    #: The decoded body, or None when it was not JSON.
    json: Any = None


#: How this package reaches Perplexity.
#:
#: AN INTERFACE, not a dependency. The reference takes Laravel's
#: ``PendingRequest`` because it is already there; here the seam keeps the
#: package at zero dependencies and lets a consumer bring their own client,
#: retry policy and way of holding an API key. It is also what makes every test
#: run without a network.
HttpClient = Callable[[HttpRequest], HttpResponse]


def urllib_transport(
    api_key: str, base_url: str = "https://api.perplexity.ai", timeout: float = 30.0
) -> HttpClient:
    """A transport over the standard library, for a caller who does not want to write one."""
    import urllib.request

    root = base_url.rstrip("/")

    def send(request: HttpRequest) -> HttpResponse:
        data = None if request.body is None else _json.dumps(request.body).encode("utf-8")
        req = urllib.request.Request(
            f"{root}{request.path}",
            data=data,
            method=request.method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = int(response.status)
                raw = response.read()
        except urllib.error.HTTPError as error:  # an error body is still a body
            status = int(error.code)
            raw = error.read()

        # A non-JSON body is reported as None rather than raised here, so the
        # caller can raise `unreadable_response` WITH the status attached --
        # far more use than a parse error with no context.
        try:
            return HttpResponse(status=status, json=_json.loads(raw))
        except ValueError:
            return HttpResponse(status=status, json=None)

    return send


# -- search ------------------------------------------------------------------


def search(
    http: HttpClient, query: str | Sequence[str], options: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Perplexity's Search API -- web results, NO MODEL.

    Returns the sources a grounded answer would have been built from, without
    paying for the answer. Useful when the application wants to do its own
    synthesis, or to show a user where information came from before spending
    tokens summarising it.

    Results come back as plain dicts rather than wrapped in a value type,
    matching the reference and for its reason: Perplexity documents this
    payload as open-ended, and a wrapper naming only today's fields would
    quietly drop tomorrow's.
    """
    body = _compact(
        {**(options or {}), "query": list(query) if not isinstance(query, str) else query}
    )
    data = _read_json(
        http,
        HttpRequest("POST", "/search", body),
        "The search endpoint did not return a JSON object.",
    )
    _assert_no_error(data, "search_error")

    results = data.get("results")

    # No results is a legitimate answer to a search, not a failure -- the same
    # rule the Agent API sets for a completed run with no sources.
    return [row for row in results if isinstance(row, dict)] if isinstance(results, list) else []


# -- embeddings --------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingsResponse:
    embeddings: list[list[float]]
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


def embeddings(
    http: HttpClient,
    model: str,
    inputs: Sequence[str],
    contextualized: bool = False,
    document_context: str | None = None,
) -> EmbeddingsResponse:
    """Two endpoints, and the difference matters.

    ``/embeddings`` treats each input independently. ``/contextualized-embeddings``
    treats the inputs as chunks of ONE document and embeds each in light of the
    others, which is what you want when a document has been split for retrieval
    and you do not want every chunk to read as though it arrived without
    context.

    Prism's embeddings abstraction is text-in, vector-out and has no concept of
    "these belong together", so the second is reached deliberately rather than
    pretended to be the same call.
    """
    data = _read_json(
        http,
        HttpRequest(
            "POST",
            "/contextualized-embeddings" if contextualized else "/embeddings",
            _compact(
                {
                    "model": model,
                    "inputs": list(inputs),
                    "document_context": document_context,
                }
            ),
        ),
        "The embeddings endpoint did not return a JSON object.",
    )
    _assert_no_error(data, "embeddings_error")

    rows = _as_list(data.get("data"))

    return EmbeddingsResponse(
        embeddings=[
            [value for value in _as_list(row.get("embedding")) if isinstance(value, (int, float))]
            for row in rows
            if isinstance(row, dict)
        ],
        usage=_as_dict(data.get("usage")),
        model=_as_optional_str(data.get("model")),
    )


# -- the Agent API -----------------------------------------------------------


class AgentStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset(
    {AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.INCOMPLETE, AgentStatus.CANCELLED}
)


@dataclass(frozen=True)
class AgentError:
    message: str
    code: str | None = None
    type: str | None = None


@dataclass(frozen=True)
class AgentResponse:
    id: str
    status: AgentStatus
    model: str | None
    output: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    usage: dict[str, Any]
    error: AgentError | None
    created_at: int | None
    raw: dict[str, Any]

    def is_terminal(self) -> bool:
        return self.status.is_terminal()

    def is_successful(self) -> bool:
        return self.status is AgentStatus.COMPLETED

    def text(self) -> str:
        """Every text part of the output, joined."""
        parts: list[str] = []

        for item in self.output:
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])

        return "\n".join(parts)


Sleeper = Callable[[float], None]


class AgentClient:
    """The long-running Agent API: start a run, poll it, cancel it.

    ``background: True`` by default, matching the reference -- a research run
    can take minutes, and a request held open that long is one a proxy will cut.
    """

    def __init__(self, http: HttpClient, sleeper: Sleeper | None = None) -> None:
        self._http = http
        #: Injected so `wait()` is testable without real time passing.
        self._sleep = sleeper if sleeper is not None else time.sleep

    def create(
        self, input: str | Sequence[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> AgentResponse:
        body = dict(options or {})
        body["input"] = input if isinstance(input, str) else list(input)
        body.setdefault("background", True)

        return self._map(HttpRequest("POST", "/v1/agent", body))

    def retrieve(self, run_id: str) -> AgentResponse:
        return self._map(HttpRequest("GET", f"/v1/agent/{urllib.parse.quote(run_id, safe='')}"))

    def cancel(self, run_id: str) -> AgentResponse:
        return self._map(
            HttpRequest("POST", f"/v1/agent/{urllib.parse.quote(run_id, safe='')}/cancel")
        )

    def wait(
        self, run_id: str, max_attempts: int = 60, interval_seconds: float = 1.0
    ) -> AgentResponse:
        """Poll until the run reaches a terminal state.

        RAISES on timeout rather than returning the last non-terminal response:
        a caller handed back a ``queued`` would have to check the status again
        to know the wait failed, and the ones who forget treat an unfinished run
        as an empty answer.
        """
        if max_attempts < 1 or interval_seconds < 0:
            raise PerplexityError(
                ErrorCode.INVALID_ARGUMENT,
                "max_attempts must be at least 1 and interval_seconds cannot be negative.",
            )

        for attempt in range(1, max_attempts + 1):
            response = self.retrieve(run_id)

            if response.is_terminal():
                return response

            # No sleep after the LAST attempt: waiting out an interval only to
            # raise makes every timeout one interval slower than it needs to be.
            if attempt < max_attempts:
                self._sleep(interval_seconds)

        raise PerplexityError(
            ErrorCode.AGENT_WAIT_TIMED_OUT,
            f"The Perplexity agent run [{run_id}] did not finish within {max_attempts} attempt(s).",
        )

    def _map(self, request: HttpRequest) -> AgentResponse:
        response = self._http(request)
        data = response.json

        if not isinstance(data, dict):
            raise PerplexityError(
                ErrorCode.UNREADABLE_RESPONSE,
                "The Agent API did not return a JSON object.",
                status=response.status,
            )

        if not 200 <= response.status < 300:
            failure = _as_dict(data.get("error"))
            raise PerplexityError(
                ErrorCode.PROVIDER_ERROR,
                _as_str(failure.get("message"), "Unknown Agent API error."),
                _as_str(failure.get("type"), "agent_request_error"),
                response.status,
            )

        run_id = data.get("id")
        status: AgentStatus | None
        try:
            status = AgentStatus(data.get("status"))
        except ValueError:
            status = None

        if not isinstance(run_id, str) or not run_id or status is None:
            raise PerplexityError(
                ErrorCode.INVALID_RESPONSE,
                "The Agent API response is missing a recognized status or response id.",
                status=response.status,
            )

        output = [item for item in (data.get("output") or []) if isinstance(item, dict)]
        annotations = [
            annotation
            for item in output
            for part in _as_list(item.get("content"))
            if isinstance(part, dict)
            for annotation in _as_list(part.get("annotations"))
            if isinstance(annotation, dict)
        ]

        raw_error = _as_dict(data.get("error"))
        error: AgentError | None = (
            AgentError(
                message=raw_error["message"],
                code=_as_optional_str(raw_error.get("code")),
                type=_as_optional_str(raw_error.get("type")),
            )
            if isinstance(raw_error.get("message"), str)
            else None
        )

        return AgentResponse(
            id=run_id,
            status=status,
            model=_as_optional_str(data.get("model")),
            output=output,
            annotations=annotations,
            usage=_as_dict(data.get("usage")),
            error=error,
            created_at=data.get("created_at") if isinstance(data.get("created_at"), int) else None,
            raw=data,
        )


# -- shared ------------------------------------------------------------------


def _read_json(http: HttpClient, request: HttpRequest, unreadable: str) -> dict[str, Any]:
    response = http(request)

    if not isinstance(response.json, dict):
        raise PerplexityError(ErrorCode.UNREADABLE_RESPONSE, unreadable, status=response.status)

    return response.json


def _assert_no_error(data: dict[str, Any], fallback_type: str) -> None:
    if data.get("error") is None:
        return

    error = _as_dict(data["error"])

    raise PerplexityError(
        ErrorCode.PROVIDER_ERROR,
        _as_str(error.get("message"), "Unknown error"),
        _as_str(error.get("type"), fallback_type),
    )


def _compact(body: dict[str, Any]) -> dict[str, Any]:
    """Drop None and empty-list values.

    Matches the reference's ``array_filter``, and for a real reason: Perplexity
    rejects some fields sent as null that it accepts as absent, so an "unset"
    option must not travel as an explicit null.
    """
    return {key: value for key, value in body.items() if value is not None and value != []}
