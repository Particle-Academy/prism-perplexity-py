"""The default transport follows a redirect only while it stays on the same origin.

urllib follows a 301, 302 or 303 (and a 307 or 308 on a GET) and re-sends every
header the request was built with, including ``Authorization``. The transport
built by ``urllib_transport`` therefore follows a redirect only while scheme,
host and port are unchanged; any other redirect is not followed and the call
fails with a ``PerplexityError``.

These cases run the REAL transport, through the functions an application calls,
against loopback servers only: every other connection is refused by the
``loopback_only`` fixture.
"""

from __future__ import annotations

import json
import socket
import socketserver
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from prism_perplexity import (
    AgentClient,
    PerplexityError,
    embeddings,
    search,
    urllib_transport,
)

KEY = "pplx-test-redirect-key"


class _LoopbackServer(ThreadingHTTPServer):
    """No reverse lookup of the bound address, which is slow for 127.0.0.2."""

    connections = 0

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def verify_request(self, request: Any, client_address: Any) -> bool:
        self.connections += 1
        return True


class Server:
    """A loopback server that records every connection and request."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.seen: list[tuple[str, str, dict[str, str], bytes]] = []
        self.redirect: tuple[int, str] | None = None
        # Per-path redirects, for chains whose Location depends on the path.
        self.route: Callable[[str], tuple[int, str] | None] | None = None
        self.landed: Any = {"error": {"message": "landed", "type": "landed"}}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.seen.append((self.command, self.path, dict(self.headers.items()), body))
                hop = outer.route(self.path) if outer.route is not None else None
                if (
                    hop is None
                    and outer.redirect is not None
                    and not self.path.startswith("/landed")
                ):
                    hop = outer.redirect
                if hop is not None:
                    status, location = hop
                    self.send_response(status)
                    self.send_header("Location", location)
                    self.send_header("Content-Type", "application/json")
                    payload = b'{"results":[]}'
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                payload = json.dumps(outer.landed).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _answer
            do_POST = _answer

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = _LoopbackServer((host, 0), Handler)
        self.port = self._server.server_port
        self.url = f"http://{host}:{self.port}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def connections(self) -> int:
        return self._server.connections

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection that is not to a loopback address."""
    original = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if not str(host).startswith("127."):
            raise OSError(f"network blocked in tests: {host!r}")
        return original(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def endpoint() -> Iterator[Server]:
    server = Server("127.0.0.1")
    yield server
    server.close()


# Another host, and the same host on another port: both are another origin.
@pytest.fixture(params=["127.0.0.2", "127.0.0.1"], ids=["other-host", "other-port"])
def other(request: pytest.FixtureRequest) -> Iterator[Server]:
    server = Server(request.param)
    yield server
    server.close()


def _calls(endpoint: Server) -> dict[str, Callable[[], Any]]:
    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)
    agent = AgentClient(http, sleeper=lambda _: None)
    return {
        "search": lambda: search(http, "prism"),
        "embeddings": lambda: embeddings(http, "pplx-embed", ["one"]),
        "contextualized-embeddings": lambda: embeddings(
            http, "pplx-embed", ["one", "two"], contextualized=True
        ),
        "agent-create": lambda: agent.create("research this"),
        "agent-retrieve": lambda: agent.retrieve("run-1"),
        "agent-cancel": lambda: agent.cancel("run-1"),
        "agent-wait": lambda: agent.wait("run-1", max_attempts=1),
    }


CALLS = [
    "search",
    "embeddings",
    "contextualized-embeddings",
    "agent-create",
    "agent-retrieve",
    "agent-cancel",
    "agent-wait",
]


def _assert_credential_absent(error: PerplexityError) -> None:
    assert KEY not in str(error)
    assert KEY not in repr(error)
    assert KEY not in error.message


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("call", CALLS)
def test_a_redirect_to_another_origin_is_not_followed(
    call: str, status: int, endpoint: Server, other: Server
) -> None:
    endpoint.redirect = (status, f"{other.url}/elsewhere")

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)[call]()

    _assert_credential_absent(raised.value)
    assert raised.value.status == status
    assert len(endpoint.seen) == 1
    assert endpoint.seen[0][2].get("Authorization") == f"Bearer {KEY}"
    assert other.seen == []
    assert other.connections == 0


@pytest.mark.parametrize("call", CALLS)
def test_a_scheme_change_is_another_origin(call: str, endpoint: Server) -> None:
    endpoint.redirect = (302, f"https://127.0.0.1:{endpoint.port}/landed")

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)[call]()

    _assert_credential_absent(raised.value)
    assert raised.value.status == 302
    assert endpoint.connections == 1
    assert len(endpoint.seen) == 1


def test_a_same_origin_redirect_is_still_followed(endpoint: Server) -> None:
    endpoint.redirect = (302, f"{endpoint.url}/landed/search")
    endpoint.landed = {"results": [{"url": "https://example.test"}]}

    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)

    assert search(http, "prism") == [{"url": "https://example.test"}]
    assert [(method, path) for method, path, _, _ in endpoint.seen] == [
        ("POST", "/search"),
        ("GET", "/landed/search"),
    ]
    assert endpoint.seen[1][2].get("Authorization") == f"Bearer {KEY}"


def test_a_relative_redirect_on_a_get_is_still_followed(endpoint: Server) -> None:
    endpoint.redirect = (307, "/landed/v1/agent/run-1")
    endpoint.landed = {"id": "run-1", "status": "completed"}

    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)
    response = AgentClient(http).retrieve("run-1")

    assert response.id == "run-1"
    assert [path for _, path, _, _ in endpoint.seen] == [
        "/v1/agent/run-1",
        "/landed/v1/agent/run-1",
    ]


@pytest.mark.parametrize("userinfo", ["u:p@", "evil.example@", "@"])
@pytest.mark.parametrize("call", CALLS)
def test_a_location_carrying_credentials_is_not_followed(
    call: str, userinfo: str, endpoint: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    looked_up: list[str] = []
    original = socket.getaddrinfo

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        looked_up.append(str(host))
        return original(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    endpoint.redirect = (302, f"http://{userinfo}127.0.0.1:{endpoint.port}/landed")

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)[call]()

    _assert_credential_absent(raised.value)
    assert raised.value.status == 302
    assert "@" not in raised.value.message
    assert len(endpoint.seen) == 1
    assert endpoint.connections == 1
    assert all("@" not in host for host in looked_up)


POST_CALLS = ["search", "embeddings", "contextualized-embeddings", "agent-create"]


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("call", POST_CALLS)
def test_a_same_origin_307_or_308_on_a_post_keeps_method_body_and_headers(
    call: str, status: int, endpoint: Server
) -> None:
    endpoint.redirect = (status, f"{endpoint.url}/landed")

    with pytest.raises(PerplexityError):  # the landing body is an error answer
        _calls(endpoint)[call]()

    assert len(endpoint.seen) == 2
    first, second = endpoint.seen
    assert first[0] == "POST"
    assert second[0] == "POST"
    assert second[1] == "/landed"
    assert second[3] == first[3]
    assert second[3] != b""
    assert second[2].get("Authorization") == f"Bearer {KEY}"
    assert second[2].get("Content-Type") == "application/json"
    assert second[2].get("Content-Length") == str(len(first[3]))


def test_a_same_origin_307_on_a_post_returns_the_landed_answer(endpoint: Server) -> None:
    endpoint.redirect = (307, f"{endpoint.url}/landed/search")
    endpoint.landed = {"results": [{"url": "https://example.test"}]}

    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)

    assert search(http, "prism") == [{"url": "https://example.test"}]
    assert [(method, path) for method, path, _, _ in endpoint.seen] == [
        ("POST", "/search"),
        ("POST", "/landed/search"),
    ]


def test_a_same_origin_307_loop_on_a_post_is_bounded(endpoint: Server) -> None:
    endpoint.redirect = (307, f"{endpoint.url}/again")

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)["search"]()

    assert raised.value.status == 307
    # urllib follows the same url at most four times (max_repeats): the first
    # request plus four to /again, and no more.
    assert len(endpoint.seen) == 5


@pytest.mark.parametrize("code", [302, 307])
def test_a_same_origin_chain_of_distinct_paths_is_bounded(endpoint: Server, code: int) -> None:
    # A relative "x/" resolves one level deeper each hop, so no url repeats.
    endpoint.redirect = (code, "x/")

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)["search"]()

    assert raised.value.status == code
    # At most ten redirects are followed: the first request plus ten, every
    # path distinct.
    paths = [path for _, path, _, _ in endpoint.seen]
    assert len(paths) == 11
    assert len(set(paths)) == 11


def _cycle(length: int, code: int) -> Callable[[str], tuple[int, str] | None]:
    """Any first path goes to /c0; /c0 .. /c<length-1> then go round for ever."""

    def route(path: str) -> tuple[int, str] | None:
        if path.startswith("/c"):
            return code, f"/c{(int(path[2:]) + 1) % length}"
        return code, "/c0"

    return route


def _chain(hops: int, code: int) -> Callable[[str], tuple[int, str] | None]:
    """Exactly ``hops`` redirects through distinct paths, then a 200 at /landed."""

    def route(path: str) -> tuple[int, str] | None:
        if path.startswith("/landed"):
            return None
        step = int(path[3:]) if path.startswith("/h/") else 0
        return code, ("/landed/done" if step + 1 >= hops else f"/h/{step + 1}")

    return route


# urllib's own limits count DISTINCT urls (ten) and repeats of one url (four),
# so on their own a chain cycling through a few urls is followed up to four
# times per url. The transport counts hops: ten followed, whatever the urls.
@pytest.mark.parametrize("length", [3, 9])
@pytest.mark.parametrize("code", [302, 307])
@pytest.mark.parametrize("call", ["search", "agent-retrieve"])
def test_a_cycling_same_origin_chain_follows_at_most_ten_redirects(
    endpoint: Server, call: str, code: int, length: int
) -> None:
    endpoint.route = _cycle(length, code)

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)[call]()

    assert raised.value.status == code
    _assert_credential_absent(raised.value)
    # The first request plus ten followed redirects; the eleventh is refused.
    assert len(endpoint.seen) == 11


@pytest.mark.parametrize("code", [302, 307])
def test_a_chain_of_exactly_ten_redirects_lands(endpoint: Server, code: int) -> None:
    endpoint.route = _chain(10, code)
    endpoint.landed = {"results": [{"url": "https://example.test"}]}

    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)

    assert search(http, "prism") == [{"url": "https://example.test"}]
    assert len(endpoint.seen) == 11
    assert endpoint.seen[-1][1] == "/landed/done"


@pytest.mark.parametrize("code", [302, 307])
def test_a_chain_of_eleven_redirects_is_refused(endpoint: Server, code: int) -> None:
    endpoint.route = _chain(11, code)

    with pytest.raises(PerplexityError) as raised:
        _calls(endpoint)["search"]()

    assert raised.value.status == code
    assert len(endpoint.seen) == 11
    assert all(not path.startswith("/landed") for _, path, _, _ in endpoint.seen)


def test_each_call_on_one_transport_gets_its_own_ten_redirects(endpoint: Server) -> None:
    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)
    endpoint.landed = {"results": []}

    endpoint.route = _cycle(9, 307)
    for _ in range(2):
        endpoint.seen.clear()
        with pytest.raises(PerplexityError):
            search(http, "prism")
        assert len(endpoint.seen) == 11

    # A budget spent by the calls above does not shorten the next one.
    endpoint.route = _chain(10, 307)
    for _ in range(2):
        endpoint.seen.clear()
        assert search(http, "prism") == []
        assert len(endpoint.seen) == 11


def test_concurrent_calls_on_one_transport_do_not_share_a_budget(endpoint: Server) -> None:
    http = urllib_transport(KEY, base_url=endpoint.url, timeout=5.0)
    endpoint.route = _chain(10, 302)
    endpoint.landed = {"results": []}
    outcomes: list[object] = []

    def run() -> None:
        try:
            outcomes.append(search(http, "prism"))
        except PerplexityError as error:
            outcomes.append(error)

    threads = [Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes == [[], []]
    assert len(endpoint.seen) == 22
