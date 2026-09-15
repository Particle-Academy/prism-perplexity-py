# Prism Perplexity for Python

Perplexity's Search API, embeddings and Agent API: the parts of Perplexity that
Prism core does not carry. The Python port of
[`particle-academy/prism-perplexity`](https://github.com/Particle-Academy/prism-perplexity).

Zero runtime dependencies. Python 3.10+.

```
pip install prism-ai-perplexity
```

```python
import os

from prism_perplexity import AgentClient, search, urllib_transport

http = urllib_transport(os.environ["PERPLEXITY_API_KEY"])

for result in search(http, "what changed in the MCP 2026-07-28 revision"):
    print(result["title"], result["url"])

agent = AgentClient(http)
run = agent.create("Summarise this week's changes to the OpenAI Responses API.")
done = agent.wait(run.id)

print(done.text())
```

`urllib_transport(api_key, base_url, timeout)` uses the standard library. Any
callable that takes an `HttpRequest` (`method`, `path`, `body`) and returns an
`HttpResponse` (`status`, `json`) works in its place, so tests need no network.

## Search

`search(http, query, options)` returns web results with no model call: the
sources, without paying for an answer. `query` can be a string or a list of
strings. Results are plain dicts, because Perplexity documents the payload as
open-ended.

## Embeddings

`embeddings(http, model, inputs, contextualized=False)` calls `/embeddings`,
which embeds each input on its own. With `contextualized=True` it calls
`/contextualized-embeddings`, which treats the inputs as chunks of one document
and embeds each with the others in view.

## Agent API

`AgentClient` starts a long-running run, polls it and cancels it.

- `create(input, options)` starts a run in the background by default, so no
  request is held open for minutes.
- `retrieve(run_id)` and `cancel(run_id)` do what they say.
- `wait(run_id, max_attempts=60, interval_seconds=1.0)` polls until the run
  finishes, and raises `agent_wait_timed_out` if it does not.
- `AgentResponse.text()` joins the text parts of the output.
  `is_successful()` is true only for a completed run.

## Errors

Every failure is a `PerplexityError` with a stable `code`:
`unreadable_response`, `provider_error`, `invalid_response`,
`agent_wait_timed_out`, `invalid_argument`. Match on the code, not the message.

## Parity

prism-parity's `perplexity-agent-response` corpus pins how Agent API responses
are read against the PHP reference and the TypeScript port.

## License

MIT. See [LICENSE](LICENSE).
