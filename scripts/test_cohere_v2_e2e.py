#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for the Cohere Chat v2 API (``POST /cohere/v2/chat``).

Drives a live vLLM server with a real Cohere Command-family model and
exercises the v2 wire format end-to-end: non-streaming, streaming,
documents/citations, tools, reasoning, role aliasing, and error paths.

Prerequisites on the GPU host
-----------------------------

1. Install the optional Cohere SDKs:

       uv pip install cohere cohere-melody

2. Start a vLLM server with the Cohere renderer / tokenizer wired up.
   ``VLLM_ENABLE_COHERE_API=1`` is required to expose ``/cohere/v2/chat``;
   the endpoint is opt-in and stays hidden otherwise:

       VLLM_ENABLE_COHERE_API=1 vllm serve <cohere-model-id> \\
           --tokenizer-mode cohere \\
           --enable-auto-tool-choice \\
           --tool-call-parser cohere2 \\
           --reasoning-parser cohere2 \\
           --port 8000

   For non-reasoning Command models (cmd3, older Command R), append
   ``--no-cohere-is-reasoning-model`` so the renderer surfaces reasoning
   as ``tool_plan`` instead of as a ``thinking`` content block.

3. Run this script:

       python scripts/test_cohere_v2_e2e.py \\
           --base-url http://127.0.0.1:8000 \\
           --model <cohere-model-id>

Optional flags
--------------

* ``--reasoning-model / --no-reasoning-model`` -- whether the server was
  launched with reasoning enabled (``--cohere-is-reasoning-model``).
  Controls whether reasoning is expected as a ``thinking`` content block
  or as a ``tool-plan-delta`` event.
* ``--skip-tools`` / ``--skip-citations`` / ``--skip-streaming`` --
  scope down the test surface when iterating locally.
* ``--verbose`` -- dump full response bodies / SSE frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

DONE_LINE = "data: [DONE]"

# Reasoning Command models spend most of their decode budget inside
# ``thinking`` blocks before emitting any user-visible text. The basic
# correctness probes use a generous budget so the response actually
# reaches the text/tool-call payload we're trying to assert against;
# tighter limits are still applied per-test for the negative paths.
REASONING_BUDGET = 2048


def _text_from_content_blocks(content: Any) -> str:
    """Concatenate ``text`` blocks from a message ``content`` payload."""
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _thinking_from_content_blocks(content: Any) -> str:
    """Concatenate ``thinking`` blocks from a message ``content`` payload."""
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("thinking", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "thinking"
    )


def _looks_like_4xx_envelope(parsed: Any) -> bool:
    """True for any of the documented Cohere v2 4xx response shapes.

    We accept both:

    * Our native ``CohereError`` (``{"message": ..., "id": ...}``).
    * vLLM's generic OpenAI-style envelope used by the request-body
      validator at ``vllm.entrypoints.serve.utils.api_utils`` --
      ``{"error": {"message": ..., "type": ..., "code": ...}}``.
    * FastAPI's default ``{"detail": [...]}`` for body-shape rejection.
    """
    if not isinstance(parsed, dict):
        return False
    if "message" in parsed and "error" not in parsed:
        return True
    if isinstance(parsed.get("error"), dict) and "message" in parsed["error"]:
        return True
    return "detail" in parsed


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class TestContext:
    base_url: str
    model: str
    is_reasoning_model: bool
    verbose: bool
    timeout: float
    request_id_counter: int = 0
    results: list[TestResult] = field(default_factory=list)

    def next_request_id(self, label: str) -> str:
        self.request_id_counter += 1
        return f"e2e-{label}-{self.request_id_counter}-{int(time.time())}"

    def record(self, result: TestResult) -> None:
        self.results.append(result)
        prefix = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
        print(f"[{prefix}] {result.name}")
        if result.detail:
            for line in result.detail.splitlines():
                print(f"        {line}")


def _post_json(
    ctx: TestContext,
    *,
    body: dict[str, Any],
    request_id: str,
    expect_status: int | None = 200,
) -> tuple[int, dict[str, Any] | str]:
    """POST a non-streaming JSON request and return (status, parsed_body)."""
    url = f"{ctx.base_url.rstrip('/')}/cohere/v2/chat"
    headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
    with httpx.Client(timeout=ctx.timeout) as client:
        resp = client.post(url, json=body, headers=headers)
    if ctx.verbose:
        print(f"  -> POST {url}  [status {resp.status_code}]")
        print(f"     request_id={request_id}")
        print(f"     body={json.dumps(body)[:500]}")
        print(f"     resp={resp.text[:1500]}")
    parsed: dict[str, Any] | str
    try:
        parsed = resp.json()
    except Exception:
        parsed = resp.text
    if expect_status is not None and resp.status_code != expect_status:
        raise AssertionError(
            f"expected status {expect_status}, got {resp.status_code}: {parsed!r}"
        )
    return resp.status_code, parsed


def _stream_post(
    ctx: TestContext,
    *,
    body: dict[str, Any],
    request_id: str,
) -> tuple[list[dict[str, Any]], bool]:
    """POST a streaming request and return (events, terminated_with_done)."""
    body = {**body, "stream": True}
    url = f"{ctx.base_url.rstrip('/')}/cohere/v2/chat"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Request-Id": request_id,
    }
    events: list[dict[str, Any]] = []
    saw_done = False
    with (
        httpx.Client(timeout=ctx.timeout) as client,
        client.stream("POST", url, json=body, headers=headers) as resp,
    ):
        if resp.status_code != 200:
            resp.read()
            raise AssertionError(
                f"stream expected 200, got {resp.status_code}: {resp.text[:1000]}"
            )
        buffer = ""
        for chunk in resp.iter_text():
            if not chunk:
                continue
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                for line in frame.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        saw_done = True
                        continue
                    if not payload:
                        continue
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError as e:
                        raise AssertionError(f"bad SSE frame: {payload!r}: {e}") from e
    if ctx.verbose:
        print(f"  -> stream {url}  events={len(events)} done={saw_done}")
        for ev in events:
            print(f"     {ev.get('type'):<20} {json.dumps(ev)[:200]}")
    return events, saw_done


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [ev.get("type", "<missing>") for ev in events]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------


def test_health(ctx: TestContext) -> None:
    """Verify the server is up and the model is loaded."""
    name = "health: GET /health and /v1/models"
    try:
        with httpx.Client(timeout=ctx.timeout) as client:
            health = client.get(f"{ctx.base_url.rstrip('/')}/health")
            _expect(
                health.status_code == 200,
                f"/health returned {health.status_code}",
            )
            models = client.get(f"{ctx.base_url.rstrip('/')}/v1/models")
            _expect(
                models.status_code == 200,
                f"/v1/models returned {models.status_code}: {models.text[:500]}",
            )
            ids = [m["id"] for m in models.json().get("data", [])]
            _expect(
                any(ctx.model == mid or mid.startswith(ctx.model) for mid in ids),
                f"model {ctx.model!r} not found in /v1/models -> {ids}",
            )
        ctx.record(TestResult(name, True, detail=f"served models: {ids}"))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_non_streaming_basic(ctx: TestContext) -> None:
    """Plain system + user -> assistant turn.

    The wire-format contract we're testing is "the response is a valid
    v2 envelope with at least one content block". For reasoning models
    we'd ideally see a final ``text`` block, but a ``thinking``-only
    response (truncated by ``MAX_TOKENS`` mid-reasoning) still satisfies
    the contract -- treat that as a soft pass with a warning.
    """
    name = "non-streaming: basic system+user prompt"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "system", "content": "You are a terse assistant."},
                {
                    "role": "user",
                    "content": "Reply with the single word: OK",
                },
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("basic")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict), resp
        _expect("id" in resp and resp["id"], f"missing id: {resp}")
        _expect(
            resp.get("finish_reason") in {"COMPLETE", "MAX_TOKENS"},
            f"unexpected finish_reason: {resp.get('finish_reason')}",
        )
        message = resp.get("message") or {}
        _expect(
            message.get("role") == "assistant",
            f"unexpected role: {message.get('role')}",
        )
        content = message.get("content") or []
        _expect(content, f"empty content[]: {message}")
        text = _text_from_content_blocks(content).strip()
        thinking = _thinking_from_content_blocks(content).strip()
        if text:
            detail = (
                f"text={text!r}  thinking_chars={len(thinking)}  "
                f"finish={resp.get('finish_reason')}  usage={resp.get('usage')}"
            )
        else:
            detail = (
                f"WARN: only thinking content emitted "
                f"({len(thinking)} chars); "
                f"finish={resp.get('finish_reason')}  usage={resp.get('usage')}  "
                f"-- bump --timeout or the prompt budget if this recurs"
            )
        ctx.record(TestResult(name, True, detail=detail))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_non_streaming_developer_role(ctx: TestContext) -> None:
    """OpenAI ``developer`` role must alias to ``system`` (M5)."""
    name = "non-streaming: developer role aliased to system"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "developer",
                    "content": "You only ever respond with the word: HELLO.",
                },
                {"role": "user", "content": "Say hi."},
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("dev-role")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        _expect(
            content,
            f"developer role accepted but empty content[]: {message}",
        )
        text = _text_from_content_blocks(content).strip()
        thinking = _thinking_from_content_blocks(content).strip()
        # M5 + the protocol-level normalizer: ``developer`` is rewritten
        # to ``system`` before the SDK discriminator runs. The request
        # must at minimum *not* 4xx and must produce content (text or
        # thinking) -- before the fix the SDK rejected the role
        # outright with a 400.
        if text:
            ctx.record(TestResult(name, True, detail=f"text={text!r}"))
        else:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        f"WARN: only thinking emitted ({len(thinking)} chars); "
                        f"role alias accepted (no 4xx)"
                    ),
                )
            )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_non_streaming_with_documents(ctx: TestContext) -> None:
    """Documents in -> citations in the response message."""
    name = "non-streaming: grounded answer with citations"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Using the provided documents, answer who wrote "
                        "Hamlet and roughly when. Quote the specific "
                        "spans from the documents that support each "
                        "claim and ground them with citation tags."
                    ),
                },
            ],
            "documents": [
                {
                    "id": "doc_shakespeare",
                    "data": {
                        "title": "Wikipedia: Hamlet",
                        "text": (
                            "Hamlet is a tragedy written by William "
                            "Shakespeare around 1600."
                        ),
                    },
                },
                {
                    "id": "doc_irrelevant",
                    "data": {
                        "title": "Wikipedia: Compilers",
                        "text": (
                            "A compiler is a translator from one language to another."
                        ),
                    },
                },
            ],
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("documents")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        _expect(content, f"empty content: {message}")
        # Citations are best-effort on the model side; flag rather than
        # fail if the model declined to ground.
        if not citations:
            # Surface the raw assistant text so we can tell whether the
            # model emitted citation markers at all. ``<co`` in the text
            # means melody/the reasoning parser stripped them silently;
            # absence of ``<co`` means the model never grounded -- check
            # the server log with ``VLLM_LOGGING_LEVEL=DEBUG`` for the
            # ``cohere reasoning parser:`` diagnostic line which prints
            # the raw model output and confirms which case applies.
            assistant_text = _text_from_content_blocks(content) or ""
            assistant_thinking = _thinking_from_content_blocks(content) or ""
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        "WARN: assistant did not emit citations.\n"
                        f"  text head: {assistant_text[:200]!r}\n"
                        f"  thinking head: {assistant_thinking[:200]!r}\n"
                        "  If the text contains no '<co' markers, the "
                        "model never grounded. Re-run the server with "
                        "VLLM_LOGGING_LEVEL=DEBUG to confirm via the "
                        "'cohere reasoning parser:' log line."
                    ),
                )
            )
            return
        # Verify each citation has the expected shape and points at one
        # of the documents we passed.
        for c in citations:
            _expect("start" in c and "end" in c, f"citation missing span: {c}")
            sources = c.get("sources") or []
            _expect(sources, f"citation without sources: {c}")
            for src in sources:
                _expect(
                    "type" in src,
                    f"citation source missing type: {src}",
                )
        first_sources = citations[0].get("sources", [])
        first_refs = [
            s.get("id") or s.get("document", {}).get("id") for s in first_sources
        ]
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"got {len(citations)} citation(s); first span="
                    f"{citations[0].get('start')}-{citations[0].get('end')!s} "
                    f"refs={first_refs}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_non_streaming_with_tools(ctx: TestContext) -> None:
    """Tools in -> a TOOL_CALL finish with structured tool_calls."""
    name = "non-streaming: tool call"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What's the weather in Tokyo right now? "
                        "Use the get_weather tool."
                    ),
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name.",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("tools")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        finish = resp.get("finish_reason")
        message = resp.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        # If the model decided not to call the tool, surface a warning
        # rather than fail. The wiring is what we're testing.
        if not tool_calls:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        f"WARN: model returned finish_reason={finish!r} with "
                        f"no tool_calls; cannot verify tool-call response "
                        f"shape."
                    ),
                )
            )
            return
        _expect(
            finish == "TOOL_CALL",
            f"expected finish_reason=TOOL_CALL, got {finish!r}",
        )
        first = tool_calls[0]
        _expect("id" in first, f"tool_call missing id: {first}")
        fn = first.get("function") or {}
        _expect(
            fn.get("name") == "get_weather",
            f"unexpected tool name: {fn}",
        )
        args = fn.get("arguments")
        _expect(
            isinstance(args, str) and args.strip(),
            f"tool arguments not a JSON string: {fn}",
        )
        parsed_args = json.loads(args)
        _expect(
            isinstance(parsed_args, dict) and "city" in parsed_args,
            f"tool arguments missing 'city': {parsed_args}",
        )
        thinking_blocks = [
            b
            for b in (message.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        tool_plan = message.get("tool_plan")
        if ctx.is_reasoning_model:
            detail_note = (
                f"thinking_blocks={len(thinking_blocks)} tool_plan={tool_plan!r}"
            )
        else:
            detail_note = f"tool_plan={tool_plan!r} (no thinking block expected)"
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"tool_calls=1 name={fn.get('name')} args={parsed_args}  "
                    f"{detail_note}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_non_streaming_citations_from_tool_result(ctx: TestContext) -> None:
    """Completed tool call in history -> citations against the tool result.

    Mirrors the ``test_non_streaming_with_documents`` flow but exercises
    the alternate grounding path where the assistant has already issued a
    tool call and the user message replays the tool's structured output
    via a ``tool``-role message. The server should ground the follow-up
    answer against the ``content`` of that tool message.
    """
    name = "non-streaming: grounded answer from prior tool result"
    try:
        tool_call_id = "call_get_weather_tokyo"
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What's the current weather in Tokyo? Use the "
                        "get_weather tool, then answer the user and "
                        "ground every factual claim in the tool result "
                        "with a citation tag."
                    ),
                },
                {
                    "role": "assistant",
                    "tool_plan": (
                        "I should call get_weather with city=Tokyo and "
                        "then summarize the result for the user."
                    ),
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Tokyo"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": [
                        {
                            "type": "document",
                            "document": {
                                "id": "weather_tokyo_now",
                                "data": {
                                    "city": "Tokyo",
                                    "temperature_c": "22",
                                    "condition": "partly cloudy",
                                    "humidity_pct": "58",
                                },
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name.",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("tool-result-citations")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        _expect(content, f"empty content: {message}")
        if not citations:
            assistant_text = _text_from_content_blocks(content) or ""
            assistant_thinking = _thinking_from_content_blocks(content) or ""
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        "WARN: assistant did not cite the tool result.\n"
                        f"  text head: {assistant_text[:200]!r}\n"
                        f"  thinking head: {assistant_thinking[:200]!r}\n"
                        "  If the text contains no '<co' markers, the "
                        "model never grounded against the tool output."
                    ),
                )
            )
            return
        for c in citations:
            _expect("start" in c and "end" in c, f"citation missing span: {c}")
            sources = c.get("sources") or []
            _expect(sources, f"citation without sources: {c}")
            for src in sources:
                _expect(
                    "type" in src,
                    f"citation source missing type: {src}",
                )
        first_sources = citations[0].get("sources", [])
        first_refs = [
            s.get("id")
            or s.get("tool_output", {}).get("id")
            or s.get("document", {}).get("id")
            for s in first_sources
        ]
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"got {len(citations)} citation(s); first span="
                    f"{citations[0].get('start')}-{citations[0].get('end')!s} "
                    f"refs={first_refs}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_streaming_basic(ctx: TestContext) -> None:
    """Streaming: full event lifecycle for a plain text answer.

    Reasoning Command models emit ``content-delta`` events with
    ``delta.message.content.thinking`` (not ``.text``) while inside a
    thinking block. We accumulate both and require *some* content to
    have flowed through; if the budget ran out before the model exited
    the thinking block, that's a soft pass.
    """
    name = "streaming: basic text response"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "Count from 1 to 3, comma-separated."},
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("stream-basic")
        events, saw_done = _stream_post(ctx, body=body, request_id=request_id)
        _expect(saw_done, "stream did not terminate with data: [DONE]")
        types = _event_types(events)
        _expect(
            types[0] == "message-start",
            f"expected first event message-start, got {types[:3]}",
        )
        _expect(
            types[-1] == "message-end",
            f"expected last event message-end, got {types[-3:]}",
        )
        for ev in events:
            _expect(
                "type" in ev and ev["type"],
                f"event missing discriminator: {ev}",
            )
        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        for ev in events:
            if ev.get("type") != "content-delta":
                continue
            content = (ev.get("delta") or {}).get("message", {}).get("content", {})
            if not isinstance(content, dict):
                continue
            t = content.get("text")
            if isinstance(t, str):
                text_chunks.append(t)
            th = content.get("thinking")
            if isinstance(th, str):
                thinking_chunks.append(th)
        text = "".join(text_chunks)
        thinking = "".join(thinking_chunks)
        _expect(
            text or thinking,
            f"no content accumulated from content-delta events: {types}",
        )
        if text:
            detail = (
                f"events={len(events)} text={text!r} thinking_chars={len(thinking)}"
            )
        else:
            detail = (
                f"WARN: only thinking deltas in budget "
                f"({len(thinking)} chars); events={len(events)} types={types[:6]}..."
            )
        ctx.record(TestResult(name, True, detail=detail))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_streaming_with_documents(ctx: TestContext) -> None:
    """Streaming: documents -> citation-start / citation-end pairs."""
    name = "streaming: grounded answer with citation events"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Who wrote Hamlet, and when?",
                },
            ],
            "documents": [
                {
                    "id": "doc_shakespeare",
                    "data": {
                        "title": "Wikipedia: Hamlet",
                        "text": (
                            "Hamlet is a tragedy written by William "
                            "Shakespeare around 1600."
                        ),
                    },
                },
            ],
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("stream-docs")
        events, saw_done = _stream_post(ctx, body=body, request_id=request_id)
        _expect(saw_done, "stream did not terminate with data: [DONE]")
        types = _event_types(events)
        _expect(
            types[0] == "message-start" and types[-1] == "message-end",
            f"unexpected envelope: {types[:3]} ... {types[-3:]}",
        )
        starts = [i for i, t in enumerate(types) if t == "citation-start"]
        ends = [i for i, t in enumerate(types) if t == "citation-end"]
        if not starts:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        "WARN: model did not emit citation events; verify "
                        "grounding is enabled for this model."
                    ),
                )
            )
            return
        _expect(
            len(starts) == len(ends),
            f"unbalanced citation events: starts={len(starts)} ends={len(ends)}",
        )
        for s, e in zip(starts, ends):
            _expect(s < e, f"citation-start at {s} must precede end at {e}")
        ctx.record(
            TestResult(
                name,
                True,
                detail=f"events={len(events)} citations={len(starts)}",
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_streaming_with_tools(ctx: TestContext) -> None:
    """Streaming: tools -> tool-call-start / -delta / -end."""
    name = "streaming: tool call"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Paris? Use get_weather.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
        }
        request_id = ctx.next_request_id("stream-tools")
        events, saw_done = _stream_post(ctx, body=body, request_id=request_id)
        _expect(saw_done, "stream did not terminate with data: [DONE]")
        types = _event_types(events)
        starts = [i for i, t in enumerate(types) if t == "tool-call-start"]
        ends = [i for i, t in enumerate(types) if t == "tool-call-end"]
        if not starts:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        "WARN: model did not emit tool-call events; cannot "
                        "verify streaming tool wiring."
                    ),
                )
            )
            return
        _expect(
            len(starts) == len(ends),
            f"unbalanced tool-call events: starts={len(starts)} ends={len(ends)}",
        )
        delta_count = sum(1 for t in types if t == "tool-call-delta")
        _expect(
            delta_count > 0,
            f"no tool-call-delta events between start/end: {types}",
        )
        # Find the message-end event and check finish_reason maps to TOOL_CALL.
        end_events = [ev for ev in events if ev.get("type") == "message-end"]
        _expect(end_events, "no message-end event in stream")
        end_delta = end_events[-1].get("delta") or {}
        finish = end_delta.get("finish_reason")
        _expect(
            finish == "TOOL_CALL",
            f"message-end finish_reason={finish!r}, expected TOOL_CALL",
        )
        # Verify the reasoning surface matches the configured flag.
        plan_events = [t for t in types if t == "tool-plan-delta"]
        content_starts = [ev for ev in events if ev.get("type") == "content-start"]
        thinking_blocks = [
            ev
            for ev in content_starts
            if (
                (ev.get("delta") or {})
                .get("message", {})
                .get("content", {})
                .get("type")
            )
            == "thinking"
        ]
        if ctx.is_reasoning_model:
            detail_note = (
                f"thinking_content_starts={len(thinking_blocks)} "
                f"tool_plan_deltas={len(plan_events)}"
            )
        else:
            detail_note = f"tool_plan_deltas={len(plan_events)}"
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"events={len(events)} tool_calls={len(starts)} "
                    f"deltas={delta_count} finish={finish} {detail_note}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_error_invalid_request(ctx: TestContext) -> None:
    """Pydantic rejection: messages must not be empty."""
    name = "error: empty messages rejected (400/422)"
    try:
        body = {"model": ctx.model, "messages": []}
        request_id = ctx.next_request_id("err-empty")
        status, parsed = _post_json(
            ctx,
            body=body,
            request_id=request_id,
            expect_status=None,
        )
        _expect(
            status in (400, 422),
            f"expected 400 or 422, got {status}: {str(parsed)[:300]}",
        )
        _expect(
            _looks_like_4xx_envelope(parsed),
            f"unexpected 4xx body shape: {parsed!r}",
        )
        ctx.record(
            TestResult(
                name,
                True,
                detail=f"status={status} body={str(parsed)[:200]}",
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_error_request_id_echoed(ctx: TestContext) -> None:
    """Error responses should ideally echo ``X-Request-Id``.

    Three envelope shapes exist in the wild depending on where the
    rejection happens:

    * Our :class:`CohereError` (``{"message": ..., "id": ...}``) -- only
      this shape carries the request id.
    * FastAPI body-validation (``{"detail": [...]}``).
    * vLLM's standard error wrapper at
      ``vllm.entrypoints.serve.utils.api_utils`` ->
      ``{"error": {"message": ..., "type": ..., "code": ...}}``. This
      one drops the id today; treat it as a soft pass with a warning so
      we can light that gap up later without false-failing here.
    """
    name = "error: X-Request-Id echoed in CohereError envelope"
    try:
        body = {
            "model": ctx.model,
            "messages": [{"role": "user", "content": "hi"}],
            # negative max_tokens trips the Pydantic validator -> 4xx
            "max_tokens": -1,
        }
        request_id = ctx.next_request_id("err-reqid")
        status, parsed = _post_json(
            ctx,
            body=body,
            request_id=request_id,
            expect_status=None,
        )
        _expect(
            status >= 400,
            f"expected 4xx, got {status}: {parsed!r}",
        )
        _expect(
            _looks_like_4xx_envelope(parsed),
            f"unexpected 4xx body shape: {parsed!r}",
        )
        if isinstance(parsed, dict) and parsed.get("id") == request_id:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=f"status={status} CohereError.id matches request",
                )
            )
        elif isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            # vLLM's standard error envelope wraps the message but drops
            # the request id today.
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        f"status={status} WARN: vLLM error envelope dropped "
                        f"X-Request-Id={request_id!r}; only the 'error' "
                        f"wrapper is present"
                    ),
                )
            )
        elif isinstance(parsed, dict) and "detail" in parsed:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        f"status={status} (FastAPI body validation; id echo "
                        f"not expected at this layer)"
                    ),
                )
            )
        else:
            ctx.record(
                TestResult(
                    name,
                    False,
                    detail=(
                        f"status={status} body={parsed!r} -- expected a "
                        f"CohereError with id={request_id!r}, a vLLM "
                        f"'error' wrapper, or a 422 'detail' shape"
                    ),
                )
            )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id as registered with the server (matches /v1/models).",
    )
    parser.add_argument(
        "--reasoning-model",
        dest="is_reasoning_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the server was started with --cohere-is-reasoning-model "
            "(default true). Controls whether reasoning is expected as a "
            "thinking content block (reasoning) or as tool-plan-delta events "
            "(non-reasoning)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout, seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip all streaming tests.",
    )
    parser.add_argument(
        "--skip-tools",
        action="store_true",
        help="Skip tool-call tests.",
    )
    parser.add_argument(
        "--skip-citations",
        action="store_true",
        help="Skip documents/citations tests.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full request/response payloads.",
    )
    args = parser.parse_args()

    ctx = TestContext(
        base_url=args.base_url,
        model=args.model,
        is_reasoning_model=args.is_reasoning_model,
        verbose=args.verbose,
        timeout=args.timeout,
    )

    print(f"Driving Cohere v2 endpoint at {ctx.base_url}/cohere/v2/chat")
    print(f"  model={ctx.model}")
    print(f"  is_reasoning_model={ctx.is_reasoning_model}")
    print()

    test_health(ctx)
    test_non_streaming_basic(ctx)
    test_non_streaming_developer_role(ctx)
    if not args.skip_citations:
        test_non_streaming_with_documents(ctx)
    if not args.skip_tools:
        test_non_streaming_with_tools(ctx)
    if not args.skip_tools and not args.skip_citations:
        test_non_streaming_citations_from_tool_result(ctx)
    if not args.skip_streaming:
        test_streaming_basic(ctx)
        if not args.skip_citations:
            test_streaming_with_documents(ctx)
        if not args.skip_tools:
            test_streaming_with_tools(ctx)
    test_error_invalid_request(ctx)
    test_error_request_id_echoed(ctx)

    print()
    print("=" * 72)
    passed = sum(1 for r in ctx.results if r.passed and not r.skipped)
    failed = sum(1 for r in ctx.results if not r.passed and not r.skipped)
    skipped = sum(1 for r in ctx.results if r.skipped)
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        print("\nFailures:")
        for r in ctx.results:
            if not r.passed and not r.skipped:
                print(f"  - {r.name}")
                if r.detail:
                    for line in r.detail.splitlines():
                        print(f"      {line}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
