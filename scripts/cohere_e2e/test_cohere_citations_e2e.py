#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test for Cohere Chat v2 documents/citations behavior.

Drives a live vLLM server with a real Cohere Command-family model, focused
specifically on grounding: multi-shape ``documents``, ``citation_options``
modes, citing prior tool results, citations coexisting with ``thinking``,
and citations already present in the conversation history.

This complements ``scripts/test_cohere_v2_e2e.py`` (which covers the
broader v2 surface -- tools, streaming lifecycle, error paths -- with only
a couple of basic grounding checks). This script instead ports one test
case per relevant grounding scenario recorded in blobheart's dory replay
suite (``go/dory/pkg/replay/data_replay/``), so the citation surface has
the same scenario coverage server-side that dory already has.

Cases NOT ported, and why:

* ``chat_documents_non_stream`` / ``chat_documents_stream`` /
  ``chat_citations_fast`` / ``chat_citations_off`` / ``cmd3_plan_citation``
  -- these exercise dory's v1 ``/v1/chat`` API (``citation_quality``
  string field). vLLM only exposes the v2 ``/cohere/v2/chat`` endpoint;
  the same grounding behavior is exercised here via v2's
  ``citation_options.mode`` instead (see ``test_citation_options_fast``
  and ``test_citation_options_off``).
* ``debugging_rag`` / ``debugging_rag_stream`` -- exercise dory's
  ``enable_debugging`` request field, which has no vLLM equivalent.
* ``return_prompt_rag`` -- exercises dory's ``return_prompt`` field,
  which has no vLLM equivalent (there's no "echo the rendered prompt"
  knob on ``CohereChatV2Request``).
* ``tool_multihop_citations_off`` / ``tool_calls_result`` /
  ``tool_multihop_3`` / ``v2_tool_multihop_3`` -- primarily tool
  orchestration tests (multihop tool calling); already covered by the
  tool-call tests in ``test_cohere_v2_e2e.py``, so not duplicated here.
* ``cmda_thinking_chat_citations_in_history_from_turns`` /
  ``..._turns`` -- the latter is an incomplete/empty recording upstream
  in dory (WIP); the former is materially similar to
  ``cmda_thinking_chat_citations_in_history`` (ported below as
  ``test_citations_in_conversation_history``), just with citations
  attached to a ``thinking`` block instead of ``text`` blocks.

Prerequisites on the GPU host
-----------------------------

1. Install the optional Cohere SDKs:

       uv pip install cohere cohere-melody

2. Run this script. If ``--base-url`` isn't already serving a healthy
   ``/health`` response, the script launches ``vllm serve`` itself with
   the Cohere renderer / tokenizer wired up (``VLLM_ENABLE_COHERE_API=1``,
   ``--tokenizer-mode cohere``, ``--enable-auto-tool-choice``,
   ``--tool-call-parser cohere_command4``,
   ``--reasoning-parser cohere_command4``), waits
   for it to become healthy, then runs the test suite:

       python scripts/test_cohere_citations_e2e.py

   defaults to the smallest available Command A+ checkpoint,
   ``CohereLabs/command-a-plus-05-2026-w4a4``. Pass ``--model`` to use a
   different one, or point ``--base-url`` at a server you started
   yourself (e.g. in another terminal, per the manual invocation below)
   and pass ``--no-auto-start-server`` to make that mandatory instead of
   just preferred:

       VLLM_ENABLE_COHERE_API=1 vllm serve <cohere-model-id> \\
           --tokenizer-mode cohere \\
           --enable-auto-tool-choice \\
           --tool-call-parser cohere_command4 \\
           --reasoning-parser cohere_command4 \\
           --port 8000

   The parser name must match the checkpoint's prompt generation: use
   ``cohere_command3`` for cmd3-generation models (and
   ``--cohere-parser cohere_command3`` when this script owns the server).

   For non-reasoning Command models (cmd3, older Command R), pass
   ``--no-reasoning-model`` to this script; when it owns the server
   process, that also adds ``--no-cohere-is-reasoning-model`` to the
   ``vllm serve`` invocation.

Optional flags
--------------

* ``--reasoning-model / --no-reasoning-model`` -- whether the server was
  (or should be) launched with reasoning enabled. Controls whether the
  ``thinking`` cases run, and whether an auto-started server gets
  ``--no-cohere-is-reasoning-model``.
* ``--auto-start-server / --no-auto-start-server`` -- whether to launch
  ``vllm serve`` ourselves when ``--base-url`` isn't already up
  (default: on).
* ``--keep-server / --no-keep-server`` -- leave an auto-started server
  running after the tests finish, so re-runs skip the (often lengthy)
  model load (default: on -- the PID and log path are printed so you can
  stop it manually).
* ``--extra-server-arg`` -- repeatable; extra ``vllm serve`` args (e.g.
  ``--extra-server-arg=--tensor-parallel-size=8``) forwarded verbatim
  when we own the server process.
* ``--startup-timeout`` -- seconds to wait for an auto-started server to
  become healthy (default: 1800; large quantized checkpoints are slow to
  load).
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
from vllm_server import add_server_args, ensure_server, release_server

DONE_LINE = "data: [DONE]"

# Smallest publicly available Command A+ checkpoint at the time of
# writing; picked as the default so this script is runnable without
# having to hunt down a model id first.
DEFAULT_MODEL = "CohereLabs/command-a-plus-05-2026-w4a4"

# Reasoning models spend most of their decode budget inside ``thinking``
# blocks before emitting user-visible text/citations; give grounding
# requests a generous budget so citations actually get a chance to appear.
REASONING_BUDGET = 2048

PENGUIN_DOCUMENTS = [
    {"data": {"snippet": "The tallest penguin is the Emperor penguin"}},
    {"data": {"snippet": "The latin name for Emperor penguin is Aptenodytes forsteri"}},
    {"data": {"snippet": "The smallest penguin is the fairy penguin"}},
    {"data": {"snippet": "The latin name for fairy penguin is Eudyptula minor"}},
]


# ----------------------------------------------------------------------
# Helpers (mirrors scripts/test_cohere_v2_e2e.py)
# ----------------------------------------------------------------------


def _text_from_content_blocks(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _thinking_from_content_blocks(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("thinking", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "thinking"
    )


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
        return f"e2e-cite-{label}-{self.request_id_counter}-{int(time.time())}"

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
    url = f"{ctx.base_url.rstrip('/')}/cohere/v2/chat"
    headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
    with httpx.Client(timeout=ctx.timeout) as client:
        resp = client.post(url, json=body, headers=headers)
    if ctx.verbose:
        print(f"  -> POST {url}  [status {resp.status_code}]")
        print(f"     request_id={request_id}")
        print(f"     body={json.dumps(body)[:800]}")
        print(f"     resp={resp.text[:2000]}")
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


def _assert_citation_shape(citations: list[dict[str, Any]]) -> None:
    for c in citations:
        _expect("start" in c and "end" in c, f"citation missing span: {c}")
        sources = c.get("sources") or []
        _expect(sources, f"citation without sources: {c}")
        for src in sources:
            _expect("type" in src, f"citation source missing type: {src}")


def _no_grounding_warning(content: Any) -> str:
    text = _text_from_content_blocks(content) or ""
    thinking = _thinking_from_content_blocks(content) or ""
    return (
        "WARN: assistant did not ground its answer.\n"
        f"  text head: {text[:200]!r}\n"
        f"  thinking head: {thinking[:200]!r}\n"
        "  If the text contains no '<co' markers, the model never "
        "grounded. Re-run with VLLM_LOGGING_LEVEL=DEBUG to confirm via "
        "the 'cohere reasoning parser:' diagnostic log line."
    )


# ----------------------------------------------------------------------
# Test cases -- one per applicable dory replay case
# ----------------------------------------------------------------------


def test_documents_basic_non_stream(ctx: TestContext) -> None:
    """Basic grounded answer, non-streaming.

    Ports dory's ``v2_chat_documents_non_stream`` replay case: plain
    ``{"data": {...}}`` documents (no explicit ``id``), single user turn.
    """
    name = "documents: basic grounded answer (non-streaming)"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "What is the tallest penguin?"},
            ],
            "documents": PENGUIN_DOCUMENTS,
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("docs-basic")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        _expect(content, f"empty content: {message}")
        if not citations:
            ctx.record(TestResult(name, True, detail=_no_grounding_warning(content)))
            return
        _assert_citation_shape(citations)
        doc_ids = {
            s.get("id") or s.get("document", {}).get("id")
            for c in citations
            for s in c.get("sources", [])
        }
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"got {len(citations)} citation(s); doc ids referenced={doc_ids}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_documents_basic_streaming(ctx: TestContext) -> None:
    """Basic grounded answer, streaming.

    Ports dory's ``v2_chat_documents_stream`` replay case.
    """
    name = "documents: basic grounded answer (streaming)"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "What is the tallest penguin?"},
            ],
            "documents": PENGUIN_DOCUMENTS,
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("docs-basic-stream")
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
                    detail="WARN: model did not emit citation events for this stream.",
                )
            )
            return
        _expect(
            len(starts) == len(ends),
            f"unbalanced citation events: starts={len(starts)} ends={len(ends)}",
        )
        for s, e in zip(starts, ends):
            _expect(s < e, f"citation-start at {s} must precede end at {e}")
        detail = f"events={len(events)} citations={len(starts)}"
        ctx.record(TestResult(name, True, detail=detail))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_citation_options_fast(ctx: TestContext) -> None:
    """``citation_options.mode=FAST`` still grounds, via a streaming call.

    Ports dory's ``v2_chat_citations_fast`` (and v1 ``chat_citations_fast``,
    which is the same scenario against the ``/v1/chat`` API) replay cases.
    """
    name = "citation_options: mode=FAST grounds the answer (streaming)"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "What is the tallest penguin?"},
            ],
            "documents": PENGUIN_DOCUMENTS,
            "citation_options": {"mode": "FAST"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("cite-fast")
        events, saw_done = _stream_post(ctx, body=body, request_id=request_id)
        _expect(saw_done, "stream did not terminate with data: [DONE]")
        types = _event_types(events)
        starts = [i for i, t in enumerate(types) if t == "citation-start"]
        if not starts:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail="WARN: mode=FAST did not produce citation events.",
                )
            )
            return
        ctx.record(TestResult(name, True, detail=f"citations={len(starts)}"))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_citation_options_off(ctx: TestContext) -> None:
    """``citation_options.mode=OFF`` must suppress citation generation.

    Ports dory's ``v2_chat_citations_off`` (and v1 ``chat_citations_off``)
    replay cases, where the recorded golden response has an *empty*
    ``citations`` array/no citation events despite grounding documents
    being present.
    """
    name = "citation_options: mode=OFF suppresses citations (streaming)"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "What is the tallest penguin?"},
            ],
            "documents": PENGUIN_DOCUMENTS,
            "citation_options": {"mode": "OFF"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("cite-off")
        events, saw_done = _stream_post(ctx, body=body, request_id=request_id)
        _expect(saw_done, "stream did not terminate with data: [DONE]")
        types = _event_types(events)
        _expect(
            "citation-start" not in types,
            f"mode=OFF but citation-start events were emitted: {types}",
        )
        text_chunks = [
            (ev.get("delta") or {}).get("message", {}).get("content", {}).get("text")
            for ev in events
            if ev.get("type") == "content-delta"
        ]
        text = "".join(t for t in text_chunks if isinstance(t, str))
        ctx.record(TestResult(name, True, detail=f"no citation events; text={text!r}"))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_mixed_document_shapes(ctx: TestContext) -> None:
    """Documents can be a ``{id, data}`` object, a bare string, or a JSON
    string all in the same request; citations should be able to point at
    any of them.

    Ports dory's ``cmd3_documents`` replay case.
    """
    name = "documents: mixed shapes (object / plain string / JSON string)"
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the tallest mountain on Mars and Venus?",
                },
            ],
            "documents": [
                {
                    "id": "mars_1",
                    "data": {
                        "mountain": "Olympus Mons",
                        "location": "Mars",
                        "height": 21088,
                    },
                },
                "Skadi Mons is the tallest mountain on Venus",
                '{"location": "Earth", "mountain": "Mount Everest", "height": 29029}',
            ],
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("mixed-docs")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        _expect(content, f"empty content: {message}")
        if not citations:
            ctx.record(TestResult(name, True, detail=_no_grounding_warning(content)))
            return
        _assert_citation_shape(citations)
        doc_ids = {
            s.get("id") or s.get("document", {}).get("id")
            for c in citations
            for s in c.get("sources", [])
        }
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"got {len(citations)} citation(s) across mixed doc shapes; "
                    f"doc ids referenced={doc_ids}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_grounding_against_tool_result_in_history(ctx: TestContext) -> None:
    """Ask the model to retroactively cite an answer it already gave,
    grounding against a tool result from earlier in the conversation.

    Ports dory's ``cmd3_cite_old_tool_result`` replay case: the history
    already contains a completed ``internet_search`` tool call/result and
    an *ungrounded* assistant answer; the new user turn asks for the same
    answer "with grounding", which should produce citations against the
    tool's document-shaped result (and/or the top-level ``documents``).
    """
    name = "documents: ground a prior answer against a tool result in history"
    try:
        body = {
            "model": ctx.model,
            "documents": ["The tallest mountain is mount everest"],
            "messages": [
                {
                    "role": "user",
                    "content": "What are the two tallest mountains?",
                },
                {
                    "role": "assistant",
                    "tool_plan": ("I will search for the second tallest mountain."),
                    "tool_calls": [
                        {
                            "id": "internet_search_0123",
                            "type": "function",
                            "function": {
                                "name": "internet_search",
                                "arguments": json.dumps(
                                    {"query": "second tallest mountain"}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "internet_search_0123",
                    "content": [
                        {
                            "type": "document",
                            "document": {
                                "data": {"result": "The second tallest mountain is K2."}
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "The two tallest mountains are Mount Everest and K2",
                },
                {
                    "role": "user",
                    "content": "Great. Can you repeat that with grounding?",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "internet_search",
                        "description": "Searches the internet",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "The query to search the internet with"
                                    ),
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("cite-old-tool-result")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        tool_calls = message.get("tool_calls") or []
        # The model might legitimately re-issue the search instead of
        # citing the existing result -- treat that as informative, not a
        # hard failure, since the wiring under test is "does grounding
        # against history work at all", not "does the model always skip
        # redundant tool calls".
        if tool_calls and not citations:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        "WARN: model re-issued a tool call instead of citing "
                        f"history: {tool_calls}"
                    ),
                )
            )
            return
        _expect(content, f"empty content: {message}")
        if not citations:
            ctx.record(TestResult(name, True, detail=_no_grounding_warning(content)))
            return
        _assert_citation_shape(citations)
        source_types = {s.get("type") for c in citations for s in c.get("sources", [])}
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"got {len(citations)} citation(s); source types={source_types} "
                    "(expect at least one 'tool' or 'document' source)"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_thinking_with_citations_non_stream(ctx: TestContext) -> None:
    """Reasoning (``thinking.type=enabled``) and grounding must coexist.

    Ports dory's ``cmda_thinking_with_citations`` replay case.
    """
    name = "documents: thinking + citations coexist (non-streaming)"
    if not ctx.is_reasoning_model:
        ctx.record(
            TestResult(
                name, True, skipped=True, detail="server is not a reasoning model"
            )
        )
        return
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "What is the tallest penguin?"},
            ],
            "documents": PENGUIN_DOCUMENTS,
            "citation_options": {"mode": "FAST"},
            "thinking": {"type": "enabled"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("thinking-cite")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        thinking = _thinking_from_content_blocks(content)
        _expect(content, f"empty content: {message}")
        if not citations:
            ctx.record(
                TestResult(
                    name,
                    True,
                    detail=(
                        f"WARN: no citations; thinking_chars={len(thinking)}\n"
                        + _no_grounding_warning(content)
                    ),
                )
            )
            return
        _assert_citation_shape(citations)
        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"got {len(citations)} citation(s); thinking_chars={len(thinking)}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_thinking_with_citations_streaming(ctx: TestContext) -> None:
    """Streaming variant: thinking content-blocks must not disrupt the
    citation-start/-end event pairing.

    Ports dory's ``cmda_thinking_stream_with_citations`` replay case.
    """
    name = "documents: thinking + citations coexist (streaming)"
    if not ctx.is_reasoning_model:
        ctx.record(
            TestResult(
                name, True, skipped=True, detail="server is not a reasoning model"
            )
        )
        return
    try:
        body = {
            "model": ctx.model,
            "messages": [
                {"role": "user", "content": "What is the tallest penguin?"},
            ],
            "documents": PENGUIN_DOCUMENTS,
            "citation_options": {"mode": "FAST"},
            "thinking": {"type": "enabled"},
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("thinking-cite-stream")
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
                    detail=f"WARN: no citation events; total events={len(events)}",
                )
            )
            return
        _expect(
            len(starts) == len(ends),
            f"unbalanced citation events: starts={len(starts)} ends={len(ends)}",
        )
        for s, e in zip(starts, ends):
            _expect(s < e, f"citation-start at {s} must precede end at {e}")
        detail = f"events={len(events)} citations={len(starts)}"
        ctx.record(TestResult(name, True, detail=detail))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_citations_in_conversation_history(ctx: TestContext) -> None:
    """A prior assistant turn already carries a populated ``citations``
    array (with both ``document`` and ``tool`` sources, one of them
    anchored to a ``PLAN``/thinking content index); the server must accept
    that history and continue grounding on the follow-up turn.

    Ports dory's ``cmda_thinking_chat_citations_in_history`` replay case.
    """
    name = "documents: history already contains citations; follow-up still grounds"
    if not ctx.is_reasoning_model:
        ctx.record(
            TestResult(
                name, True, skipped=True, detail="server is not a reasoning model"
            )
        )
        return
    try:
        body = {
            "model": ctx.model,
            "documents": [
                {
                    "data": {
                        "title": "Weather in Tokyo",
                        "snippet": "The weather in tokyo is 27 degrees",
                    }
                }
            ],
            "thinking": {"type": "enabled"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "What is the weather in London, Toronto, and Tokyo?"
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I will use the get_weather tool",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "tool_call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Toronto"}),
                            },
                        },
                        {
                            "id": "tool_call_2",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "London"}),
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_call_1",
                    "content": [
                        {"type": "text", "text": "it's colder than usual"},
                        {"type": "text", "text": "10 degrees"},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_call_2",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"id": "test_res_id", "degrees": 25}),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "The user asked about London, Toronto, and Tokyo. "
                                "I already have results for London and Toronto from "
                                "the tool calls, and Tokyo is covered by the "
                                "provided document."
                            ),
                        },
                        {
                            "type": "text",
                            "text": (
                                "The weather in London is 25 degrees. The weather "
                                "in Toronto is 10 degrees. The weather in Tokyo is "
                                "27 degrees."
                            ),
                        },
                    ],
                    "citations": [
                        {
                            "start": 25,
                            "end": 36,
                            "text": "25 degrees.",
                            "type": "TEXT_CONTENT",
                            "content_index": 1,
                            "sources": [
                                {
                                    "type": "tool",
                                    "id": "test_res_id",
                                    "tool_output": {"degrees": "25"},
                                }
                            ],
                        },
                        {
                            "start": 63,
                            "end": 74,
                            "text": "10 degrees.",
                            "type": "TEXT_CONTENT",
                            "content_index": 1,
                            "sources": [
                                {
                                    "type": "tool",
                                    "id": "tool_call_1:1",
                                    "tool_output": {"content": "10 degrees"},
                                }
                            ],
                        },
                        {
                            "start": 99,
                            "end": 110,
                            "text": "27 degrees.",
                            "type": "TEXT_CONTENT",
                            "content_index": 1,
                            "sources": [
                                {
                                    "type": "document",
                                    "id": "doc:0",
                                    "document": {
                                        "id": "doc:0",
                                        "title": "Weather in Tokyo",
                                        "snippet": "The weather in tokyo is 27 degrees",
                                    },
                                }
                            ],
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Which city is warmest? Please cite your result as "
                                "described in the Grounding section"
                            ),
                        }
                    ],
                },
            ],
            "max_tokens": REASONING_BUDGET,
            "temperature": 0.0,
            "seed": 0,
        }
        request_id = ctx.next_request_id("cites-in-history")
        _, resp = _post_json(ctx, body=body, request_id=request_id)
        assert isinstance(resp, dict)
        message = resp.get("message") or {}
        content = message.get("content") or []
        citations = message.get("citations") or []
        text = _text_from_content_blocks(content)
        _expect(content, f"empty content: {message}")
        # "London" (25 degrees) should be identified as the warmest city.
        _expect(
            "london" in text.lower() or "25" in text,
            f"expected the warmest-city answer to reference London/25 degrees: "
            f"{text!r}",
        )
        if not citations:
            ctx.record(TestResult(name, True, detail=_no_grounding_warning(content)))
            return
        _assert_citation_shape(citations)
        ctx.record(
            TestResult(
                name,
                True,
                detail=f"answer={text!r} citations={len(citations)}",
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_server_args(parser, default_model=DEFAULT_MODEL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout, seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full request/response payloads.",
    )
    args = parser.parse_args()

    managed_server = ensure_server(args, log_prefix="vllm-cohere-citations-e2e-")

    try:
        ctx = TestContext(
            base_url=args.base_url,
            model=args.model,
            is_reasoning_model=args.is_reasoning_model,
            verbose=args.verbose,
            timeout=args.timeout,
        )

        print(
            f"Driving Cohere v2 documents/citations tests at "
            f"{ctx.base_url}/cohere/v2/chat"
        )
        print(f"  model={ctx.model}")
        print(f"  is_reasoning_model={ctx.is_reasoning_model}")
        print()

        test_documents_basic_non_stream(ctx)
        test_documents_basic_streaming(ctx)
        test_citation_options_fast(ctx)
        test_citation_options_off(ctx)
        test_mixed_document_shapes(ctx)
        test_grounding_against_tool_result_in_history(ctx)
        test_thinking_with_citations_non_stream(ctx)
        test_thinking_with_citations_streaming(ctx)
        test_citations_in_conversation_history(ctx)

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
    finally:
        release_server(managed_server, keep=args.keep_server)


if __name__ == "__main__":
    sys.exit(main())
