#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the Cohere v2 output conversion paths.

Drives ``CohereServingChatV2._chat_completion_to_v2`` (non-streaming) and
``CohereServingChatV2._chat_completion_stream_to_v2`` (streaming) with
synthetic upstream chat-completion responses, including reasoning,
tool calls, and citations. No engine boot, no model download.

Run with:

    .venv/bin/python scripts/test_cohere_output.py
    .venv/bin/python scripts/test_cohere_output.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator

from pydantic import ValidationError

from vllm.entrypoints.cohere.cohere_chat_message import (
    Citation,
    CitationSource,
    CohereChatMessage,
    CohereDeltaMessage,
)
from vllm.entrypoints.cohere.protocol import CohereChatV2Request
from vllm.entrypoints.cohere.serving import CohereServingChatV2
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    PromptTokenUsageInfo,
    UsageInfo,
)


def _make_handler(
    is_reasoning_model: bool = True,
) -> CohereServingChatV2:
    """Build a converter without going through ``__init__``.

    The non-streaming and streaming converters only touch a handful of
    instance attrs (currently just ``_is_reasoning_model``), which we set
    here so we don't need to spin up the full handler.
    """
    handler = CohereServingChatV2.__new__(CohereServingChatV2)
    handler._is_reasoning_model = is_reasoning_model
    return handler


def _make_request() -> CohereChatV2Request:
    return CohereChatV2Request.model_validate(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Who wrote Hamlet, and when?"},
            ],
            "documents": [
                {
                    "id": "doc_0",
                    "data": {
                        "title": "Wikipedia: Hamlet",
                        "text": (
                            "Hamlet was written by William Shakespeare around 1600."
                        ),
                    },
                },
            ],
            "citation_options": {"mode": "ACCURATE"},
        }
    )


# ----------------------------------------------------------------------
# Non-streaming
# ----------------------------------------------------------------------


def test_non_streaming(verbose: bool) -> None:
    print("=" * 72)
    print("non-streaming: ChatCompletionResponse -> CohereChatV2Response")
    print("=" * 72)

    handler = _make_handler()
    request = _make_request()

    citation = Citation(
        start=22,
        end=41,
        text="William Shakespeare",
        sources=[
            CitationSource(
                type="document",
                id="doc_0",
                document={
                    "title": "Wikipedia: Hamlet",
                    "text": "Hamlet was written by William Shakespeare around 1600.",
                },
            )
        ],
        content_index=1,
        type="TEXT_CONTENT",
    )

    msg = CohereChatMessage(
        role="assistant",
        content="Hamlet was written by William Shakespeare around 1600.",
        reasoning="The user asked about Hamlet's authorship and date.",
        citations=[citation],
    )

    response = ChatCompletionResponse(
        id="chatcmpl-abc123",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(index=0, message=msg, finish_reason="stop")
        ],
        usage=UsageInfo(prompt_tokens=12, completion_tokens=20, total_tokens=32),
    )

    v2 = handler._chat_completion_to_v2(response, request)
    payload = json.loads(v2.model_dump_json(exclude_none=True))

    if verbose:
        print(json.dumps(payload, indent=2))
        print()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert payload["id"] == "chatcmpl-abc123", payload["id"]
    assert payload["finish_reason"] == "COMPLETE", payload["finish_reason"]

    message = payload["message"]
    assert message["role"] == "assistant"

    block_types = [b["type"] for b in message["content"]]
    assert block_types == ["thinking", "text"], block_types
    assert (
        message["content"][0]["thinking"]
        == "The user asked about Hamlet's authorship and date."
    )
    assert (
        message["content"][1]["text"]
        == "Hamlet was written by William Shakespeare around 1600."
    )

    cits = message["citations"]
    assert len(cits) == 1
    c = cits[0]
    assert c["text"] == "William Shakespeare"
    assert c["start"] == 22 and c["end"] == 41
    assert c["sources"][0]["id"] == "doc_0"
    assert c["sources"][0]["type"] == "document"

    usage = payload["usage"]
    assert usage["billed_units"]["input_tokens"] == 12
    assert usage["billed_units"]["output_tokens"] == 20
    assert usage["tokens"]["input_tokens"] == 12
    assert usage["tokens"]["output_tokens"] == 20

    print("OK: thinking + text content blocks")
    print("OK: 1 citation (William Shakespeare -> doc_0)")
    print("OK: finish_reason mapped 'stop' -> 'COMPLETE'")
    print("OK: usage billed_units / tokens populated")
    print()


def _build_tool_call_response() -> ChatCompletionResponse:
    """Build a tool-call response with reasoning attached, for the two
    parallel test cases below.
    """
    from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall

    msg = ChatMessage(
        role="assistant",
        content=None,
        reasoning="I should look up Hamlet's authorship in the knowledge base.",
        tool_calls=[
            ToolCall(
                id="call_1",
                function=FunctionCall(
                    name="lookup",
                    arguments='{"query": "Hamlet authorship"}',
                ),
            )
        ],
    )

    return ChatCompletionResponse(
        id="chatcmpl-tool-1",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=msg, finish_reason="tool_calls"
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=15, total_tokens=25),
    )


def test_non_streaming_with_tool_call_reasoning_default(verbose: bool) -> None:
    """Default (reasoning-model) behavior: reasoning is kept as a thinking
    content block alongside the tool calls; ``tool_plan`` is never set.
    """
    print("=" * 72)
    print("non-streaming: tool call (default, reasoning -> thinking block)")
    print("=" * 72)

    handler = _make_handler(is_reasoning_model=True)
    request = _make_request()
    response = _build_tool_call_response()

    v2 = handler._chat_completion_to_v2(response, request)
    payload = json.loads(v2.model_dump_json(exclude_none=True))

    if verbose:
        print(json.dumps(payload, indent=2))
        print()

    assert payload["finish_reason"] == "TOOL_CALL"
    message = payload["message"]
    assert "tool_plan" not in message or message["tool_plan"] is None
    assert message["content"] is not None
    block_types = [b["type"] for b in message["content"]]
    assert block_types == ["thinking"], block_types
    assert message["content"][0]["thinking"] == (
        "I should look up Hamlet's authorship in the knowledge base."
    )
    assert len(message["tool_calls"]) == 1

    print("OK: thinking block emitted alongside tool_calls")
    print("OK: tool_plan is not set (reasoning model assumption)")
    print()


def test_non_streaming_with_tool_call_tool_plan_flag(verbose: bool) -> None:
    """Flag-enabled (non-reasoning-model) behavior: reasoning is surfaced
    as ``tool_plan`` and no thinking content block is emitted.
    """
    print("=" * 72)
    print("non-streaming: tool call (non-reasoning, reasoning -> tool_plan)")
    print("=" * 72)

    handler = _make_handler(is_reasoning_model=False)
    request = _make_request()
    response = _build_tool_call_response()

    v2 = handler._chat_completion_to_v2(response, request)
    payload = json.loads(v2.model_dump_json(exclude_none=True))

    if verbose:
        print(json.dumps(payload, indent=2))
        print()

    assert payload["finish_reason"] == "TOOL_CALL"
    message = payload["message"]
    assert message.get("content") is None or message["content"] == []
    assert message["tool_plan"] == (
        "I should look up Hamlet's authorship in the knowledge base."
    )
    assert len(message["tool_calls"]) == 1
    tc = message["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "lookup"
    assert json.loads(tc["function"]["arguments"]) == {"query": "Hamlet authorship"}

    print("OK: reasoning surfaced as tool_plan, no thinking block emitted")
    print("OK: tool_calls preserved with id + function.name + arguments")
    print()


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


def _wrap(chunk: ChatCompletionStreamResponse) -> str:
    """Format a chunk like it would appear on the upstream OpenAI SSE stream."""
    return f"data: {chunk.model_dump_json()}\n\n"


def _stream_chunk(
    delta_kwargs: dict | None = None,
    finish_reason: str | None = None,
    chunk_id: str = "chatcmpl-stream-1",
) -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        id=chunk_id,
        model="test-model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=CohereDeltaMessage(**(delta_kwargs or {})),
                finish_reason=finish_reason,
            )
        ],
    )


async def _fake_upstream() -> AsyncIterator[str]:
    """Simulate vLLM's upstream chat-completion SSE stream.

    Order chosen to exercise every block-transition path:

      role -> reasoning -> text -> citation -> tool_call -> usage-only -> [DONE]
    """
    # Initial role-only delta.
    yield _wrap(_stream_chunk({"role": "assistant"}))

    # Reasoning -> emits a 'thinking' content block.
    yield _wrap(_stream_chunk({"reasoning": "Thinking about Hamlet..."}))
    yield _wrap(_stream_chunk({"reasoning": " It's Shakespeare."}))

    # Visible text -> closes thinking block, opens a text block.
    yield _wrap(_stream_chunk({"content": "Hamlet was written by "}))
    yield _wrap(_stream_chunk({"content": "William Shakespeare."}))

    # Citation grounding the text we just emitted.
    citation = Citation(
        start=22,
        end=41,
        text="William Shakespeare",
        sources=[CitationSource(type="document", id="doc_0")],
        content_index=1,
        type="TEXT_CONTENT",
    )
    yield _wrap(_stream_chunk({"citations": [citation]}))

    # Tool call -> closes text block, opens tool_call block.
    yield _wrap(
        _stream_chunk(
            {
                "tool_calls": [
                    DeltaToolCall(
                        id="call_0",
                        type="function",
                        index=0,
                        function=DeltaFunctionCall(name="lookup", arguments=""),
                    )
                ]
            }
        )
    )
    # Last delta-bearing chunk also carries the OpenAI ``finish_reason``,
    # mirroring real upstream behavior. The very last chunk is usage-only.
    yield _wrap(
        _stream_chunk(
            delta_kwargs={
                "tool_calls": [
                    DeltaToolCall(
                        index=0,
                        function=DeltaFunctionCall(arguments='{"q":"Hamlet"}'),
                    )
                ]
            },
            finish_reason="tool_calls",
        )
    )

    # Final, choices-empty chunk carries usage only.
    final_chunk = ChatCompletionStreamResponse(
        id="chatcmpl-stream-1",
        model="test-model",
        choices=[],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=15, total_tokens=25),
    )
    yield _wrap(final_chunk)
    yield "data: [DONE]\n\n"


async def test_streaming(verbose: bool) -> None:
    print("=" * 72)
    print("streaming: default (reasoning -> thinking content blocks)")
    print("=" * 72)

    handler = _make_handler(is_reasoning_model=True)
    request = _make_request()

    events: list[dict] = []
    frames: list[str] = []
    async for sse in handler._chat_completion_stream_to_v2(_fake_upstream(), request):
        assert sse.startswith("data: "), repr(sse)
        frames.append(sse)
        body = sse[len("data: ") :].strip()
        if not body or body == "[DONE]":
            continue
        events.append(json.loads(body))

    # Cohere v2 streams must terminate with ``data: [DONE]\n\n``.
    assert frames[-1] == "data: [DONE]\n\n", repr(frames[-1])

    if verbose:
        for ev in events:
            print(json.dumps(ev, indent=2))
            print()
    else:
        for ev in events:
            t = ev.get("type", "?")
            extra = ""
            if t in ("content-start", "content-end"):
                extra = f"  index={ev.get('index')}"
            elif t == "content-delta":
                content = ev.get("delta", {}).get("message", {}).get("content", {})
                extra = f"  index={ev.get('index')}  delta={content!r}"
            elif t in (
                "tool-call-start",
                "tool-call-delta",
                "tool-call-end",
                "citation-start",
                "citation-end",
            ):
                extra = f"  index={ev.get('index')}"
            elif t == "message-end":
                extra = f"  finish_reason={ev.get('delta', {}).get('finish_reason')}"
            print(f"  {t}{extra}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    types = [ev["type"] for ev in events]

    # Lifecycle bookends.
    assert types[0] == "message-start", types[0]
    assert types[-1] == "message-end", types[-1]

    # Each kind of event we care about appears at least once.
    for required in (
        "content-start",
        "content-delta",
        "content-end",
        "tool-call-start",
        "tool-call-delta",
        "tool-call-end",
        "citation-start",
        "citation-end",
    ):
        assert required in types, f"missing event type: {required}"

    # Block-ordering invariants:
    # thinking opens before any text is opened, text opens before tool_call.
    def _content_start_kind(ev: dict) -> str | None:
        c = ev.get("delta", {}).get("message", {}).get("content")
        return c.get("type") if isinstance(c, dict) else None

    thinking_start = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "content-start" and _content_start_kind(e) == "thinking"
    )
    text_start = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "content-start" and _content_start_kind(e) == "text"
    )
    tool_start = types.index("tool-call-start")
    assert thinking_start < text_start < tool_start, (
        thinking_start,
        text_start,
        tool_start,
    )

    # The text block has to be closed before the tool block opens.
    text_block_index = events[text_start].get("index")
    text_end_idx = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "content-end" and e.get("index") == text_block_index
    )
    assert text_end_idx < tool_start

    # citation-start / citation-end must come as a pair, in order.
    cit_starts = [i for i, t in enumerate(types) if t == "citation-start"]
    cit_ends = [i for i, t in enumerate(types) if t == "citation-end"]
    assert len(cit_starts) == len(cit_ends) == 1
    assert cit_starts[0] < cit_ends[0]
    cit_payload = events[cit_starts[0]]["delta"]["message"]["citations"]
    assert cit_payload["text"] == "William Shakespeare"
    assert cit_payload["sources"][0]["id"] == "doc_0"

    # tool-call-start carries id + function name; subsequent deltas carry args.
    tool_start_ev = events[tool_start]
    tc = tool_start_ev["delta"]["message"]["tool_calls"]
    assert tc["id"] == "call_0"
    assert tc["function"]["name"] == "lookup"

    tool_delta_ev = next(e for e in events if e["type"] == "tool-call-delta")
    args = tool_delta_ev["delta"]["message"]["tool_calls"]["function"]["arguments"]
    assert args == '{"q":"Hamlet"}'

    # message-end: finish_reason + usage.
    end = events[-1]
    assert end["delta"]["finish_reason"] == "TOOL_CALL"
    usage = end["delta"]["usage"]
    assert usage["billed_units"]["input_tokens"] == 10
    assert usage["billed_units"]["output_tokens"] == 15
    assert usage["tokens"]["input_tokens"] == 10
    assert usage["tokens"]["output_tokens"] == 15

    print()
    print("OK: full event lifecycle (message-start ... message-end)")
    print("OK: thinking -> text -> tool_call block ordering")
    print("OK: citation-start / citation-end pair carries text + source")
    print("OK: tool-call-start carries id+name, deltas carry arg fragments")
    print("OK: message-end maps finish_reason and usage")
    print()


async def test_streaming_tool_plan(verbose: bool) -> None:
    """With ``is_reasoning_model=False`` (older non-reasoning Command
    model), reasoning chunks are emitted as ``tool-plan-delta`` events
    instead of opening a thinking content block.
    """
    print("=" * 72)
    print("streaming: non-reasoning (reasoning -> tool-plan-delta)")
    print("=" * 72)

    handler = _make_handler(is_reasoning_model=False)
    request = _make_request()

    events: list[dict] = []
    frames: list[str] = []
    async for sse in handler._chat_completion_stream_to_v2(_fake_upstream(), request):
        frames.append(sse)
        body = sse[len("data: ") :].strip()
        if body and body != "[DONE]":
            events.append(json.loads(body))

    assert frames[-1] == "data: [DONE]\n\n", repr(frames[-1])

    if verbose:
        for ev in events:
            print(json.dumps(ev, indent=2))
            print()
    else:
        for ev in events:
            print(f"  {ev.get('type', '?')}")

    types = [ev["type"] for ev in events]

    # No thinking content block ever opens.
    for ev in events:
        if ev["type"] == "content-start":
            content = ev.get("delta", {}).get("message", {}).get("content")
            kind = content.get("type") if isinstance(content, dict) else None
            assert kind != "thinking", (
                "thinking content-start should not be emitted when "
                "is_reasoning_model=False"
            )

    # ``tool-plan-delta`` events are emitted, with the reasoning text payload.
    plan_events = [e for e in events if e["type"] == "tool-plan-delta"]
    assert len(plan_events) >= 1
    plan_text = "".join(e["delta"]["message"]["tool_plan"] for e in plan_events)
    assert "Thinking about Hamlet" in plan_text
    assert "Shakespeare" in plan_text

    # The visible text and tool call still flow correctly.
    assert "content-start" in types  # for the text block
    assert "tool-call-start" in types
    assert types[-1] == "message-end"

    print()
    print("OK: reasoning chunks emitted as tool-plan-delta events")
    print("OK: no thinking content-start event in the stream")
    print("OK: text + tool call events flow normally")
    print()


# ----------------------------------------------------------------------
# Cached tokens (#11) + fallback message-end (#6)
# ----------------------------------------------------------------------


def test_non_streaming_cached_tokens(verbose: bool) -> None:
    """``UsageInfo.prompt_tokens_details.cached_tokens`` must flow through
    into ``CohereUsage.cached_tokens`` on the v2 response.
    """
    print("=" * 72)
    print("non-streaming: cached_tokens plumbed through usage")
    print("=" * 72)

    handler = _make_handler()
    request = _make_request()

    msg = ChatMessage(role="assistant", content="hi")
    response = ChatCompletionResponse(
        id="chatcmpl-cache",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(index=0, message=msg, finish_reason="stop")
        ],
        usage=UsageInfo(
            prompt_tokens=42,
            completion_tokens=8,
            total_tokens=50,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=37),
        ),
    )

    v2 = handler._chat_completion_to_v2(response, request)
    payload = json.loads(v2.model_dump_json(exclude_none=True))

    if verbose:
        print(json.dumps(payload, indent=2))
        print()

    usage = payload["usage"]
    assert usage["cached_tokens"] == 37, usage
    assert usage["billed_units"]["input_tokens"] == 42

    print("OK: usage.cached_tokens == 37")
    print()


async def _fake_upstream_with_cached_tokens() -> AsyncIterator[str]:
    """Minimal stream that includes cached_tokens in the usage chunk."""
    yield _wrap(_stream_chunk({"role": "assistant"}))
    yield _wrap(_stream_chunk({"content": "hi"}, finish_reason="stop"))
    yield _wrap(
        ChatCompletionStreamResponse(
            id="chatcmpl-cache-stream",
            model="test-model",
            choices=[],
            usage=UsageInfo(
                prompt_tokens=42,
                completion_tokens=8,
                total_tokens=50,
                prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=37),
            ),
        )
    )
    yield "data: [DONE]\n\n"


async def test_streaming_cached_tokens(verbose: bool) -> None:
    print("=" * 72)
    print("streaming: cached_tokens propagated onto message-end")
    print("=" * 72)

    handler = _make_handler()
    request = _make_request()

    events: list[dict] = []
    frames: list[str] = []
    async for sse in handler._chat_completion_stream_to_v2(
        _fake_upstream_with_cached_tokens(), request
    ):
        frames.append(sse)
        body = sse[len("data: ") :].strip()
        if body and body != "[DONE]":
            events.append(json.loads(body))

    if verbose:
        for ev in events:
            print(json.dumps(ev, indent=2))

    end = events[-1]
    assert end["type"] == "message-end", end
    assert end["delta"]["usage"]["cached_tokens"] == 37, end["delta"]["usage"]
    assert frames[-1] == "data: [DONE]\n\n", repr(frames[-1])

    print("OK: message-end delta.usage.cached_tokens == 37")
    print("OK: stream terminates with data: [DONE]")
    print()


async def _fake_upstream_no_usage_chunk() -> AsyncIterator[str]:
    """Upstream that closes with ``[DONE]`` but never sends the usage-only
    final chunk (some inference backends do this).
    """
    yield _wrap(_stream_chunk({"role": "assistant"}))
    yield _wrap(_stream_chunk({"content": "hello"}))
    yield _wrap(_stream_chunk({"content": " world"}, finish_reason="stop"))
    yield "data: [DONE]\n\n"


async def _fake_upstream_no_done() -> AsyncIterator[str]:
    """Upstream that just exhausts the iterator without ``[DONE]`` and
    without a usage-only chunk (e.g. cancellation, abrupt shutdown).
    """
    yield _wrap(_stream_chunk({"role": "assistant"}))
    yield _wrap(_stream_chunk({"content": "hello"}, finish_reason="stop"))


async def test_streaming_emits_message_end_without_usage_chunk(
    verbose: bool,
) -> None:
    """Even when upstream skips the usage-only final chunk, the v2 stream
    must terminate cleanly with ``message-end``.
    """
    print("=" * 72)
    print("streaming: fallback message-end (no usage chunk, [DONE] only)")
    print("=" * 72)

    handler = _make_handler()
    request = _make_request()

    events: list[dict] = []
    frames: list[str] = []
    async for sse in handler._chat_completion_stream_to_v2(
        _fake_upstream_no_usage_chunk(), request
    ):
        frames.append(sse)
        body = sse[len("data: ") :].strip()
        if body and body != "[DONE]":
            events.append(json.loads(body))

    if verbose:
        for ev in events:
            print(json.dumps(ev, indent=2))

    types = [ev["type"] for ev in events]
    assert types[0] == "message-start", types[0]
    assert types[-1] == "message-end", types[-1]
    # Open content block must be closed before message-end.
    assert types.count("content-start") == types.count("content-end")
    end = events[-1]
    assert end["delta"]["finish_reason"] == "COMPLETE", end
    # No usage chunk arrived from upstream, so the synthetic message-end
    # omits the ``usage`` field.
    assert "usage" not in end["delta"], end["delta"]
    # Still must terminate with [DONE] even in the fallback path.
    assert frames[-1] == "data: [DONE]\n\n", repr(frames[-1])

    print("OK: message-end emitted despite missing usage chunk")
    print("OK: open content blocks closed before message-end")
    print("OK: finish_reason mapped from last delta-bearing chunk")
    print("OK: stream terminates with data: [DONE]")
    print()


async def test_streaming_emits_message_end_without_done(verbose: bool) -> None:
    """Upstream that exhausts without ``[DONE]`` and without a usage chunk
    must still produce a closing ``message-end`` event.
    """
    print("=" * 72)
    print("streaming: fallback message-end (iterator exhausts, no [DONE])")
    print("=" * 72)

    handler = _make_handler()
    request = _make_request()

    events: list[dict] = []
    frames: list[str] = []
    async for sse in handler._chat_completion_stream_to_v2(
        _fake_upstream_no_done(), request
    ):
        frames.append(sse)
        body = sse[len("data: ") :].strip()
        if body and body != "[DONE]":
            events.append(json.loads(body))

    if verbose:
        for ev in events:
            print(json.dumps(ev, indent=2))

    types = [ev["type"] for ev in events]
    assert types[-1] == "message-end", types
    assert events[-1]["delta"]["finish_reason"] == "COMPLETE"
    # Even when upstream never sent [DONE], we synthesize it on our side.
    assert frames[-1] == "data: [DONE]\n\n", repr(frames[-1])

    print("OK: message-end emitted on plain iterator exhaustion")
    print("OK: stream terminates with data: [DONE]")
    print()


# ----------------------------------------------------------------------
# Request validation (#12 id construction, #14 max_tokens=0 acceptance)
# ----------------------------------------------------------------------


def test_request_accepts_max_tokens_zero(verbose: bool) -> None:
    """Cohere's API treats ``max_tokens=0`` as valid; the v2 request
    validator must accept it (only true negatives are rejected).
    """
    print("=" * 72)
    print("request: max_tokens=0 accepted, negative rejected")
    print("=" * 72)

    req = CohereChatV2Request.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        }
    )
    assert req.max_tokens == 0, req.max_tokens

    # None still allowed.
    req2 = CohereChatV2Request.model_validate(
        {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert req2.max_tokens is None

    # Negative still rejected.
    try:
        CohereChatV2Request.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": -1,
            }
        )
    except ValidationError as exc:
        assert "non-negative" in str(exc), str(exc)
    else:
        raise AssertionError("max_tokens=-1 should have been rejected")

    print("OK: max_tokens=0 accepted")
    print("OK: max_tokens=None accepted")
    print("OK: max_tokens=-1 rejected")
    print()


def test_response_id_synthesized_when_upstream_missing(verbose: bool) -> None:
    """When the upstream ``ChatCompletionResponse.id`` is empty, the v2
    converter must synthesize a non-empty id at the call site.
    """
    print("=" * 72)
    print("response: id synthesized when upstream id is empty")
    print("=" * 72)

    handler = _make_handler()
    request = _make_request()

    msg = ChatMessage(role="assistant", content="hi")
    response = ChatCompletionResponse(
        id="",  # upstream forgot to set it
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(index=0, message=msg, finish_reason="stop")
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    v2 = handler._chat_completion_to_v2(response, request)
    assert v2.id, f"expected non-empty id, got {v2.id!r}"
    assert v2.id.startswith("chat_"), v2.id

    # And the upstream id is preserved when present.
    response.id = "chatcmpl-real-id"
    v2b = handler._chat_completion_to_v2(response, request)
    assert v2b.id == "chatcmpl-real-id", v2b.id

    print(f"OK: empty upstream id -> synthesized {v2.id!r}")
    print("OK: non-empty upstream id preserved")
    print()


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Dump full JSON for each event",
    )
    args = parser.parse_args()

    test_request_accepts_max_tokens_zero(args.verbose)
    test_response_id_synthesized_when_upstream_missing(args.verbose)
    test_non_streaming(args.verbose)
    test_non_streaming_with_tool_call_reasoning_default(args.verbose)
    test_non_streaming_with_tool_call_tool_plan_flag(args.verbose)
    test_non_streaming_cached_tokens(args.verbose)
    asyncio.run(test_streaming(args.verbose))
    asyncio.run(test_streaming_tool_plan(args.verbose))
    asyncio.run(test_streaming_cached_tokens(args.verbose))
    asyncio.run(test_streaming_emits_message_end_without_usage_chunk(args.verbose))
    asyncio.run(test_streaming_emits_message_end_without_done(args.verbose))

    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
