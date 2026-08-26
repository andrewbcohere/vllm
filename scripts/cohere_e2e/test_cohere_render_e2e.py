# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks for ``POST /cohere/v2/chat/render``.

Drives a live vLLM server with a real Cohere Command-family model. The
render endpoint is the Cohere counterpart to
``POST /v1/chat/completions/render``: it converts the v2 body to a
``ChatCompletionRequest``, runs it through the same renderer pipeline the
chat endpoint uses, and returns the resulting ``GenerateRequest``
(prompt ``token_ids`` plus ``sampling_params``) *without* generating.

That "no generation" property is what makes this script cheap: every
case below is a prompt-construction check, so the model never decodes a
token. It still needs a loaded server because tokenization and the
melody templating step both run server-side against the real tokenizer.

What the cases are actually pinning down:

* The returned tokens really are a *melody* prompt, not a Jinja one --
  verified by detokenizing them and looking for Cohere turn markers.
  Renderer selection is a server-level property (``--tokenizer-mode
  cohere`` picks ``CohereRenderer``), so this is the check that the
  render route inherits it rather than falling back to ``HfRenderer``.
* Cohere-only request surface (``documents``, ``tools``, ``safety_mode``)
  reaches the template. These ride on ``chat_template_kwargs``, which is
  the part of the conversion most likely to silently stop being
  forwarded.
* v2 sampling fields land in ``sampling_params`` under their OpenAI
  names (``p`` -> ``top_p``, ``k`` -> ``top_k``, ``stop_sequences`` ->
  ``stop``).
* Byte-for-byte token parity with ``/v1/chat/completions/render`` for a
  request that carries no Cohere-specific fields -- both routes share
  one ``ServingRender``, so any divergence means the v2 conversion is
  perturbing the prompt.

Usage
-----

Same server-management flags as the sibling e2e scripts (see
``vllm_server.py``): point ``--base-url`` at a running server, or let
this script start one::

    python scripts/cohere_e2e/test_cohere_render_e2e.py

    python scripts/cohere_e2e/test_cohere_render_e2e.py \\
        --base-url http://127.0.0.1:8000 --no-auto-start-server

The server must be started with ``VLLM_ENABLE_COHERE_API=1`` and
``--tokenizer-mode cohere`` for these checks to mean anything;
``vllm_server.py`` does both when it owns the process.
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

DEFAULT_MODEL = "CohereLabs/command-a-plus-05-2026-w4a4"

# Turn markers every cmd3/cmd4 melody template emits. A Jinja-rendered
# prompt for a non-Cohere template would not contain these, so they are
# how we tell which renderer actually ran.
START_OF_TURN = "<|START_OF_TURN_TOKEN|>"
USER_TOKEN = "<|USER_TOKEN|>"

PENGUIN_DOCUMENTS = [
    {"data": {"snippet": "The tallest penguin is the Emperor penguin"}},
    {"data": {"snippet": "The latin name for Emperor penguin is Aptenodytes forsteri"}},
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather_for_city",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


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
        return f"e2e-render-{label}-{self.request_id_counter}-{int(time.time())}"

    def record(self, result: TestResult) -> None:
        self.results.append(result)
        prefix = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
        print(f"[{prefix}] {result.name}")
        if result.detail:
            for line in result.detail.splitlines():
                print(f"        {line}")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _looks_like_4xx_envelope(parsed: Any) -> bool:
    """True for any of the documented Cohere v2 4xx response shapes.

    Mirrors the helper in ``test_cohere_v2_e2e.py``: the rejection can
    come from our ``CohereError``, vLLM's generic OpenAI-style envelope,
    or FastAPI's default ``{"detail": [...]}``, depending on how far into
    the stack the request got.
    """
    if not isinstance(parsed, dict):
        return False
    if "message" in parsed and "error" not in parsed:
        return True
    if isinstance(parsed.get("error"), dict) and "message" in parsed["error"]:
        return True
    return "detail" in parsed


def _post(
    ctx: TestContext,
    *,
    path: str,
    body: dict[str, Any],
    request_id: str,
    expect_status: int | None = 200,
) -> tuple[int, dict[str, Any] | str]:
    url = f"{ctx.base_url.rstrip('/')}{path}"
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


def _render(
    ctx: TestContext,
    *,
    body: dict[str, Any],
    label: str,
    expect_status: int | None = 200,
) -> tuple[int, dict[str, Any] | str]:
    return _post(
        ctx,
        path="/cohere/v2/chat/render",
        body=body,
        request_id=ctx.next_request_id(label),
        expect_status=expect_status,
    )


def _token_ids(payload: Any) -> list[int]:
    _expect(isinstance(payload, dict), f"render response is not an object: {payload!r}")
    token_ids = payload.get("token_ids")
    _expect(
        isinstance(token_ids, list) and bool(token_ids),
        f"token_ids missing or empty: {str(payload)[:300]}",
    )
    _expect(
        all(isinstance(t, int) and t >= 0 for t in token_ids),
        "token_ids must be non-negative ints",
    )
    return token_ids


def _sampling_params(payload: Any) -> dict[str, Any]:
    params = payload.get("sampling_params") if isinstance(payload, dict) else None
    _expect(
        isinstance(params, dict),
        f"sampling_params missing or not an object: {str(payload)[:300]}",
    )
    return params


def _detokenize(ctx: TestContext, token_ids: list[int]) -> str:
    """Decode rendered tokens back to the prompt string.

    Uses the server's own ``/detokenize`` so we read the prompt through
    the same tokenizer that produced it, rather than loading one here.
    """
    _, parsed = _post(
        ctx,
        path="/detokenize",
        body={"model": ctx.model, "tokens": token_ids},
        request_id=ctx.next_request_id("detok"),
    )
    _expect(isinstance(parsed, dict), f"detokenize returned non-object: {parsed!r}")
    prompt = parsed.get("prompt")
    _expect(isinstance(prompt, str), f"detokenize response missing prompt: {parsed!r}")
    return prompt


def _simple_body(ctx: TestContext, text: str = "What is the tallest penguin?") -> dict:
    return {
        "model": ctx.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 32,
    }


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------


def test_render_basic_shape(ctx: TestContext) -> None:
    """A minimal v2 body renders to a GenerateRequest and nothing else."""
    name = "render: returns GenerateRequest shape"
    try:
        _, parsed = _render(ctx, body=_simple_body(ctx), label="basic")
        token_ids = _token_ids(parsed)
        assert isinstance(parsed, dict)

        _expect("request_id" in parsed, f"no request_id: {str(parsed)[:200]}")
        _sampling_params(parsed)

        # Render must not generate: none of the chat-response keys should
        # be here. This is the cheap guard that the route didn't get
        # wired to the chat handler by mistake.
        for generated_key in ("message", "text", "finish_reason", "citations"):
            _expect(
                generated_key not in parsed,
                f"render response leaked generated field {generated_key!r}",
            )

        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"{len(token_ids)} tokens, request_id={parsed.get('request_id')!r}"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_tokens_decode_to_melody_prompt(ctx: TestContext) -> None:
    """The rendered tokens are a melody prompt, not a Jinja one."""
    name = "render: tokens decode to a melody prompt"
    try:
        question = "What is the tallest penguin?"
        _, parsed = _render(ctx, body=_simple_body(ctx, question), label="melody")
        prompt = _detokenize(ctx, _token_ids(parsed))

        _expect(
            START_OF_TURN in prompt,
            f"no {START_OF_TURN} in decoded prompt; renderer may not be "
            f"CohereRenderer. prompt={prompt[:400]!r}",
        )
        _expect(
            USER_TOKEN in prompt,
            f"no {USER_TOKEN} in decoded prompt: {prompt[:400]!r}",
        )
        _expect(
            question in prompt,
            f"user message missing from decoded prompt: {prompt[:400]!r}",
        )

        ctx.record(TestResult(name, True, detail=f"prompt starts: {prompt[:160]!r}"))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_documents_reach_the_prompt(ctx: TestContext) -> None:
    """``documents`` are grounded into the prompt, not dropped."""
    name = "render: documents reach the prompt"
    try:
        baseline_body = _simple_body(ctx)
        _, baseline = _render(ctx, body=baseline_body, label="docs-baseline")
        baseline_tokens = _token_ids(baseline)

        grounded_body = {**baseline_body, "documents": PENGUIN_DOCUMENTS}
        _, grounded = _render(ctx, body=grounded_body, label="docs")
        grounded_tokens = _token_ids(grounded)

        _expect(
            len(grounded_tokens) > len(baseline_tokens),
            f"documents did not grow the prompt "
            f"({len(baseline_tokens)} -> {len(grounded_tokens)} tokens)",
        )

        prompt = _detokenize(ctx, grounded_tokens)
        for doc in PENGUIN_DOCUMENTS:
            snippet = doc["data"]["snippet"]
            _expect(
                snippet in prompt,
                f"document snippet missing from prompt: {snippet!r}",
            )

        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"{len(baseline_tokens)} -> {len(grounded_tokens)} tokens "
                    f"with {len(PENGUIN_DOCUMENTS)} documents"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_tools_reach_the_prompt(ctx: TestContext) -> None:
    """``tools`` are templated into the prompt."""
    name = "render: tools reach the prompt"
    try:
        body = {**_simple_body(ctx, "What is the weather in Toronto?")}
        body["tools"] = [WEATHER_TOOL]
        _, parsed = _render(ctx, body=body, label="tools")
        prompt = _detokenize(ctx, _token_ids(parsed))

        tool_name = WEATHER_TOOL["function"]["name"]
        _expect(
            tool_name in prompt,
            f"tool name {tool_name!r} missing from prompt: {prompt[:600]!r}",
        )

        ctx.record(TestResult(name, True, detail=f"found {tool_name!r} in prompt"))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_safety_mode_changes_the_prompt(ctx: TestContext) -> None:
    """``safety_mode`` flows through chat_template_kwargs into melody.

    Two different modes must produce two different preambles; if the
    field stopped being forwarded both would render identically.
    """
    name = "render: safety_mode changes the prompt"
    try:
        base = _simple_body(ctx)
        _, strict = _render(
            ctx, body={**base, "safety_mode": "STRICT"}, label="safety-strict"
        )
        _, contextual = _render(
            ctx, body={**base, "safety_mode": "CONTEXTUAL"}, label="safety-ctx"
        )

        strict_tokens = _token_ids(strict)
        contextual_tokens = _token_ids(contextual)
        _expect(
            strict_tokens != contextual_tokens,
            "STRICT and CONTEXTUAL safety_mode rendered identical prompts; "
            "safety_mode is probably not reaching the template",
        )

        ctx.record(
            TestResult(
                name,
                True,
                detail=(
                    f"STRICT={len(strict_tokens)} tokens, "
                    f"CONTEXTUAL={len(contextual_tokens)} tokens"
                ),
            )
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_sampling_params_mapped(ctx: TestContext) -> None:
    """v2 sampling fields land in sampling_params under OpenAI names."""
    name = "render: v2 sampling params map into sampling_params"
    try:
        body = {
            "model": ctx.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 17,
            "temperature": 0.42,
            "p": 0.83,
            "k": 7,
            "stop_sequences": ["STOP_HERE"],
            "seed": 1234,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
        }
        _, parsed = _render(ctx, body=body, label="sampling")
        params = _sampling_params(parsed)

        expected = {
            "max_tokens": 17,
            "temperature": 0.42,
            "top_p": 0.83,
            "top_k": 7,
            "seed": 1234,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
        }
        mismatches = [
            f"{key}: expected {value!r}, got {params.get(key)!r}"
            for key, value in expected.items()
            if params.get(key) != value
        ]
        _expect(
            not mismatches, "sampling_params mismatch:\n  " + "\n  ".join(mismatches)
        )

        stop = params.get("stop")
        _expect(
            stop == ["STOP_HERE"] or stop == "STOP_HERE",
            f"stop_sequences did not map to stop: {stop!r}",
        )

        ctx.record(
            TestResult(name, True, detail=f"all {len(expected) + 1} fields mapped")
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_matches_chat_completions_render(ctx: TestContext) -> None:
    """A plain request renders identically on both render routes.

    Both go through the same ``ServingRender`` and the same
    ``CohereRenderer``, so for a body with no Cohere-specific fields the
    token ids should match exactly. A difference means the v2
    conversion is perturbing the prompt.
    """
    name = "render: token parity with /v1/chat/completions/render"
    try:
        question = "What is the tallest penguin?"
        _, v2 = _render(ctx, body=_simple_body(ctx, question), label="parity-v2")
        v2_tokens = _token_ids(v2)

        _, v1 = _post(
            ctx,
            path="/v1/chat/completions/render",
            body={
                "model": ctx.model,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 32,
            },
            request_id=ctx.next_request_id("parity-v1"),
        )
        v1_tokens = _token_ids(v1)

        if v2_tokens != v1_tokens:
            v2_prompt = _detokenize(ctx, v2_tokens)
            v1_prompt = _detokenize(ctx, v1_tokens)
            raise AssertionError(
                f"token ids differ ({len(v2_tokens)} vs {len(v1_tokens)} tokens)\n"
                f"  v2: {v2_prompt[:300]!r}\n"
                f"  v1: {v1_prompt[:300]!r}"
            )

        ctx.record(
            TestResult(name, True, detail=f"identical {len(v2_tokens)} token prompt")
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_stream_flag_echoed(ctx: TestContext) -> None:
    """``stream`` is carried into the GenerateRequest rather than
    turning the render response itself into a stream."""
    name = "render: stream flag carried into GenerateRequest"
    try:
        _, parsed = _render(
            ctx, body={**_simple_body(ctx), "stream": True}, label="stream"
        )
        _token_ids(parsed)
        assert isinstance(parsed, dict)
        _expect(
            parsed.get("stream") is True,
            f"stream not carried through: {parsed.get('stream')!r}",
        )
        ctx.record(TestResult(name, True, detail="stream=True echoed"))
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


def test_render_invalid_request_rejected(ctx: TestContext) -> None:
    """A malformed v2 body is rejected as 4xx, not a 500."""
    name = "render: empty messages rejected (400/422)"
    try:
        status, parsed = _render(
            ctx,
            body={"model": ctx.model, "messages": []},
            label="err-empty",
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
            TestResult(name, True, detail=f"status={status} body={str(parsed)[:200]}")
        )
    except Exception as e:
        ctx.record(TestResult(name, False, detail=str(e)))


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

    managed_server = ensure_server(args, log_prefix="vllm-cohere-render-e2e-")

    try:
        ctx = TestContext(
            base_url=args.base_url,
            model=args.model,
            is_reasoning_model=args.is_reasoning_model,
            verbose=args.verbose,
            timeout=args.timeout,
        )

        print(f"Driving Cohere v2 render tests at {ctx.base_url}/cohere/v2/chat/render")
        print(f"  model={ctx.model}")
        print()

        test_render_basic_shape(ctx)
        test_render_tokens_decode_to_melody_prompt(ctx)
        test_render_documents_reach_the_prompt(ctx)
        test_render_tools_reach_the_prompt(ctx)
        test_render_safety_mode_changes_the_prompt(ctx)
        test_render_sampling_params_mapped(ctx)
        test_render_matches_chat_completions_render(ctx)
        test_render_stream_flag_echoed(ctx)
        test_render_invalid_request_rejected(ctx)

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
