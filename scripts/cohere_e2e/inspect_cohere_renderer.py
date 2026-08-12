#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect the prompt that the Cohere renderer produces for a chat request.

This script bypasses the engine entirely: it constructs a `VllmConfig` via
`EngineArgs`, loads the renderer registered for the chosen tokenizer mode,
and runs `render_messages_async` on a sample chat. Use this to iterate on
`vllm/renderers/cohere.py` and the cmd3/cmd4 templates without having to
boot a model.

Run with:

    .venv/bin/python scripts/inspect_cohere_renderer.py \
        --model hmellor/tiny-random-LlamaForCausalLM \
        --tokenizer-mode cohere

Pass ``--tokenizer-mode hf`` to compare against the default Jinja-based
renderer for the same model. The model only needs a tokenizer/config on
disk; we never instantiate the engine, so any small public model works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cohere.protocol import CohereChatV2Request
from vllm.entrypoints.cohere.serving import CohereServingChatV2
from vllm.renderers import ChatParams
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config

SAMPLE_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You answer concisely."},
    {"role": "user", "content": "Who wrote Hamlet, and when?"},
]

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_0",
        "data": {
            "title": "Wikipedia: Hamlet",
            "text": (
                "Hamlet was written by William Shakespeare some time between"
                " 1599 and 1601."
            ),
        },
    },
    {
        "id": "doc_1",
        "data": {"title": "Britannica", "text": "Shakespeare lived 1564-1616."},
    },
]

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_play_metadata",
            "description": "Look up metadata about a Shakespeare play.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The play's title"}
                },
                "required": ["title"],
            },
        },
    }
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="hmellor/tiny-random-LlamaForCausalLM",
        help=(
            "HF repo id or local path. Any model with a fast tokenizer and a "
            "config.json works; we never load weights."
        ),
    )
    p.add_argument(
        "--tokenizer-mode",
        default="cohere",
        choices=["auto", "hf", "cohere"],
        help="Renderer/tokenizer mode to exercise. Defaults to 'cohere'.",
    )
    p.add_argument(
        "--cohere-format",
        default="cmd3",
        choices=["cmd3", "cmd4"],
        help="When tokenizer-mode=cohere, which template family to render.",
    )
    p.add_argument(
        "--with-documents",
        action="store_true",
        help="Include sample documents (grounding) in chat_template_kwargs.",
    )
    p.add_argument(
        "--with-tools",
        action="store_true",
        help="Include sample tools in chat_template_kwargs.",
    )
    p.add_argument(
        "--safety-mode",
        default=None,
        choices=[None, "contextual", "strict", "none"],
        help="cmd3 safety mode forwarded via chat_template_kwargs.",
    )
    p.add_argument(
        "--reasoning",
        default=None,
        choices=[None, "enabled", "disabled"],
        help="Reasoning toggle forwarded via chat_template_kwargs.",
    )
    p.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print prompt_token_ids alongside the text.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to ModelConfig (needed for some Cohere repos).",
    )
    p.add_argument(
        "--from-v2-request",
        action="store_true",
        help=(
            "Construct a Cohere v2 request, run the same"
            " v2 -> ChatCompletion conversion that POST /cohere/v2/chat does, and"
            " render the result through the configured renderer. Demonstrates"
            " what a non-cohere (`--tokenizer-mode hf`) model actually sees"
            " when called with the v2 input shape."
        ),
    )
    return p.parse_args()


def _build_v2_request(args: argparse.Namespace) -> CohereChatV2Request:
    """Build a representative Cohere v2 request mirroring the script's flags."""
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You answer concisely."},
            {"role": "user", "content": "Who wrote Hamlet, and when?"},
        ],
    }
    if args.with_documents:
        body["documents"] = [
            {
                "id": "doc_0",
                "data": {
                    "title": "Wikipedia: Hamlet",
                    "text": (
                        "Hamlet was written by William Shakespeare some time"
                        " between 1599 and 1601."
                    ),
                },
            },
            {
                "id": "doc_1",
                "data": {
                    "title": "Britannica",
                    "text": "Shakespeare lived 1564-1616.",
                },
            },
        ]
    if args.with_tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_play_metadata",
                    "description": "Look up metadata about a Shakespeare play.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The play's title",
                            }
                        },
                        "required": ["title"],
                    },
                },
            }
        ]
    if args.safety_mode:
        body["safety_mode"] = args.safety_mode.upper()
    if args.reasoning:
        body["thinking"] = {"type": args.reasoning}
    body["citation_options"] = {"mode": "FAST"}
    return CohereChatV2Request.model_validate(body)


async def main() -> int:
    args = _parse_args()

    # Build a VllmConfig the same way the engine does, but never start the
    # engine. ``skip_tokenizer_init=False`` is the default and is what the
    # renderer needs.
    eng_args = EngineArgs(
        model=args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        # Keep this small so HF config validation is fast on tiny models.
        max_model_len=4096,
        # We don't actually run the engine; just satisfy the validators.
        enforce_eager=True,
    )
    config = eng_args.create_engine_config()

    tokenizer = cached_tokenizer_from_config(config.model_config)
    renderer = RENDERER_REGISTRY.load_renderer(
        args.tokenizer_mode if args.tokenizer_mode != "auto" else "hf",
        config,
        tokenizer,
    )

    chat_template_kwargs: dict[str, Any] = {}
    messages: list[dict[str, Any]] = SAMPLE_MESSAGES

    if args.from_v2_request:
        v2_req = _build_v2_request(args)
        chat_req = CohereServingChatV2._convert_v2_to_chat_completion(v2_req)
        messages = [
            m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
            for m in chat_req.messages
        ]
        chat_template_kwargs.update(chat_req.chat_template_kwargs or {})
        if args.tokenizer_mode == "cohere":
            chat_template_kwargs["cohere_format"] = args.cohere_format
        if chat_req.tools:
            chat_template_kwargs.setdefault(
                "tools",
                [t.model_dump(exclude_none=True) for t in chat_req.tools],
            )
    else:
        if args.tokenizer_mode == "cohere":
            chat_template_kwargs["cohere_format"] = args.cohere_format
        if args.with_documents:
            chat_template_kwargs["documents"] = SAMPLE_DOCUMENTS
        if args.with_tools:
            chat_template_kwargs["tools"] = SAMPLE_TOOLS
        if args.safety_mode is not None:
            chat_template_kwargs["safety_mode"] = args.safety_mode
        if args.reasoning is not None:
            chat_template_kwargs["reasoning_type"] = args.reasoning

    params = ChatParams(chat_template_kwargs=chat_template_kwargs)

    print("=" * 72)
    print(
        f"model={args.model}  tokenizer_mode={args.tokenizer_mode}"
        + (f"  format={args.cohere_format}" if args.tokenizer_mode == "cohere" else "")
    )
    print(f"renderer={type(renderer).__name__}")
    if chat_template_kwargs:
        print("chat_template_kwargs:")
        print(json.dumps(chat_template_kwargs, indent=2, default=str))
    print("=" * 72)

    conversation, prompt = await renderer.render_messages_async(messages, params)

    print("\n--- conversation (post parse_chat_messages) ---")
    print(json.dumps(conversation, indent=2, default=str))

    print("\n--- rendered prompt text ---")
    text = prompt.get("prompt")
    if text is None and "prompt_token_ids" in prompt:
        text = tokenizer.decode(prompt["prompt_token_ids"])
    print(text)

    if args.show_token_ids and "prompt_token_ids" in prompt:
        print("\n--- prompt_token_ids ---")
        print(prompt["prompt_token_ids"])

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
