# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cohere Chat v2 API protocol.

The bulk of the wire types come straight from the official ``cohere``
Python SDK so we stay in lockstep with the upstream specification and
avoid hand-mirroring the schema. We only own three things locally:

1. The top-level request body model (the SDK doesn't ship one — its
   ``ClientV2.chat`` takes the body as kwargs), with vLLM-specific
   extensions (``kv_transfer_params`` / ``chat_template_kwargs``).
2. The non-streaming response envelope (the SDK exposes the message
   shape via :class:`AssistantMessageResponse` but no full response
   wrapper).
3. The streaming discriminated union (the SDK exports each event type
   individually but not as a combined ``Annotated[Union[...],
   discriminator]``).

Importing this module pulls in the ``cohere`` package. The router that
mounts ``POST /v2/chat`` guards on that import succeeding so vLLM still
boots without the SDK installed.

See https://docs.cohere.com/reference/chat for the upstream spec.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from cohere.types import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    ChatContentDeltaEvent,
    ChatContentEndEvent,
    ChatContentStartEvent,
    ChatMessageEndEvent,
    ChatMessageStartEvent,
    ChatMessageV2,
    ChatRequestSafetyMode,
    ChatToolCallDeltaEvent,
    ChatToolCallEndEvent,
    ChatToolCallStartEvent,
    ChatToolPlanDeltaEvent,
    Citation,
    CitationEndEvent,
    CitationOptions,
    CitationStartEvent,
    Document,
    ResponseFormatV2,
    SystemChatMessageV2,
    Thinking,
    ToolCallV2,
    ToolChatMessageV2,
    ToolV2,
    UserChatMessageV2,
)
from pydantic import BaseModel, Field, field_validator

# Re-export the SDK types so that ``vllm.entrypoints.cohere.serving`` and
# friends can import everything they need from this module.
__all__ = [
    "AssistantChatMessageV2",
    "AssistantMessageResponse",
    "ChatContentDeltaEvent",
    "ChatContentEndEvent",
    "ChatContentStartEvent",
    "ChatMessageEndEvent",
    "ChatMessageStartEvent",
    "ChatMessageV2",
    "ChatToolCallDeltaEvent",
    "ChatToolCallEndEvent",
    "ChatToolCallStartEvent",
    "ChatToolPlanDeltaEvent",
    "Citation",
    "CitationEndEvent",
    "CitationOptions",
    "CitationStartEvent",
    "CohereChatV2Request",
    "CohereChatV2Response",
    "CohereError",
    "CohereFinishReason",
    "CohereLogprobItem",
    "CohereUsage",
    "CohereUsageBilledUnits",
    "CohereUsageTokens",
    "Document",
    "ResponseFormatV2",
    "SystemChatMessageV2",
    "Thinking",
    "ToolCallV2",
    "ToolChatMessageV2",
    "ToolV2",
    "UserChatMessageV2",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CohereError(BaseModel):
    """Top-level error body returned by ``/v2/chat`` error responses.

    Cohere's documented error schemas are uniform: ``{message, id}``.
    """

    message: str
    id: str | None = None


# ---------------------------------------------------------------------------
# Tool choice / finish reasons
# ---------------------------------------------------------------------------
#
# These literals aren't first-class enums in the SDK but are documented at
# https://docs.cohere.com/reference/chat. We declare them here so the
# request/response models can validate them.

CohereToolChoice = Literal["REQUIRED", "NONE"]

CohereFinishReason = Literal[
    "COMPLETE",
    "STOP_SEQUENCE",
    "MAX_TOKENS",
    "TOOL_CALL",
    "ERROR",
    "TIMEOUT",
]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CohereChatV2Request(BaseModel):
    """Cohere Chat v2 request body.

    Mirrors the schema documented at https://docs.cohere.com/reference/chat.
    All structured fields delegate to the official SDK types so the body
    schema stays in sync with the upstream spec.
    """

    model: str
    messages: list[ChatMessageV2]
    stream: bool | None = False

    # Tooling
    tools: list[ToolV2] | None = None
    strict_tools: bool | None = None
    tool_choice: CohereToolChoice | None = None

    # Grounding
    documents: list[str | Document] | None = None
    citation_options: CitationOptions | None = None

    # Output
    response_format: ResponseFormatV2 | None = None
    safety_mode: ChatRequestSafetyMode | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None

    # Sampling
    temperature: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    k: int | None = None
    p: float | None = None
    logprobs: bool | None = None

    # Reasoning
    thinking: Thinking | None = None

    # Scheduling
    priority: int | None = None

    # vLLM-specific extensions (not in Cohere spec). These mirror what the
    # Anthropic and OpenAI surfaces already expose so V2 callers can reach
    # the same engine knobs when needed.
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        if not v:
            raise ValueError("model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


# ---------------------------------------------------------------------------
# Usage / Logprobs
# ---------------------------------------------------------------------------
#
# The Cohere SDK only exposes a v1 ``ApiMetaBilledUnits``. The v2 usage
# envelope is documented separately in the OpenAPI spec and we declare it
# here.


class CohereUsageBilledUnits(BaseModel):
    input_tokens: float | None = None
    output_tokens: float | None = None
    search_units: float | None = None
    classifications: float | None = None


class CohereUsageTokens(BaseModel):
    input_tokens: float | None = None
    output_tokens: float | None = None


class CohereUsage(BaseModel):
    billed_units: CohereUsageBilledUnits | None = None
    tokens: CohereUsageTokens | None = None
    cached_tokens: float | None = None


class CohereLogprobItem(BaseModel):
    text: str | None = None
    token_ids: list[int]
    logprobs: list[float] | None = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


class CohereChatV2Response(BaseModel):
    """Cohere Chat v2 non-streaming response body.

    Wraps the SDK :class:`AssistantMessageResponse` (the message shape) in
    the documented v2 response envelope (``id``, ``finish_reason``,
    ``usage``, ``logprobs``).
    """

    id: str
    finish_reason: CohereFinishReason
    message: AssistantMessageResponse
    usage: CohereUsage | None = None
    logprobs: list[CohereLogprobItem] | None = None

    # vLLM-specific extension.
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"chat_{int(time.time() * 1000)}"


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------
#
# Cohere V2 streams a sequence of typed JSON events delivered as Server-
# Sent Events. Each event class is exported above (``ChatMessageStartEvent``,
# ``ChatContentDeltaEvent``, ``CitationStartEvent``, etc.). We don't build
# a discriminated union here because the SDK's event classes don't carry
# the ``type`` literal as a Pydantic field (it's handled by the SDK's own
# deserializer), so pydantic ``Field(discriminator="type")`` would reject
# it. Surfaces emit each event with ``model_dump_json()`` and clients
# parse them by reading ``type`` directly.
#
# See https://docs.cohere.com/v2/docs/streaming and the OpenAPI
# ``StreamedChatResponseV2`` schema for the wire-format reference.
