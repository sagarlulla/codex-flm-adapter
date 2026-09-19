#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = os.getenv("ADAPTER_HOST", "127.0.0.1")
PORT = int(os.getenv("ADAPTER_PORT", "8080"))

FLM_BASE_URL = os.getenv(
    "FLM_BASE_URL",
    "http://127.0.0.1:52625/v1",
).rstrip("/")

MODELS_FILE = Path(
    os.getenv(
        "FLM_MODELS_FILE",
        str(Path(__file__).with_name("models.json")),
    )
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("codex-flm-adapter")

app = FastAPI(
    title="Codex <-> FastFlowLM Responses Adapter",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

with MODELS_FILE.open("r", encoding="utf-8") as f:
    MODELS: dict[str, dict[str, Any]] = json.load(f)


def model_info(model: str) -> dict[str, Any]:
    return MODELS.get(
        model,
        {
            "display_name": model,
            "description": "FastFlowLM model",
            "context_window": 8192,
            "think": False,
            "think_toggleable": False,
            "reasoning_efforts": [],
            "tool_calling": False,
            "vision": False,
        },
    )


def supported_model(model: str) -> bool:
    return model in MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now() -> int:
    return int(time.time())


def sse(event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
    )


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def text_from_content(content: Any) -> str:
    """
    Convert Responses content parts / Chat content into plain text.

    Handles:
      - string
      - Responses input_text
      - Responses output_text
      - simple lists of content parts
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            typ = item.get("type")

            if typ in (
                "input_text",
                "output_text",
                "text",
            ):
                value = item.get("text")
                if value:
                    parts.append(value)

        return "".join(parts)

    return str(content)


# ---------------------------------------------------------------------------
# Responses input -> Chat Completions messages
# ---------------------------------------------------------------------------

def responses_input_to_messages(
    input_data: Any,
) -> list[dict[str, Any]]:
    """
    Convert the important Responses input forms to Chat messages.

    Supports:
      - input="..."
      - input=[...]
      - message items
      - function_call_output items
      - assistant/function_call history
    """

    if isinstance(input_data, str):
        return [
            {
                "role": "user",
                "content": input_data,
            }
        ]

    if not isinstance(input_data, list):
        return []

    messages: list[dict[str, Any]] = []

    # Function-call arguments/results need to be associated with the
    # preceding assistant tool call. Keep a lookup by call_id.
    assistant_tool_calls: dict[str, dict[str, Any]] = {}

    for item in input_data:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        # ---------------------------------------------------------------
        # Responses "message"
        # ---------------------------------------------------------------

        if item_type in (None, "message"):
            role = item.get("role", "user")
            content = item.get("content")

            if isinstance(content, list):
                converted_content = []

                for part in content:
                    if not isinstance(part, dict):
                        continue

                    ptype = part.get("type")

                    if ptype in ("input_text", "text"):
                        converted_content.append(
                            {
                                "type": "text",
                                "text": part.get("text", ""),
                            }
                        )

                    elif ptype == "input_image":
                        # FLM accepts OpenAI-style image_url content.
                        image_url = part.get("image_url")

                        if image_url:
                            converted_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    },
                                }
                            )

                content = converted_content

            else:
                content = text_from_content(content)

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

            continue

        # ---------------------------------------------------------------
        # Responses function_call
        # ---------------------------------------------------------------

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")

            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }

            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }

            messages.append(assistant_message)

            if call_id:
                assistant_tool_calls[call_id] = assistant_message

            continue

        # ---------------------------------------------------------------
        # Responses function_call_output
        # ---------------------------------------------------------------

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output", "")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": (
                        output
                        if isinstance(output, str)
                        else json.dumps(output)
                    ),
                }
            )

            continue

        # ---------------------------------------------------------------
        # Legacy/custom assistant message
        # ---------------------------------------------------------------

        if item_type == "assistant":
            messages.append(
                {
                    "role": "assistant",
                    "content": text_from_content(
                        item.get("content")
                    ),
                }
            )

    return messages


# ---------------------------------------------------------------------------
# Responses request -> FLM request
# ---------------------------------------------------------------------------

def responses_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    model = body["model"]

    info = model_info(model)

    messages: list[dict[str, Any]] = []

    instructions = body.get("instructions")

    if instructions:
        messages.append(
            {
                "role": "system",
                "content": instructions,
            }
        )

    messages.extend(
        responses_input_to_messages(
            body.get("input", "")
        )
    )

    chat: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(body.get("stream", False)),
    }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    # Codex Responses does not use max_tokens in the same way as Chat.
    # Keep this configurable.
    max_output_tokens = body.get("max_output_tokens")

    if max_output_tokens is not None:
        chat["max_tokens"] = max_output_tokens

    if "temperature" in body:
        chat["temperature"] = body["temperature"]

    if "top_p" in body:
        chat["top_p"] = body["top_p"]

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    reasoning = body.get("reasoning")

    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")

        if effort:
            allowed = info.get("reasoning_efforts", [])

            if effort in allowed:
                # FastFlowLM/OpenAI-compatible Chat endpoint accepts
                # reasoning_effort for reasoning-capable models.
                chat["reasoning_effort"] = effort

                log.debug(
                    "reasoning: model=%s effort=%s",
                    model,
                    effort,
                )
            else:
                log.warning(
                    "Ignoring unsupported reasoning effort "
                    "model=%s effort=%s allowed=%s",
                    model,
                    effort,
                    allowed,
                )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    tools = body.get("tools")

    if tools and info.get("tool_calling", False):
        chat["tools"] = convert_responses_tools_to_chat(tools)

        if "tool_choice" in body:
            chat["tool_choice"] = body["tool_choice"]

        # Codex commonly wants parallel tool calls.
        chat["parallel_tool_calls"] = body.get(
            "parallel_tool_calls",
            True,
        )

    elif tools:
        log.warning(
            "Model %s does not advertise tool calling; "
            "not forwarding tools to FLM",
            model,
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    if "stop" in body:
        chat["stop"] = body["stop"]

    return chat


def convert_responses_tools_to_chat(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            # Ignore built-in Responses tools such as web_search.
            # FLM cannot execute Codex-hosted tools itself.
            continue

        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get(
                        "description",
                        "",
                    ),
                    "parameters": tool.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                },
            }
        )

    return result


# ---------------------------------------------------------------------------
# Chat response helpers
# ---------------------------------------------------------------------------

def extract_chat_message(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []

    if not choices:
        return {
            "content": "",
            "tool_calls": [],
        }

    message = choices[0].get("message") or {}

    return {
        "content": message.get("content"),
        "reasoning_content": (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        ),
        "tool_calls": message.get("tool_calls") or [],
    }


# ---------------------------------------------------------------------------
# Non-streaming Chat -> Responses
# ---------------------------------------------------------------------------

def chat_to_responses(
    body: dict[str, Any],
    chat_response: dict[str, Any],
) -> dict[str, Any]:
    response_id = make_id("resp")
    model = body["model"]

    message = extract_chat_message(chat_response)

    output: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Reasoning
    # ---------------------------------------------------------------

    reasoning_content = message.get(
        "reasoning_content",
        "",
    )

    if reasoning_content:
        output.append(
            {
                "id": make_id("rs"),
                "type": "reasoning",
                "status": "completed",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": reasoning_content,
                    }
                ],
            }
        )

    # ---------------------------------------------------------------
    # Tool calls
    # ---------------------------------------------------------------

    for tool_call in message.get("tool_calls", []):
        function = tool_call.get("function") or {}

        output.append(
            {
                "id": tool_call.get("id")
                or make_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.get("id")
                or make_id("call"),
                "name": function.get("name", ""),
                "arguments": function.get(
                    "arguments",
                    "{}",
                ),
            }
        )

    # ---------------------------------------------------------------
    # Text
    # ---------------------------------------------------------------

    content = message.get("content")

    if content:
        output.append(
            {
                "id": make_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )

    usage = chat_response.get("usage") or {}

    return {
        "id": response_id,
        "object": "response",
        "created_at": now(),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get(
                "prompt_tokens",
                0,
            ),
            "output_tokens": usage.get(
                "completion_tokens",
                0,
            ),
            "total_tokens": usage.get(
                "total_tokens",
                0,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

async def stream_response(
    body: dict[str, Any],
    request: Request,
) -> AsyncIterator[str]:

    response_id = make_id("resp")
    model = body["model"]

    connection_id = make_id("conn")
    log.info(
        "[%s] Responses stream started response_id=%s model=%s client=%s",
        connection_id,
        response_id,
        model,
        request.client.host if request.client else "unknown",
    )

    chat_request = responses_to_chat(body)
    chat_request["stream"] = True

    log.info(
        "stream model=%s",
        model,
    )

    yield sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": now(),
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        },
    )

    text_started = False
    content_part_started = False

    text_item_id = make_id("msg")
    content_index = 0
    text_buffer = ""

    tool_state: dict[int, dict[str, Any]] = {}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=None,
            write=30.0,
            pool=30.0,
        )
    ) as client:
        flm_url = f"{FLM_BASE_URL}/chat/completions"
        log.info(
            "[%s] FLM CONNECT start method=POST url=%s",
            connection_id,
            flm_url,
        )
        try:
            async with client.stream(
                "POST",
                flm_url,
                json=chat_request,
                headers={
                    "Authorization": "Bearer flm",
                    "Content-Type": "application/json",
                },
            ) as upstream:
                
                log.info(
                    "[%s] FLM CONNECT established "
                    "status=%s http_version=%s",
                    connection_id,
                    upstream.status_code,
                    getattr(upstream, "http_version", "unknown"),
                )

                if upstream.status_code >= 400:
                    error_text = await upstream.aread()
                    
                    log.error(
                        "[%s] FLM HTTP error status=%s body=%s",
                        connection_id,
                        upstream.status_code,
                        error_text.decode(
                            "utf-8",
                            errors="replace",
                        ),
                    )

                    yield sse(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "message": error_text.decode(
                                    "utf-8",
                                    errors="replace",
                                )
                            },
                        },
                    )
                    return

                async for line in upstream.aiter_lines():
                    
                    if await request.is_disconnected():
                        log.warning(
                            "[%s] Codex client disconnected; "
                            "closing upstream FLM stream response_id=%s",
                            connection_id,
                            response_id,
                        )
                        return
        
                    if await request.is_disconnected():
                        log.warning(
                            "[%s] Codex client disconnected; "
                            "closing upstream FLM stream response_id=%s",
                            connection_id,
                            response_id,
                        )
                        return

                    if not line:
                        continue

                    if line.startswith(":"):
                        continue

                    if line.startswith("data:"):
                        raw = line[5:].strip()

                        if raw == "[DONE]":
                            break

                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning(
                                "Invalid FLM SSE data: %s",
                                raw,
                            )
                            continue

                        choices = chunk.get("choices") or []

                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}

                        # ------------------------------------------------
                        # Reasoning
                        # ------------------------------------------------

                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )

                        if reasoning:
                            yield sse(
                                "response.reasoning_summary_text.delta",
                                {
                                    "type": (
                                        "response."
                                        "reasoning_summary_text.delta"
                                    ),
                                    "item_id": make_id("rs"),
                                    "summary_index": 0,
                                    "delta": reasoning,
                                },
                            )

                        # ------------------------------------------------
                        # Text
                        # ------------------------------------------------

                        content = delta.get("content")

                        if content:
                            text_buffer += content

                            # ------------------------------------------------------------
                            # Create the assistant message output item exactly once.
                            # ------------------------------------------------------------
                            if not text_started:
                                text_started = True

                                yield sse(
                                    "response.output_item.added",
                                    {
                                        "type": (
                                            "response."
                                            "output_item.added"
                                        ),
                                        "output_index": 0,
                                        "item": {
                                            "id": text_item_id,
                                            "type": "message",
                                            "status": "in_progress",
                                            "role": "assistant",
                                            "content": [],
                                        },
                                    },
                                )
                            # ------------------------------------------------------------
                            # Create the output_text content part exactly once.
                            # ------------------------------------------------------------
                            if not content_part_started:
                                content_part_started = True
                                
                                yield sse(
                                    "response.content_part.added",
                                    {
                                        "type": (
                                            "response."
                                            "content_part.added"
                                        ),
                                        "item_id": text_item_id,
                                        "output_index": 0,
                                        "content_index": content_index,
                                        "part": {
                                            "type": "output_text",
                                            "text": "",
                                            "annotations": [],
                                        },
                                    },
                                )
                            # ------------------------------------------------------------
                            # Stream the actual text delta.
                            # ------------------------------------------------------------
                            yield sse(
                                "response.output_text.delta",
                                {
                                    "type": (
                                        "response."
                                        "output_text.delta"
                                    ),
                                    "item_id": text_item_id,
                                    "output_index": 0,
                                    "content_index": content_index,
                                    "delta": content,
                                },
                            )

                        # ------------------------------------------------
                        # Tool calls
                        # ------------------------------------------------

                        for index, tc in enumerate(
                            delta.get("tool_calls") or []
                        ):
                            function = tc.get(
                                "function"
                            ) or {}

                            if index not in tool_state:
                                call_id = (
                                    tc.get("id")
                                    or make_id("call")
                                )

                                item_id = make_id("fc")

                                tool_state[index] = {
                                    "call_id": call_id,
                                    "item_id": item_id,
                                    "name": function.get(
                                        "name",
                                        "",
                                    ),
                                    "arguments": "",
                                }

                                state = tool_state[index]

                                yield sse(
                                    "response.output_item.added",
                                    {
                                        "type": (
                                            "response."
                                            "output_item.added"
                                        ),
                                        "output_index": (
                                            len(tool_state) - 1
                                        ),
                                        "item": {
                                            "id": state["item_id"],
                                            "type": "function_call",
                                            "status": "in_progress",
                                            "call_id": state["call_id"],
                                            "name": state["name"],
                                            "arguments": "",
                                        },
                                    },
                                )

                            state = tool_state[index]

                            arguments = function.get(
                                "arguments",
                                "",
                            )

                            if arguments:
                                state["arguments"] += arguments

                                yield sse(
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": (
                                            "response."
                                            "function_call_arguments.delta"
                                        ),
                                        "item_id": state[
                                            "item_id"
                                        ],
                                        "output_index": index,
                                        "delta": arguments,
                                    },
                                )

                log.info(
                    "[%s] FLM stream ended normally",
                    connection_id,
                )
                
                # --------------------------------------------------------
                # Finish tool calls
                # --------------------------------------------------------

                for index, state in tool_state.items():
                    yield sse(
                        "response.function_call_arguments.done",
                        {
                            "type": (
                                "response."
                                "function_call_arguments.done"
                            ),
                            "item_id": state["item_id"],
                            "output_index": index,
                            "arguments": state["arguments"],
                        },
                    )

                    yield sse(
                        "response.output_item.done",
                        {
                            "type": (
                                "response."
                                "output_item.done"
                            ),
                            "output_index": index,
                            "item": {
                                "id": state["item_id"],
                                "type": "function_call",
                                "status": "completed",
                                "call_id": state["call_id"],
                                "name": state["name"],
                                "arguments": state["arguments"],
                            },
                        },
                    )

                if text_started:
                    yield sse(
                        "response.output_text.done",
                        {
                            "type": (
                                "response.output_text.done"
                            ),
                            "item_id": text_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": text_buffer,
                        },
                    )

                    yield sse(
                        "response.content_part.done",
                        {
                            "type": (
                                "response.content_part.done"
                            ),
                            "item_id": text_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": text_buffer,
                                "annotations": [],
                            },
                        },
                    )

                    yield sse(
                        "response.output_item.done",
                        {
                            "type": (
                                "response.output_item.done"
                            ),
                            "output_index": 0,
                            "item": {
                                "id": text_item_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": text_buffer,
                                        "annotations": [],
                                    }
                                ],
                            },
                        },
                    )

        except Exception as exc:
            log.exception(
                "[%s] FLM streaming failure "
                "response_id=%s model=%s",
                connection_id,
                response_id,
                model,
            )

            yield sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "message": str(exc),
                    },
                },
            )

            return

    log.info(
        "[%s] Responses stream completed "
        "response_id=%s text_chars=%d tool_calls=%d",
        connection_id,
        response_id,
        len(text_buffer),
        len(tool_state),
    )
    
    yield sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": now(),
                "status": "completed",
                "model": model,
                "output": [
                        {
                                "id": text_item_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                        {
                                                "type": "output_text",
                                                "text": text_buffer,
                                                "annotations": [],
                                        }
                                ],
                        }
                if text_started else []
                ],
            },
        },
    )

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Codex model catalog
# ---------------------------------------------------------------------------

def reasoning_presets(
    info: dict[str, Any],
) -> list[dict[str, Any]]:

    result = []

    descriptions = {
        "none": "No reasoning",
        "minimal": "Minimal reasoning",
        "low": "Low reasoning effort",
        "medium": "Medium reasoning effort",
        "high": "High reasoning effort",
        "xhigh": "Extra-high reasoning effort",
    }

    for effort in info.get(
        "reasoning_efforts",
        [],
    ):
        result.append(
            {
                "effort": effort,
                "description": descriptions.get(
                    effort,
                    f"{effort} reasoning effort",
                ),
            }
        )

    return result


def codex_model_info(
    slug: str,
    info: dict[str, Any],
) -> dict[str, Any]:

    reasoning = reasoning_presets(info)

    default_reasoning = None

    if reasoning:
        default_reasoning = next(
            (
                x
                for x in reasoning
                if x["effort"] == "medium"
            ),
            reasoning[0],
        )["effort"]

    # These fields correspond to the Codex ModelInfo contract.
    #
    # The catalog is deliberately conservative:
    # no web search, no apply_patch, no native hosted tools.
    return {
        "slug": slug,
        "display_name": info.get(
            "display_name",
            slug,
        ),
        "description": info.get(
            "description",
            "FastFlowLM model",
        ),

        "default_reasoning_level": default_reasoning,
        "supported_reasoning_levels": reasoning,

        "shell_type": (
            "unified_exec"
            if info.get("tool_calling", False)
            else "disabled"
        ),

        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,

        "additional_speed_tiers": [],
        "service_tiers": [],
        "default_service_tier": None,

        "available_access_programs": None,
        "availability_nux": None,
        "upgrade": None,

        "model_messages": {
            "instructions_template": (
                "You are a coding agent running on "
                "a locally hosted FastFlowLM model."
            )
        },

        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": False,

        "supports_reasoning_summary_parameter": True,
        "default_reasoning_summary": "auto",

        "support_verbosity": False,
        "default_verbosity": None,

        "apply_patch_tool_type": None,
        "web_search_tool_type": "text",

        "truncation_policy": {
            "mode": "tokens",
            "limit": max(
                1000,
                int(
                    info.get(
                        "context_window",
                        8192,
                    )
                    * 0.85
                ),
            ),
        },

        "supports_image_detail_original": False,

        "context_window": info.get(
            "context_window"
        ),

        "max_context_window": info.get(
            "context_window"
        ),

        "auto_compact_token_limit": None,
        "comp_hash": None,

        "effective_context_window_percent": 85,

        "experimental_supported_tools": [],

        "input_modalities": (
            ["text", "image"]
            if info.get("vision", False)
            else ["text"]
        ),

        "supports_search_tool": False,
        "supports_experimental_context": False,
        "use_responses_lite": False,

        "supports_reasoning_effort_updates": False,

        "node_repl_auto_review_required": False,
        "node_repl_disabled": True,

        "auto_review_model_override": None,
        "model_specialty": None,

        "tool_mode": (
            "code_mode"
            if info.get("tool_calling", False)
            else None
        ),

        "multi_agent_version": None,
        "multi_agent_reasoning_effort": None
    }


@app.get("/v1/models")
async def models_endpoint(
    request: Request,
):
    """
    Return Codex's native model catalog.

    Codex custom-provider discovery uses the internal catalog format,
    not merely the ordinary OpenAI {"data":[...]} model list.
    """

    catalog = []

    for slug, info in MODELS.items():
        if info.get("embedding") or info.get("audio"):
            continue

        catalog.append(
            codex_model_info(
                slug,
                info,
            )
        )

    return {
        "models": catalog
    }


# Also provide the ordinary OpenAI-ish representation.
# This is useful for diagnostics and clients other than Codex.
@app.get("/v1/models/openai")
async def openai_models_endpoint():
    data = []

    for slug, info in MODELS.items():
        data.append(
            {
                "id": slug,
                "object": "model",
                "created": now(),
                "owned_by": "fastflowlm",
            }
        )

    return {
        "object": "list",
        "data": data,
    }


# ---------------------------------------------------------------------------
# Responses endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/responses")
async def responses_endpoint(
    request: Request,
):
    client_host = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else "unknown"
    
    body = await request.json()
    
    log.info(
        "Responses request received method=%s path=%s "
        "client=%s:%s stream=%s model=%s",
        request.method,
        request.url.path,
        client_host,
        client_port,
        body.get("stream", False),
        body.get("model"),
    )
    
    model = body.get("model")
        
    if not model:
        return JSONResponse(
            {
                "error": {
                    "message": "model is required",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    if not supported_model(model):
        return JSONResponse(
            {
                "error": {
                    "message": (
                        f"Unknown FastFlowLM model: {model}"
                    ),
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    if body.get("stream", False):
        return StreamingResponse(
            stream_response(body, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    chat_request = responses_to_chat(body)

    log.info(
        "request model=%s stream=false",
        model,
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=30.0,
            pool=30.0,
        )
    ) as client:

        try:
            response = await client.post(
                f"{FLM_BASE_URL}/chat/completions",
                json=chat_request,
                headers={
                    "Authorization": "Bearer flm",
                    "Content-Type": "application/json",
                },
            )

        except Exception as exc:
            log.exception("FLM request failed")

            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                    }
                },
                status_code=502,
            )

    if response.status_code >= 400:
        return JSONResponse(
            {
                "error": {
                    "message": response.text,
                    "type": "upstream_error",
                }
            },
            status_code=502,
        )

    result = chat_to_responses(
        body,
        response.json(),
    )

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "ok": True,
        "adapter": "codex-flm",
        "flm": FLM_BASE_URL,
        "models": len(MODELS),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
    
    uvicorn.run(
        "adapter:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
