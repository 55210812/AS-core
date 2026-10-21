"""Gemini native REST transport with an OpenAI Chat Completions surface."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion

MODEL = "gemini-2.5-flash"
Stage = Literal["agent", "defense"]


class GeminiProtocolError(RuntimeError):
    """The native endpoint returned a response outside the accepted protocol."""


def _is_not_given(value: Any) -> bool:
    return value is NOT_GIVEN or value.__class__.__name__ == "NotGiven"


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            part = _plain(part)
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise ValueError("unsupported Chat Completions message content")
            pieces.append(part["text"])
        return "".join(pieces)
    raise ValueError("unsupported Chat Completions message content")


def _resolve_schema(schema: Any, definitions: Mapping[str, Any] | None = None, stack: tuple[str, ...] = ()) -> Any:
    """Inline local JSON Schema references for Gemini function declarations."""
    if isinstance(schema, list):
        return [_resolve_schema(item, definitions, stack) for item in schema]
    if not isinstance(schema, dict):
        return schema

    local_definitions = dict(definitions or {})
    local_definitions.update(schema.get("$defs") or {})
    local_definitions.update(schema.get("definitions") or {})
    ref = schema.get("$ref")
    if ref is not None:
        prefixes = ("#/$defs/", "#/definitions/")
        prefix = next((item for item in prefixes if ref.startswith(item)), None)
        if prefix is None:
            raise ValueError(f"unsupported non-local JSON Schema reference: {ref}")
        name = ref[len(prefix) :]
        if "/" in name or name not in local_definitions:
            raise ValueError(f"unresolved JSON Schema reference: {ref}")
        if name in stack:
            raise ValueError(f"recursive JSON Schema reference is unsupported: {ref}")
        resolved = _resolve_schema(local_definitions[name], local_definitions, (*stack, name))
        siblings = {key: value for key, value in schema.items() if key not in {"$ref", "$defs", "definitions"}}
        if siblings:
            if not isinstance(resolved, dict):
                raise ValueError(f"JSON Schema reference with siblings did not resolve to an object: {ref}")
            resolved = {**resolved, **_resolve_schema(siblings, local_definitions, stack)}
        return resolved

    return {
        key: _resolve_schema(value, local_definitions, stack)
        for key, value in schema.items()
        if key not in {"$defs", "definitions", "$schema"}
    }


def _thinking_from_extra(extra_body: Any) -> int | None:
    if extra_body is None or _is_not_given(extra_body):
        return None
    extra = _plain(extra_body)
    if not isinstance(extra, dict):
        raise TypeError("extra_body must be an object")
    if extra == {}:
        return None
    google = extra.get("google")
    if isinstance(google, dict):
        config = google.get("thinking_config")
        if isinstance(config, dict) and set(config) == {"thinking_budget"}:
            return int(config["thinking_budget"])
    template = extra.get("chat_template_kwargs")
    if isinstance(template, dict) and set(template) == {"enable_thinking"}:
        return -1 if template["enable_thinking"] is True else 0
    if set(extra) == {"enable_thinking"}:
        return -1 if extra["enable_thinking"] is True else 0
    raise ValueError("unsupported extra_body; thinking is controlled by the explicit transport stage")


def _response_format(value: Any) -> dict[str, Any]:
    value = _plain(value)
    if not isinstance(value, dict):
        raise TypeError("response_format must be an object")
    kind = value.get("type")
    if kind == "json_object" and set(value) == {"type"}:
        return {"responseMimeType": "application/json"}
    if kind == "json_schema" and set(value) == {"type", "json_schema"}:
        wrapper = _plain(value["json_schema"])
        if not isinstance(wrapper, dict):
            raise ValueError("response_format.json_schema must be an object")
        schema = wrapper.get("schema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("response JSON Schema must have an object at its top level")
        return {
            "responseMimeType": "application/json",
            "responseJsonSchema": _resolve_schema(schema),
        }
    raise ValueError("only json_object and top-level-object json_schema response formats are supported")


def _tool_choice(value: Any, has_tools: bool) -> dict[str, Any] | None:
    if value is None or _is_not_given(value):
        value = "auto" if has_tools else "none"
    value = _plain(value)
    if not has_tools:
        if value != "none":
            raise ValueError("tool_choice requires tools")
        return None
    if value == "auto":
        return None
    if value == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if value == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(value, dict) and value.get("type") == "function":
        function = _plain(value.get("function"))
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [function["name"]],
                }
            }
    raise ValueError(f"unsupported tool_choice: {value!r}")


class _GeminiCompletions:
    def __init__(self, owner: NativeClient) -> None:
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        if args:
            raise ValueError("positional Chat Completions arguments are unsupported")
        return self._owner._create(kwargs)


class _GeminiChat:
    def __init__(self, owner: NativeClient) -> None:
        self.completions = _GeminiCompletions(owner)


class NativeClient:
    """Expose ``client.chat.completions.create`` over Gemini generateContent.

    ``stage`` is mandatory because it determines the sole thinking policy:
    adaptive thinking for the agent and disabled thinking for defenses.
    """
    model = MODEL
    provider = "gemini"
    usage_field = "usageMetadata"

    def __init__(
        self,
        *,
        ledger: str | Path,
        stage: Stage,
        key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        request_check: Any = None,
        _sleep: Any = time.sleep,
    ) -> None:
        if stage not in ("agent", "defense"):
            raise ValueError("stage must be 'agent' or 'defense'")
        self.stage = stage
        self._thinking_budget = -1 if stage == "agent" else 0
        self._key = key if key is not None else os.environ.get("GEMINI_API_KEY")
        self._base_url = base_url if base_url is not None else os.environ.get("GEMINI_BASE_URL")
        if not self._key:
            raise ValueError("GEMINI_API_KEY is required")
        if not self._base_url:
            raise ValueError("GEMINI_BASE_URL is required")
        self._url = self._generate_content_url(self._base_url)
        self._ledger = Path(ledger)
        self._ledger.parent.mkdir(parents=True, exist_ok=True)
        self._ledger_lock = threading.Lock()
        self._http = http_client or httpx.Client(timeout=httpx.Timeout(connect=20.0, read=180.0, write=60.0, pool=20.0))
        self._owns_http = http_client is None
        self._sleep = _sleep
        self._request_check = request_check
        self._thought_signatures: dict[str, str] = {}
        self._native_history: dict[str, dict[str, Any]] = {}
        self.chat = _GeminiChat(self)

    @staticmethod
    def _generate_content_url(base_url: str) -> str:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GEMINI_BASE_URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("GEMINI_BASE_URL must not embed user credentials or fragments")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        if not path.endswith("/v1beta"):
            path += "/v1beta"
        path += f"/models/{MODEL}:generateContent"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def _headers(self):
        return {"x-goog-api-key": self._key, "Content-Type": "application/json"}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _messages(self, messages: list[Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        contents: list[dict[str, Any]] = []
        call_names: dict[str, str] = {}
        native_ids: dict[str, str] = {}
        pending_calls: set[str] = set()
        response_parts: list[dict[str, Any]] | None = None
        for raw_message in messages:
            message = _plain(raw_message)
            if not isinstance(message, dict):
                raise TypeError("messages must contain objects")
            role = message.get("role")
            if role in ("system", "developer"):
                if pending_calls:
                    raise ValueError("tool-call turn is missing function responses")
                instructions.append(_text(message.get("content")))
                continue
            if role == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending_calls:
                    raise ValueError(f"tool result has no preceding call: {call_id}")
                if response_parts is None:
                    response_parts = []
                    contents.append({"role": "user", "parts": response_parts})
                response_parts.append(
                    {
                        "functionResponse": {
                            "name": call_names[call_id],
                            "response": {"result": _text(message.get("content"))},
                        }
                    }
                )
                if call_id in native_ids:
                    response_parts[-1]["functionResponse"]["id"] = native_ids[call_id]
                pending_calls.remove(call_id)
                continue
            if pending_calls:
                raise ValueError("tool-call turn is missing function responses")
            response_parts = None
            if role not in ("user", "assistant"):
                raise ValueError(f"unsupported Chat Completions role: {role}")
            parts: list[dict[str, Any]] = []
            content = _text(message.get("content"))
            if content:
                parts.append({"text": content})
            for raw_call in message.get("tool_calls") or []:
                call = _plain(raw_call)
                function = _plain(call.get("function"))
                if not isinstance(function, dict):
                    raise TypeError("tool call is missing its function")
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments must be a JSON object")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("tool-call ID is required")
                if call_id in call_names:
                    raise ValueError("duplicate tool-call ID")
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("tool-call function name is required")
                call_names[call_id] = name
                pending_calls.add(call_id)
                part: dict[str, Any] = {"functionCall": {"name": name, "args": arguments}}
                signature = self._thought_signatures.get(call_id)
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            if not parts:
                raise ValueError(f"empty {role} message")
            saved = self._native_history.get(self._history_key(message)) if role == "assistant" else None
            if saved is not None:
                parts = deepcopy(saved["parts"])
                native_calls = [p["functionCall"] for p in parts if "functionCall" in p]
                for call, native in zip(message.get("tool_calls") or [], native_calls, strict=True):
                    call_names[call["id"]] = native["name"]
                    if native.get("id"):
                        native_ids[call["id"]] = native["id"]
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        if pending_calls:
            raise ValueError("tool-call turn is missing function responses")
        if not contents:
            raise ValueError("generateContent requires at least one conversation message")
        system = {"parts": [{"text": "\n".join(instructions)}]} if instructions else None
        return system, contents

    def _request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "model",
            "messages",
            "temperature",
            "tools",
            "tool_choice",
            "extra_body",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stream",
        }
        unknown = {key for key, value in kwargs.items() if key not in allowed and not _is_not_given(value)}
        if unknown:
            raise ValueError(f"unsupported generateContent request fields: {sorted(unknown)}")
        if kwargs.get("model") != MODEL:
            raise ValueError(f"model must be {MODEL}")
        if kwargs.get("temperature", 0) not in (None, 0, 0.0, NOT_GIVEN):
            raise ValueError("temperature is fixed at 0")
        stream = kwargs.get("stream")
        if stream not in (None, False) and not _is_not_given(stream):
            raise ValueError("streaming is unsupported")
        supplied_budget = _thinking_from_extra(kwargs.get("extra_body"))
        if supplied_budget is not None and supplied_budget != self._thinking_budget:
            raise ValueError("extra_body thinking configuration conflicts with the explicit transport stage")

        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        system, contents = self._messages(messages)
        generation: dict[str, Any] = {
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": self._thinking_budget},
        }
        first_limit = kwargs.get("max_completion_tokens")
        second_limit = kwargs.get("max_tokens")
        if _is_not_given(first_limit):
            first_limit = None
        if _is_not_given(second_limit):
            second_limit = None
        if first_limit is not None and second_limit is not None and first_limit != second_limit:
            raise ValueError("max_tokens and max_completion_tokens conflict")
        limit = first_limit if first_limit is not None else second_limit
        if limit is not None:
            generation["maxOutputTokens"] = int(limit)
        response_format = kwargs.get("response_format")
        if response_format is not None and not _is_not_given(response_format):
            generation.update(_response_format(response_format))

        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system is not None:
            body["systemInstruction"] = system
        raw_tools = kwargs.get("tools")
        if raw_tools is None or _is_not_given(raw_tools):
            raw_tools = []
        declarations: list[dict[str, Any]] = []
        for raw_tool in raw_tools:
            tool = _plain(raw_tool)
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("only function tools are supported")
            function = _plain(tool.get("function"))
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise TypeError("function tool name is required")
            declaration: dict[str, Any] = {
                "name": function["name"],
                "description": function.get("description", ""),
            }
            parameters = function.get("parameters")
            if parameters is not None:
                declaration["parametersJsonSchema"] = _resolve_schema(parameters)
            declarations.append(declaration)
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]
        config = _tool_choice(kwargs.get("tool_choice"), bool(declarations))
        if config is not None:
            body["toolConfig"] = config
        return body

    @staticmethod
    def _history_key(message: dict[str, Any]) -> str:
        calls = [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                 for c in message.get("tool_calls") or []]
        return json.dumps([_text(message.get("content")), calls], sort_keys=True, ensure_ascii=False)

    def _completion(self, data: dict[str, Any]) -> ChatCompletion:
        if data.get("modelVersion") and not re.fullmatch(r"gemini-2\.5-flash(?:-\d{3}|@[^\s]+)?", data["modelVersion"]):
            raise GeminiProtocolError("ModelIdentityMismatch")
        candidates = data.get("candidates") or []
        if len(candidates) != 1:
            raise RuntimeError("EmptyModelResponse: Gemini native response has no single candidate")
        candidate = candidates[0]
        finish = candidate.get("finishReason", "STOP")
        if finish not in ("STOP", "MAX_TOKENS"):
            raise RuntimeError(f"GeminiNativeResponseRejected: {finish}")
        parts = (candidate.get("content") or {}).get("parts") or []
        texts = [part["text"] for part in parts if "text" in part and not part.get("thought")]
        thought_text = "".join(part["text"] for part in parts if "text" in part and part.get("thought"))
        calls: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            if "functionCall" not in part:
                continue
            function = part["functionCall"]
            call_id = (
                function.get("id")
                or "call_" + hashlib.sha256(f"{data.get('responseId', '')}:{index}".encode()).hexdigest()[:24]
            )
            signature = part.get("thoughtSignature")
            if isinstance(signature, str) and signature:
                self._thought_signatures[call_id] = signature
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(function.get("args") or {}, ensure_ascii=False),
                    },
                }
            )
        content = "".join(texts)
        if not content and not calls:
            raise RuntimeError("EmptyModelResponse: Gemini native response has no text or tool call")
        if calls and finish != "STOP":
            raise RuntimeError("GeminiNativeResponseRejected: incomplete function call")

        raw_usage = data.get("usageMetadata")
        usage: dict[str, Any] | None = None
        if raw_usage is not None:
            required = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
            if all(type(raw_usage.get(key)) is int for key in required):
                prompt = raw_usage["promptTokenCount"]
                candidate_tokens = raw_usage["candidatesTokenCount"]
                total = raw_usage["totalTokenCount"]
                thoughts = raw_usage.get("thoughtsTokenCount")
                if thoughts is None:
                    thoughts = total - prompt - candidate_tokens
                if type(thoughts) is not int or thoughts < 0 or total != prompt + candidate_tokens + thoughts:
                    raise RuntimeError("Gemini native token counts are inconsistent")
                usage = {
                    "prompt_tokens": prompt,
                    "completion_tokens": candidate_tokens + thoughts,
                    "total_tokens": total,
                    "completion_tokens_details": {"reasoning_tokens": thoughts},
                }
                cached = raw_usage.get("cachedContentTokenCount")
                if type(cached) is int:
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}

        finish_reason = "tool_calls" if calls else ("length" if finish == "MAX_TOKENS" else "stop")
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if thought_text:
            message["reasoning_content"] = thought_text
        if calls:
            message["tool_calls"] = calls
        completion: dict[str, Any] = {
            "id": data.get("responseId") or f"gemini-native-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("modelVersion") or MODEL,
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        }
        if usage is not None:
            completion["usage"] = usage
        result = ChatCompletion.model_validate(completion)
        self._native_history[self._history_key(result.model_dump(mode="json")["choices"][0]["message"])] = deepcopy(
            candidate["content"])
        return result

    def _sanitize(self, value: Any) -> Any:
        secret = self._key

        def scrub(item: Any) -> Any:
            if isinstance(item, dict):
                return {str(key): scrub(val) for key, val in item.items()}
            if isinstance(item, list):
                return [scrub(val) for val in item]
            if isinstance(item, str):
                text = item.replace(secret, "[REDACTED]") if secret else item
                return re.sub(r"(?i)([?&](?:key|api_key|token|access_token)=)[^&#\s]+", r"\1[REDACTED]", text)
            return item

        return scrub(deepcopy(value))

    def _append(self, event: dict[str, Any]) -> None:
        encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._ledger_lock, self._ledger.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()

    def _create(self, kwargs: dict[str, Any]) -> ChatCompletion:
        if self._request_check is not None:
            self._request_check()
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        body = self._request(kwargs)
        safe_request = self._sanitize(body)
        common = {
            "request_id": request_id,
            "stage": self.stage,
            "model": self.model,
            "provider": self.provider,
            "thinking_budget": self._thinking_budget,
            "full_request": safe_request,
        }
        self._append({"event": "request_started", **common})
        attempts: list[dict[str, Any]] = []
        data = None
        try:
            response: httpx.Response | None = None
            for attempt in range(1, 4):
                if self._request_check is not None:
                    self._request_check()
                try:
                    response = self._http.post(
                        self._url,
                        headers=self._headers(),
                        json=body,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    attempts.append({"attempt": attempt, "retry": attempt < 3, "error_type": type(exc).__name__})
                    if attempt == 3:
                        raise
                    self._sleep(0.25 * (2 ** (attempt - 1)))
                    continue
                if response.status_code == 429 and attempt < 3:
                    try:
                        rejected = response.json()
                    except ValueError:
                        rejected = {}
                    attempts.append({"attempt": attempt, "retry": True, "status_code": 429,
                                     "raw_usage": self._sanitize(rejected.get(self.usage_field))
                                     if isinstance(rejected, dict) else None})
                    self._sleep(0.25 * (2 ** (attempt - 1)))
                    continue
                break
            if response is None:
                raise RuntimeError("Gemini request ended without a response")
            attempts.append({"attempt": len(attempts) + 1, "status_code": response.status_code})
            try:
                data = response.json()
            except ValueError:
                data = None
            response.raise_for_status()
            if not isinstance(data, dict):
                raise GeminiProtocolError("Gemini native response must be a JSON object")
            completion = self._completion(data)
            elapsed = time.monotonic() - started
            self._append(
                {
                    "event": "request_completed",
                    **common,
                    "elapsed_seconds": elapsed,
                    "transport_attempts": attempts,
                    "raw_usage": self._sanitize(data.get(self.usage_field)),
                    "response": self._sanitize(data),
                }
            )
            return completion
        except Exception as exc:
            self._append(
                {
                    "event": "request_error",
                    **common,
                    "elapsed_seconds": time.monotonic() - started,
                    "transport_attempts": attempts,
                    "raw_usage": self._sanitize(data.get(self.usage_field)) if isinstance(data, dict) else None,
                    "response": self._sanitize(data),
                    "error": {"type": type(exc).__name__, "message": self._sanitize(str(exc))},
                }
            )
            raise
