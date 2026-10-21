from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from as_all.transport import MODEL, NativeClient


def test_existing_openai_style_base_uses_native_route_without_double_version():
    assert NativeClient._generate_content_url("https://example.invalid/v1") == (
        "https://example.invalid/v1beta/models/gemini-2.5-flash:generateContent")
    assert NativeClient._generate_content_url("https://example.invalid/proxy/v1beta") == (
        "https://example.invalid/proxy/v1beta/models/gemini-2.5-flash:generateContent")


def _response(*, tool: bool = False, usage: bool = True, signature: str | None = None) -> dict:
    if tool:
        part = {"functionCall": {"name": "lookup", "args": {"item": "a"}}}
        if signature is not None:
            part["thoughtSignature"] = signature
    else:
        part = {"text": '{"ok":true}'}
    result = {
        "responseId": "response-1",
        "modelVersion": MODEL,
        "candidates": [{"finishReason": "STOP", "content": {"parts": [part]}}],
    }
    if usage:
        result["usageMetadata"] = {
            "promptTokenCount": 10,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 7,
            "totalTokenCount": 20,
            "cachedContentTokenCount": 4,
        }
    return result


def _client(tmp_path: Path, handler, *, stage: str = "agent", key: str = "secret-key") -> NativeClient:
    return NativeClient(
        key=key,
        base_url="https://example.test/proxy?access_token=base-secret",
        ledger=tmp_path / "ledger.jsonl",
        stage=stage,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        _sleep=lambda _: None,
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_agent_request_schema_json_mode_usage_and_redacted_ledger(tmp_path: Path) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response(), request=request)

    client = _client(tmp_path, handler)
    result = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "Use secret-key only as text."},
            {"role": "user", "content": "Return an object."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup an item",
                    "parameters": {
                        "$defs": {"Item": {"type": "string", "enum": ["a", "b"]}},
                        "type": "object",
                        "properties": {"item": {"$ref": "#/$defs/Item"}},
                        "required": ["item"],
                    },
                },
            }
        ],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )

    assert seen["key"] == "secret-key"
    assert "access_token=base-secret" in seen["url"]
    assert seen["body"]["generationConfig"] == {
        "temperature": 0,
        "thinkingConfig": {"thinkingBudget": -1},
        "responseMimeType": "application/json",
    }
    schema = seen["body"]["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]
    assert schema["properties"]["item"] == {"type": "string", "enum": ["a", "b"]}
    assert "$defs" not in json.dumps(schema)
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 10
    assert result.usage.completion_tokens_details.reasoning_tokens == 7
    assert result.usage.prompt_tokens_details.cached_tokens == 4

    events = _events(tmp_path / "ledger.jsonl")
    assert [event["event"] for event in events] == ["request_started", "request_completed"]
    serialized = json.dumps(events)
    assert "secret-key" not in serialized
    assert "base-secret" not in serialized
    assert events[1]["raw_usage"]["promptTokenCount"] == 10
    assert events[1]["response"]["responseId"] == "response-1"


def test_defense_json_schema_thinking_conflict_and_missing_usage(tmp_path: Path) -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response(usage=False), request=request)

    client = _client(tmp_path, handler, stage="defense")
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Judge."}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"decision": {"type": "string"}},
                    "required": ["decision"],
                },
            },
        },
        extra_body={"google": {"thinking_config": {"thinking_budget": 0}}},
    )
    assert result.usage is None
    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert bodies[0]["generationConfig"]["responseJsonSchema"]["type"] == "object"

    with pytest.raises(ValueError, match="conflicts"):
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Judge."}],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )


def test_tool_errors_parallel_responses_and_thought_signature_round_trip(tmp_path: Path) -> None:
    bodies = []
    responses = [_response(tool=True, signature="sig-1"), _response()]

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0), request=request)

    client = _client(tmp_path, handler)
    first = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Lookup."}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    )
    call_id = first.choices[0].message.tool_calls[0].id
    client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Lookup."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"item":"a"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "ValueError: missing item"},
        ],
    )
    model_part = bodies[1]["contents"][1]["parts"][0]
    assert model_part["thoughtSignature"] == "sig-1"
    assert bodies[1]["contents"][2]["parts"][0]["functionResponse"]["response"] == {
        "result": "ValueError: missing item"
    }


def test_retry_policy_only_connect_and_429(tmp_path: Path) -> None:
    attempts = 0

    def retrying(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("not connected", request=request)
        if attempts == 2:
            return httpx.Response(429, json={"error": "limited"}, request=request)
        return httpx.Response(200, json=_response(), request=request)

    client = _client(tmp_path, retrying)
    client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hello"}])
    assert attempts == 3
    completed = _events(tmp_path / "ledger.jsonl")[-1]
    assert [attempt["retry"] for attempt in completed["transport_attempts"][:2]] == [True, True]

    reads = 0

    def read_timeout(request: httpx.Request) -> httpx.Response:
        nonlocal reads
        reads += 1
        raise httpx.ReadTimeout("partial response", request=request)

    client = _client(tmp_path / "read", read_timeout)
    with pytest.raises(httpx.ReadTimeout):
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Hello"}])
    assert reads == 1
    assert _events(tmp_path / "read" / "ledger.jsonl")[-1]["event"] == "request_error"


def test_rejects_wrong_model_stream_and_non_object_response_schema(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, json=_response(), request=request))
    with pytest.raises(ValueError, match="model must"):
        client.chat.completions.create(model="other", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="streaming"):
        client.chat.completions.create(model=MODEL, stream=True, messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="top level"):
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "x"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "items", "schema": {"type": "array", "items": {"type": "string"}}},
            },
        )
