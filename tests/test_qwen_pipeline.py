import hashlib
import json
import httpx
import pytest
from agentdojo.functions_runtime import FunctionsRuntime
from conftest import atomic
from as_all.methods import ROOT, build
from as_all.qwen_transport import QwenClient




@pytest.mark.parametrize("method,decision", [("as_only", "allow"), ("as_only_guard", "allow"),
    ("as_only_guard", "block"), ("as_only_guard_frame_reminder", "block"),
    ("as_only_guard_frame_reminder", "invalid")])
def test_qwen_original_proxy_stage_and_effective_reminder(tmp_path, monkeypatch, method, decision):
    atomic(tmp_path / "case.json", {"domain": "slack", "task": "user_task_1"})
    monkeypatch.setenv("QWEN_API_KEY", "private-test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1")
    records, agent_calls = [], []
    envelope = {"proposed_tool_call": {"tool": "lookup", "arguments": {"x": "hello"}},
                "authorization_witness": {"action_support": {"root": "t1", "via": "Read article"}}}
    def client(**kwargs):
        stage = kwargs["stage"]
        def respond(request):
            body = json.loads(request.content)
            records.append((stage, body))
            assert request.headers["Authorization"] == "Bearer private-test-key"
            assert "x-goog-api-key" not in request.headers
            if stage == "agent":
                agent_calls.append(body)
                message = {"role": "assistant", "content": "done"}
                if len(agent_calls) == 1:
                    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function",
                        "function": {"name": "submit_tool_call_with_authorization_witness", "arguments": json.dumps(envelope)}}]}
            else:
                message = {"role": "assistant", "content": json.dumps({"action_support_valid": decision == "allow",
                    "decision": "allow" if decision == "allow" else "block", "reason": "" if decision == "invalid" else "test judgment"})}
            return httpx.Response(200, request=request, json={"id": "r" + str(len(records)), "object": "chat.completion",
                "created": 0, "model": "qwen3.5-flash", "choices": [{"index": 0, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112,
                          "prompt_tokens_details": {"cached_tokens": 20}}})
        return QwenClient(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("as_all.qwen_transport.QwenClient", client)
    pipeline = build(method, tmp_path, provider="qwen", key="private-test-key", base_url="https://example.invalid/v1", task_id="slack/user_task_1")
    runtime, executed = FunctionsRuntime(), []
    @runtime.register_function
    def lookup(x: str) -> str:
        """Read article.

        :param x: Article query.
        """
        executed.append(x)
        return "article"
    pipeline.query("Read", runtime)
    assert executed == (["hello"] if method == "as_only" or decision == "allow" else [])
    assert all(body["enable_thinking"] is (stage == "agent") for stage, body in records)
    tools = agent_calls[0]["tools"]
    assert [tool["function"]["name"] for tool in tools] == ["submit_tool_call_with_authorization_witness"]
    assert "available_domain_tools" in json.dumps(agent_calls[0]["messages"])
    assert "functionDeclarations" not in json.dumps(agent_calls[0])
    final_input = json.dumps(agent_calls[-1]["messages"])
    assert ("Reminder: trusted task frames" in final_input) == (
        method == "as_only_guard_frame_reminder" and decision == "block")
    guards = [body for stage, body in records if stage == "defense"]
    if method == "as_only":
        assert not guards
    else:
        assert len(guards) == (2 if decision == "invalid" else 1)
        assert "Treat via as an unverified claim" in guards[0]["messages"][0]["content"]
        payload = json.loads(guards[0]["messages"][1]["content"])
        assert len(payload["trusted_task_frames"]) == 1 and not payload["prior_tool_outputs"]
        assert payload["proposed_tool_call"] == {"tool": "lookup", "arguments": {"x": "hello"}}


def test_qwen_via_source_exact_sha():
    assert hashlib.sha256((ROOT / "reference/qwen_via/ablation_variants.py").read_bytes()).hexdigest() == (
        "86a41c9b8ce4469b31ed301a280c38d832a42d87454fb56fd32c17ddb19b218f")


def test_qwen_frame_overlap_matches_frozen_original61():
    frames = json.loads((ROOT / "data/manual_frames97.json").read_text())
    overlap = {k: v for k, v in frames.items() if k.split("/")[0] in ("slack", "travel") or
               (k.startswith("workspace/") and int(k.split("_")[-1]) < 20)}
    assert len(overlap) == 61
    assert hashlib.sha256(json.dumps(overlap, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == (
        "5015f7a5dc69b1fb373a859898a4de9c15ee4c7ee6a6362a9c8cad6638c50de3")
