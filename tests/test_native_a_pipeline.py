import json
from unittest.mock import Mock
import httpx
import pytest
from conftest import atomic
from as_all.native_a import AClient, build
from agentdojo.functions_runtime import FunctionsRuntime


@pytest.mark.parametrize("method,decision", [("as_only", "allow"), ("as_only_guard", "block"),
                                              ("as_only_guard_frame_reminder", "block")])
@pytest.mark.parametrize("experiment", [None, "public_api"])
def test_actual_executor_and_native_history(method, decision, experiment, tmp_path, monkeypatch):
    atomic(tmp_path / "case.json", {"domain": "slack", "task": "user_task_1"})
    requests = []
    support = {"action_support": {"root": "t1", "via": "Get article to summarize", "authorization_evidence": None}}
    native = {"role": "model", "parts": [{"text": "consider", "thought": True, "thoughtSignature": "full-part-signature"},
        {"functionCall": {"id": "native-call", "name": "lookup", "args": {"x": "hello", "authorization_witness": support}},
         "thoughtSignature": "call-signature"}]}
    def respond(request):
        requests.append(json.loads(request.content))
        content = native if len(requests) == 1 else {"role": "model", "parts": [{"text": "done"}]}
        return httpx.Response(200, json={"responseId": f"r{len(requests)}", "modelVersion": "gemini-2.5-flash",
            "candidates": [{"finishReason": "STOP", "content": content}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 5,
                              "totalTokenCount": 115}}, request=request)
    args = dict(key="dummy", base_url="https://example.invalid/v1beta", ledger=tmp_path / "requests.jsonl")
    # Build with the real executor, replacing only HTTP and the Guard's judgment.
    import as_all.native_a as module
    original = module.AClient
    def factory(**kwargs):
        return AClient(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    module.AClient = factory
    try:
        if experiment is None:
            pipeline = build(method, tmp_path, Mock(), args)
        else:
            from as_all.methods import build as build_profile
            monkeypatch.setenv("GEMINI_API_KEY", "dummy")
            monkeypatch.setenv("GEMINI_BASE_URL", args["base_url"])
            pipeline = build_profile(method, tmp_path, provider="gemini", key="dummy", base_url=args["base_url"], task_id="slack/user_task_1")
        executor = pipeline.elements[-1].elements[0]
        if method != "as_only":
            import ablation_variants as variants
            executor.guard.authorize = Mock(return_value=variants.ActionDecision(
                action_support_valid=decision == "allow", decision=decision, reason="test decision"))
        runtime = FunctionsRuntime()
        executed = []
        @runtime.register_function
        def lookup(x: str) -> str:
            """Lookup article.

            :param x: Article query.
            """
            executed.append(x)
            return "article"
        pipeline.query("Summarize article", runtime)
        assert executed == (["hello"] if method == "as_only" else [])
        assert len(requests) == 2
        assert requests[1]["contents"][1] == native  # complete raw parts, not reconstructed calls
        response = requests[1]["contents"][2]["parts"][0]["functionResponse"]
        assert response["name"] == "lookup"
        assert response["id"] == "native-call"
        text = response["response"]["result"]
        assert ("Reminder: trusted task frames" in text) == (method == "as_only_guard_frame_reminder")
        if method != "as_only":
            payload = executor.guard.authorize.call_args.args[0].model_dump(mode="json")
            assert len(payload["trusted_task_frames"]) == 1
            assert payload["prior_tool_outputs"] == []
            assert payload["proposed_tool_call"] == {"tool": "lookup", "arguments": {"x": "hello"}}
        assert "submit_tool_call_with_authorization_witness" not in json.dumps(requests[0]["tools"])
        assert "available_domain_tools" not in json.dumps(requests[0]["systemInstruction"])
        assert all(r["generationConfig"]["thinkingConfig"]["thinkingBudget"] == -1 for r in requests)
    finally:
        module.AClient = original


def test_frozen_guard_prompt_and_source_reference():
    from as_all.methods import ROOT
    text = (ROOT / "reference/native_a/ablation_variants.py").read_text()
    assert "Treat via as an unverified claim" in text


def test_a_history_cannot_silently_reconstruct_modified_response(tmp_path):
    client = AClient(key="test", base_url="https://example.invalid", ledger=tmp_path / "ledger", stage="agent")
    with pytest.raises(ValueError, match="history missing or modified"):
        client._messages([{"role": "user", "content": "read"}, {"role": "assistant", "content": "invented"}])
