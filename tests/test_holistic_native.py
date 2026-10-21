import copy
import json

import httpx

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import get_text_content_as_str
from conftest import atomic
from as_all.holistic import HolisticEnvelope, augment, prompts
from as_all.methods import build
from as_all.native_a import AClient


def test_gemini_holistic_native_tools_witness_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.invalid/v1beta")
    atomic(tmp_path / "case.json", {"domain": "slack", "task": "user_task_1"})
    pipeline = build("holistic", tmp_path, provider="gemini", key="test-only", base_url="https://example.invalid/v1beta")
    client = pipeline.elements[2].client
    assert isinstance(client, AClient)
    runtime, executed, requests = FunctionsRuntime(), [], []

    @runtime.register_function
    def lookup(x: str) -> str:
        """Read article.

        :param x: Article query.
        """
        executed.append(x)
        return "article"

    witness = {"contribute_to_score": 0.0, "reason": "No contribution"}
    native = {"role": "model", "parts": [
        {"text": "consider", "thought": True, "thoughtSignature": "full-signature"},
        {"functionCall": {"id": "h1", "name": "lookup", "args": {"x": "hello", "authorization_witness": witness}},
         "thoughtSignature": "call-signature"}]}

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        content = native if len(requests) == 1 else {"role": "model", "parts": [{"text": "done"}]}
        return httpx.Response(200, request=request, json={"responseId": f"r{len(requests)}",
            "modelVersion": "gemini-2.5-flash", "candidates": [{"finishReason": "STOP", "content": content}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 5,
                              "totalTokenCount": 115}})

    client._http.close()
    client._http = httpx.Client(transport=httpx.MockTransport(respond))
    pipeline.query("Read article", runtime)
    # Holistic is self-assessment without a Guard; a zero score does not block execution.
    assert executed == ["hello"]
    assert len(requests) == 2
    assert requests[1]["contents"][1] == native
    response = requests[1]["contents"][2]["parts"][0]["functionResponse"]
    assert response["id"] == "h1" and response["name"] == "lookup"
    assert "article" in response["response"]["result"]
    first = requests[0]
    assert all(r["generationConfig"]["thinkingConfig"]["thinkingBudget"] == -1 for r in requests)
    assert [t["name"] for t in first["tools"][0]["functionDeclarations"]] == ["lookup"]
    schema = first["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]
    assert "authorization_witness" in schema["required"]
    witness_schema = schema["properties"]["authorization_witness"]
    assert witness_schema == HolisticEnvelope.model_json_schema()["$defs"]["HolisticWitness"]
    original = copy.deepcopy(schema)
    original["properties"].pop("authorization_witness")
    original["required"].remove("authorization_witness")
    original.pop("$defs")
    assert original == runtime.functions["lookup"].parameters.model_json_schema()
    system = first["systemInstruction"]["parts"][0]["text"]
    for text in prompts():
        assert text in system
    assert "contribute_to_score" in system
    assert "available_domain_tools" not in system and "trusted_task_frames" not in system
    assert "submit_tool_call_with_authorization_witness" not in system
    assert json.loads(system.split("PROGRAM-GENERATED CONTEXT:\n")[1]) == {"trusted_user_task": "Read article"}
    ledger = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert all(row["stage"] == "agent" for row in ledger)


def test_qwen_holistic_keeps_original_proxy(tmp_path, monkeypatch):
    from as_all.qwen_transport import QwenClient
    monkeypatch.setenv("QWEN_API_KEY", "test-only")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1")
    pipeline = build("holistic", tmp_path, provider="qwen", key="test-only", base_url="https://example.invalid/v1")
    assert isinstance(pipeline.elements[2].client, QwenClient)
    runtime = FunctionsRuntime()
    messages, _ = augment([{"role": "system", "content": []}], "Read article", runtime)
    text = get_text_content_as_str(messages[0]["content"])
    assert "available_domain_tools" in text
    assert "call only submit_tool_call_with_authorization_witness" in text


def test_holistic_all_four_domain_schemas_preserved(tmp_path):
    from types import SimpleNamespace
    from agentdojo.task_suite.load_suites import get_suite
    from as_all.holistic import HolisticAClient

    client = HolisticAClient(key="test-only", base_url="https://example.invalid/v1beta",
                            ledger=tmp_path / "ledger", stage="agent")
    count = 0
    for domain in ("slack", "travel", "workspace", "banking"):
        suite = get_suite("v1.2.2", domain)
        runtime = SimpleNamespace(functions={tool.name: tool for tool in suite.tools})
        messages, _ = augment([{"role": "system", "content": []}], "Trusted task", runtime)
        messages[0]["content"] = get_text_content_as_str(messages[0]["content"])
        messages.append({"role": "user", "content": "Trusted task"})
        body = client._request(dict(model="gemini-2.5-flash", messages=messages,
            tools=[{"type": "function", "function": {"name": "submit_tool_call_with_authorization_witness",
                   "parameters": HolisticEnvelope.model_json_schema()}}]))
        for tool, native in zip(suite.tools, body["tools"][0]["functionDeclarations"], strict=True):
            assert native["name"] == tool.name and native["description"] == tool.description
            schema = copy.deepcopy(native["parametersJsonSchema"])
            witness = schema["properties"].pop("authorization_witness")
            assert witness["properties"]["contribute_to_score"]["minimum"] == 0
            assert witness["properties"]["contribute_to_score"]["maximum"] == 1
            assert set(witness["required"]) == {"contribute_to_score", "reason"}
            schema["required"].remove("authorization_witness")
            original = tool.parameters.model_json_schema()
            if not schema["required"] and "required" not in original:
                schema.pop("required")
            if not schema.get("$defs") and "$defs" not in original:
                schema.pop("$defs", None)
            assert schema == original, (domain, tool.name)
            count += 1
    assert count > 63
