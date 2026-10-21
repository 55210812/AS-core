import copy
import json

import httpx
import pytest

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.types import text_content_block_from_string
from conftest import atomic
from as_all.methods import WITNESS_FORM_METHODS
from as_all.methods import ROOT, build
from as_all.qwen_transport import QwenClient
from as_all.witness_forms import REFERENCE, component, guard_payload







def test_original_prompts_and_component_bodies_preserved():
    import ast
    import hashlib
    assert hashlib.sha256((REFERENCE / "history_prompt.txt").read_bytes()).hexdigest() == (
        "f244cd51e0f4e1b16efd9435af8daf1aa50a8168ab737abfbf325842290ba84d")
    assert hashlib.sha256((REFERENCE / "reasoning_prompt.txt").read_bytes()).hexdigest() == (
        "fd75d107a941611cd67d94baf38d07e396df19325bd344bd7aa30d839447af28")
    receipt = json.loads((REFERENCE / "component_sources.json").read_text())
    for name, provenance in receipt.items():
        text = (REFERENCE / name).read_text()
        nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for symbol, expected in provenance["extracted_symbol_sha256"].items():
            segment = ast.get_source_segment(text, nodes[symbol]) + "\n"
            assert hashlib.sha256(segment.encode()).hexdigest() == expected
    text = (ROOT / "vendor/agentdojo/src/agentdojo/agent_pipeline/task_shield_defense.py").read_text()
    helpers = {node.name: ast.get_source_segment(text, node) for node in ast.walk(ast.parse(text))
               if isinstance(node, ast.FunctionDef) and node.name in ("bind", "_message_text")}
    assert {name: hashlib.sha256(body.encode()).hexdigest() for name, body in helpers.items()} == {
        "bind": "587b3c8916c93fe33031f0762ab43f46051300cfdd3a61a4a3db4737ea09f815",
        "_message_text": "b3e5537e438f13eadcc8bd2f6fedca8cabfc11b64f2d5c916c44530ca0d07e7c"}


def test_pre_action_snapshot_omits_current_and_sibling_calls_and_reasoning():
    old = FunctionCall(function="lookup", args={"x": "old"}, id="old")
    current = FunctionCall(function="lookup", args={"x": "new"}, id="new")
    sibling = FunctionCall(function="lookup", args={"x": "sibling"}, id="sibling")
    messages = [{"role": "user", "content": [text_content_block_from_string("Task")]},
        {"role": "assistant", "content": [], "tool_calls": [old], "reasoning_content": "old hidden reasoning"},
        {"role": "tool", "content": [text_content_block_from_string("old output")], "tool_call": old,
         "tool_call_id": "old", "error": "original tool error"},
        {"role": "assistant", "content": [], "tool_calls": [current, sibling], "reasoning_content": "current reasoning"}]
    before = copy.deepcopy(messages)
    frames = [{"id": "t1", "task": "Task"}]
    packet = guard_payload("no_witness_guard", frames, messages, current)
    assert set(packet) == {"trusted_task_frames", "proposed_tool_call", "conversation_history", "related_content"}
    assert len(packet["conversation_history"]) == 3
    assert packet["conversation_history"][1]["tool_calls"][0]["call_id"] == "old"
    assert packet["conversation_history"][2]["error"] == "original tool error"
    history = json.dumps(packet["conversation_history"])
    assert '"new"' not in history and '"sibling"' not in history and "reasoning" not in history
    reasoning = guard_payload("reasoning_context_guard", frames, messages, current)
    assert set(reasoning) == {"trusted_task_frames", "proposed_tool_call", "reasoning_context"}
    assert reasoning["reasoning_context"] == "current reasoning" and messages == before
    messages[-1]["reasoning_content"] = None
    assert guard_payload("reasoning_context_guard", frames, messages, current)["reasoning_context"] == ""


@pytest.mark.parametrize("method", WITNESS_FORM_METHODS)
@pytest.mark.parametrize("decision", ["allow", "block", "invalid"])
def test_real_pipeline_protocol_execution_thinking_and_usage(tmp_path, monkeypatch, method, decision):
    atomic(tmp_path / "case.json", {"domain": "slack", "task": "user_task_1"})
    monkeypatch.setenv("QWEN_API_KEY", "private-test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1")
    records, agents, executed = [], [], []
    witness = {"action_support": {"root": "t1", "via": "Read the required article"},
               "parameter_grounding": {"roots": ["t1"], "via": "The trusted task names this query"}}
    envelope = {"proposed_tool_call": {"tool": "lookup", "arguments": {"x": "hello"}}, "authorization_witness": witness}
    def client(**kwargs):
        stage = kwargs["stage"]
        def respond(request):
            body = json.loads(request.content)
            records.append((stage, body))
            if stage == "agent":
                agents.append(body)
                message = {"role": "assistant", "content": "done"}
                if len(agents) == 1:
                    name = "submit_tool_call_with_authorization_witness" if method.startswith("as_pg") else "lookup"
                    args = envelope if method.startswith("as_pg") else {"x": "hello"}
                    message.update(content=None, reasoning_content="  current original reasoning\n",
                        tool_calls=[{"id": "proposal", "type": "function", "function": {
                            "name": name, "arguments": json.dumps(args)}}])
            else:
                result = {"decision": "allow" if decision == "allow" else "block", "reason": "Synthetic judgment"}
                if method == "no_witness_guard":
                    result.update(root="t1" if decision == "allow" else None,
                                  via="Original Guard action support" if decision == "allow" else None)
                if method == "as_pg_guard":
                    result["action_support_valid"] = decision == "allow"
                    result["parameter_grounding_valid"] = decision == "allow"
                message = {"role": "assistant", "content": "invalid" if decision == "invalid" else json.dumps(result)}
            return httpx.Response(200, request=request, json={"id": "r" + str(len(records)), "object": "chat.completion",
                "created": 0, "model": "qwen3.5-flash", "choices": [{"index": 0,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112,
                          "prompt_tokens_details": {"cached_tokens": 20},
                          "completion_tokens_details": {"reasoning_tokens": 4 if stage == "agent" else 0}}})
        return QwenClient(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr("as_all.qwen_transport.QwenClient", client)
    pipeline = build(method, tmp_path, provider="qwen", key="private-test-key", base_url="https://example.invalid/v1", task_id="slack/user_task_1")
    domain = FunctionsRuntime()
    @domain.register_function
    def lookup(x: str) -> str:
        """Read article.

        :param x: Article query.
        """
        executed.append(x)
        return "article"
    pipeline.query("Read", domain)
    assert executed == (["hello"] if method == "as_pg" or decision == "allow" else [])
    assert all(body["enable_thinking"] is (stage == "agent") for stage, body in records)
    expected_tool = "submit_tool_call_with_authorization_witness" if method.startswith("as_pg") else "lookup"
    assert agents[0]["tools"][0]["function"]["name"] == expected_tool
    defenses = [body for stage, body in records if stage == "defense"]
    assert len(defenses) == (0 if method == "as_pg" else 2 if decision == "invalid" else 1)
    assert all(body["max_tokens"] == 2048 for body in defenses)
    if defenses:
        if method in ("no_witness_guard", "reasoning_context_guard"):
            prompt_name = "history" if method == "no_witness_guard" else "reasoning"
            expected_prompt = (REFERENCE / (prompt_name + "_prompt.txt")).read_text()
        else:
            from semantic_action_frame_guard.joint_witness_runtime_guard import JOINT_WITNESS_GUARD_SYSTEM
            expected_prompt = JOINT_WITNESS_GUARD_SYSTEM
        assert defenses[0]["messages"][0]["content"] == expected_prompt
    if defenses:
        payload = json.loads(defenses[0]["messages"][1]["content"])
        if method == "no_witness_guard":
            assert set(payload) == {"trusted_task_frames", "proposed_tool_call", "conversation_history", "related_content"}
            assert "proposal" not in json.dumps(payload["conversation_history"])
        elif method == "reasoning_context_guard":
            assert payload["reasoning_context"] == "  current original reasoning\n"
            assert "conversation_history" not in payload
        else:
            expected_witness = copy.deepcopy(witness)
            expected_witness["action_support"]["authorization_evidence"] = None
            assert payload["authorization_witness"] == expected_witness
            assert len(payload["trusted_task_frames"]) == 1 and not payload["prior_tool_outputs"]
    assert "Reminder: trusted task frames" not in json.dumps(agents[-1])
    from as_all.usage import cost
    usage = cost(tmp_path / "requests.jsonl")
    assert usage["agent"]["output"] == 12 * len(agents)




def test_empty_pg_program_normalization_and_nonempty_pg_required(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1")
    atomic(tmp_path / "case.json", {"domain": "slack", "task": "user_task_0"})
    pipeline = build("as_pg_guard", tmp_path, provider="qwen", key="test", base_url="https://example.invalid/v1", task_id="slack/user_task_0")
    executor = pipeline.elements[3].elements[0]
    runtime = FunctionsRuntime()
    @runtime.register_function
    def empty() -> str:
        """Read status."""
        return "ok"
    @runtime.register_function
    def lookup(x: str) -> str:
        """Read a record.

        :param x: Query.
        """
        return x
    proxy = "submit_tool_call_with_authorization_witness"
    payload = {"proposed_tool_call": {"tool": "empty", "arguments": {}},
        "authorization_witness": {"action_support": {"root": "t1", "via": "Read status"}}}
    call = FunctionCall(function=proxy, id="empty", args=payload)
    accepted = executor._validate_batch([call], {"t1"}, runtime)
    assert accepted[0][1].authorization_witness.parameter_grounding is None
    packet = executor._resolved_guard_input(accepted[0][1], None)
    assert packet.authorization_witness.parameter_grounding is None
    payload["proposed_tool_call"] = {"tool": "lookup", "arguments": {"x": "a"}}
    with pytest.raises(ValueError, match="parameter_grounding must be non-null"):
        executor._validate_batch([FunctionCall(function=proxy, id="nonempty", args=payload)], {"t1"}, runtime)




def test_pg_root_slice_and_only_explicit_successful_prior_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1")
    atomic(tmp_path / "case.json", {"domain": "slack", "task": "user_task_0"})
    pipeline = build("as_pg_guard", tmp_path, provider="qwen", key="test", base_url="https://example.invalid/v1", task_id="slack/user_task_0")
    executor = pipeline.elements[3].elements[0]
    frames = executor.context.frame_cache.get("Task")
    runtime = FunctionsRuntime()
    @runtime.register_function
    def lookup(x: str) -> str:
        """Read record.

        :param x: Query.
        """
        return x
    selected = executor.context.register_output(tool_call_id="previous", tool_name="lookup",
        message_index=1, output="SELECTED_PRIOR_OUTPUT")
    executor.context.register_output(tool_call_id="other", tool_name="lookup",
        message_index=3, output="UNREFERENCED_PRIOR_OUTPUT")
    arguments = {"proposed_tool_call": {"tool": "lookup", "arguments": {"x": "a"}},
        "authorization_witness": {"action_support": {"root": frames[0]["id"], "via": "Required",
            "authorization_evidence": [selected]},
            "parameter_grounding": {"roots": [frames[0]["id"]], "via": "Task constraints"}}}
    def validate(value):
        call = FunctionCall(function="submit_tool_call_with_authorization_witness", id="current", args=value)
        return executor._validate_batch([call], {f["id"] for f in frames}, runtime,
            executor.context.available_output_ids())[0][1]
    packet = executor._resolved_guard_input(validate(arguments), frames).model_dump(mode="json")
    assert packet["trusted_task_frames"] == [frames[0]]
    assert [item["output"] for item in packet["prior_tool_outputs"]] == ["SELECTED_PRIOR_OUTPUT"]
    assert "UNREFERENCED_PRIOR_OUTPUT" not in json.dumps(packet)
    invalid = copy.deepcopy(arguments)
    invalid["authorization_witness"]["action_support"]["authorization_evidence"] = ["future_or_sibling"]
    with pytest.raises(ValueError, match="unavailable authorization_evidence"):
        validate(invalid)
    invalid = copy.deepcopy(arguments)
    invalid["authorization_witness"]["action_support"]["root"] = "unknown_root"
    with pytest.raises(ValueError, match="unknown task root"):
        validate(invalid)


@pytest.mark.parametrize("method", ["no_witness_guard", "reasoning_context_guard"])
def test_sibling_snapshots_precede_all_execution(tmp_path, method):
    from types import SimpleNamespace
    source = component(method)
    source.stopped = lambda: False
    packets = []
    class Stage:
        def run(self, payload, call_id, validator=None):
            packets.append(copy.deepcopy(payload))
            decision = {"decision": "allow", "reason": "Required step"}
            if method == "no_witness_guard":
                decision.update(root="t1", via="Original Guard witness")
            result = source.Decision.model_validate(decision)
            if validator:
                validator(result)
            return result
    recorder = SimpleNamespace(record=lambda event: None)
    executor = source.GuardExecutor([{"id": "t1", "task": "Task"}], recorder, Stage())
    runtime = FunctionsRuntime()
    @runtime.register_function
    def lookup(x: str) -> str:
        """Read record.

        :param x: Query.
        """
        return "UNIQUE_EXECUTION_RESULT_" + x
    calls = [FunctionCall(function="lookup", args={"x": str(i)}, id=str(i)) for i in range(2)]
    messages = [{"role": "user", "content": [text_content_block_from_string("Task")]},
                {"role": "assistant", "content": [], "tool_calls": calls, "reasoning_content": "current only"}]
    result = executor.query("Task", runtime, env={}, messages=messages, extra_args={})
    assert len(packets) == 2 and len(result[3]) == 4
    assert "UNIQUE_EXECUTION_RESULT" not in json.dumps(packets)
    if method == "no_witness_guard":
        assert packets[0]["conversation_history"] == packets[1]["conversation_history"]
    else:
        assert packets[0]["reasoning_context"] == packets[1]["reasoning_context"] == "current only"
