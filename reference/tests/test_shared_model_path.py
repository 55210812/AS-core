"""No-network checks of the model adapter used by Injection Detection."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentdojo.agent_pipeline.llms import openai_llm as llm_module
from agentdojo.agent_pipeline import tool_execution
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import text_content_block_from_string


REFERENCE = Path(__file__).resolve().parents[1] / "reference/ipiguard"


def upstream_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes
    ], type_ignores=[])
    result = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), result)
    return result


def completion(arguments):
    call = SimpleNamespace(
        id="call-1", function=SimpleNamespace(name="read_file", arguments=arguments)
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def test_malformed_arguments_do_not_get_an_extra_generation(monkeypatch):
    requests = []

    def request(*args):
        requests.append(args)
        return completion("{invalid")

    monkeypatch.setattr(llm_module, "chat_completion_request", request)
    model = llm_module.OpenAILLM(None, "qwen3.5-flash")
    history = [{"role": "user", "content": [text_content_block_from_string("Read the file")]}]
    with pytest.raises(ValueError):
        model.query("Read the file", SimpleNamespace(functions={}), messages=history, extra_args={})
    assert len(requests) == 1


def test_valid_response_keeps_tool_arguments_and_usage(monkeypatch):
    requests = []

    def request(*args):
        requests.append(args)
        return completion('{"filename": "notes.txt"}')

    monkeypatch.setattr(llm_module, "chat_completion_request", request)
    model = llm_module.OpenAILLM(None, "qwen3.5-flash")
    history = [{"role": "user", "content": [text_content_block_from_string("Read the file")]}]
    accounting = {}
    _, _, _, messages, _ = model.query(
        "Read the file", SimpleNamespace(functions={}), messages=history, extra_args=accounting
    )
    assert len(requests) == 1
    assert messages[-1]["tool_calls"][0].args == {"filename": "notes.txt"}
    assert accounting["token_usage"]["total_tokens"] == 12


@pytest.mark.parametrize("temperature", [0.0, 0.25, None])
def test_request_and_wire_format_match_pinned_fork(temperature):
    upstream = upstream_definitions(
        REFERENCE / "llms/openai_llm.py",
        {"chat_completion_request", "_message_to_openai", "_tool_call_to_openai"}, vars(llm_module)
    )
    call = FunctionCall(function="read_file", args={"filename": "notes.txt"}, id="call-1")
    old = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "content": "result", "error": "null", "tool_call_id": call.id, "tool_call": call},
    ]
    modern = [{**m, "content": [text_content_block_from_string(m["content"])]
               if m["content"] is not None else None} for m in old]
    a = [upstream["_message_to_openai"](m) for m in old]
    b = [llm_module._message_to_openai(m, "qwen3.5-flash") for m in modern]
    assert a == b
    requests = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: requests.append({k: v for k, v in kw.items() if v is not llm_module.NOT_GIVEN})
    )))
    upstream["chat_completion_request"](client, "qwen3.5-flash", a, [], temperature)
    llm_module.chat_completion_request.__wrapped__(client, "qwen3.5-flash", b, [], None, temperature)
    assert requests[0] == requests[1]


@pytest.mark.parametrize("raw", ['{"filename":"a"}', json.dumps('{"filename":"a"}')])
def test_tool_arguments_decode_matches_pinned_fork(raw):
    upstream = upstream_definitions(
        REFERENCE / "llms/openai_llm.py", {"_openai_to_tool_call"}, vars(llm_module)
    )
    call = completion(raw).choices[0].message.tool_calls[0]
    assert upstream["_openai_to_tool_call"](call) == llm_module._openai_to_tool_call(call)


def test_tools_executor_does_not_coerce_string_lists():
    upstream = upstream_definitions(
        REFERENCE / "tool_execution.py", {"ToolsExecutor"}, vars(tool_execution)
    )
    attempts = [[], []]
    for index, executor in enumerate((upstream["ToolsExecutor"](), tool_execution.ToolsExecutor())):
        def run(env, name, args):
            attempts[index].append(copy.deepcopy(args))
            return "ok", None
        runtime = SimpleNamespace(functions={"write": SimpleNamespace(name="write")}, run_function=run)
        call = FunctionCall(function="write", args={"recipients": "['a@example.test']"}, id="call-1")
        executor.query("task", runtime, messages=[{"role": "assistant", "content": None, "tool_calls": [call]}])
    assert attempts[0] == attempts[1] == [{"recipients": "['a@example.test']"}]
