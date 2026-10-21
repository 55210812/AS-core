"""Real framework/dependency integration with scripted responses and toy tools."""

import json
from types import SimpleNamespace

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector, TransformersBasedPIDetector
from agentdojo.functions_runtime import FunctionsRuntime


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = next(self.responses)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


def reply(text, calls=None):
    return SimpleNamespace(content=text, tool_calls=calls)


def config(model, defense):
    return PipelineConfig(
        llm=model, model_id=None, defense=defense,
        system_message_name=None, system_message="Help the user.",
    )


def test_ipiguard_public_pipeline_fake_then_real():
    runtime = FunctionsRuntime()
    writes = []

    @runtime.register_function
    def read_file(filename: str) -> str:
        """Read a file.

        :param filename: File to read.
        """
        return "data"

    @runtime.register_function
    def write_note(text: str) -> str:
        """Write a note.

        :param text: Note text.
        """
        writes.append(text)
        return "written"

    client = ScriptedClient([
        reply("{}"),
        reply(json.dumps({"tool_calls": [
            {"id": 1, "function_name": "read_file", "args": {"filename": "fixture"}, "depends_on": []},
            {"id": 2, "function_name": "write_note", "args": {"text": "value"}, "depends_on": [1]},
        ]})),
        reply('{"new_tool_calls":[{"function_name":"write_note","args":{"text":"value"}}]}'),
        reply('{"new_tool_calls":[]}'),
        reply("done"),
    ])
    model = OpenAILLM(client, "qwen3.5-flash")
    model.name = "qwen3.5-flash"
    pipeline = AgentPipeline.from_config(config(model, "ipiguard_official"))
    result = pipeline.query("Read and write", runtime, extra_args={})
    assert writes == ["value"]
    assert len(client.requests) == 5
    assert all(r["temperature"] == 0 for r in client.requests)
    assert any(m["role"] == "tool" and m.get("error") == "null" for m in result[3])
    assert any(m["role"] == "tool" and m["content"] == "null" for m in client.requests[-1]["messages"])


def test_detector_public_pipeline_uses_official_defaults_and_content(monkeypatch):
    detected = []
    settings = {}

    def init(self, model_name, safe_label, threshold, mode, raise_on_injection=False):
        PromptInjectionDetector.__init__(self, mode, raise_on_injection)
        self.model_name, self.safe_label, self.threshold = model_name, safe_label, threshold
        settings.update(model_name=model_name, safe_label=safe_label, threshold=threshold, mode=mode)
        def classify(text):
            detected.append(text)
            return [{"label": "INJECTION", "score": 0.9}]
        self.pipeline = classify

    monkeypatch.setattr(TransformersBasedPIDetector, "__init__", init)
    runtime = FunctionsRuntime()

    @runtime.register_function
    def read_file(filename: str) -> str:
        """Read a file.

        :param filename: File to read.
        """
        return "fixture text"

    call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="read_file", arguments='{"filename":"fixture"}'))
    client = ScriptedClient([reply(None, [call]), reply("done")])
    model = OpenAILLM(client, "qwen3.5-flash")
    model.name = "qwen3.5-flash"
    AgentPipeline.from_config(config(model, "transformers_pi_detector")).query("Read", runtime, extra_args={})
    assert settings == {"model_name": "protectai/deberta-v3-base-prompt-injection-v2", "safe_label": "SAFE", "threshold": 0.5, "mode": "message"}
    assert detected == ["fixture text"]
    assert "Data omitted" in client.requests[-1]["messages"][-1]["content"]
