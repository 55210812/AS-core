from as_all.methods import ROOT, load_module
import pytest


@pytest.mark.parametrize("selection", ["", "I cannot fulfill this request.", "unknown_tool"])
def test_tool_filter_unmatched_response_retains_no_tools(selection):
    from as_all.methods import compose
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.functions_runtime import FunctionsRuntime

    fixture = load_module("as_all.empty_filter_fixture", ROOT / "reference/tests/test_pipeline_integration.py")
    runtime = FunctionsRuntime()

    @runtime.register_function
    def read_file(filename: str) -> str:
        """Read a file.

        :param filename: File to read.
        """
        raise AssertionError("Filtering must not execute domain tools")

    client = fixture.ScriptedClient([fixture.reply(selection)])
    pipeline = compose("tool_filter", OpenAILLM(client, "gemini-2.5-flash"), "Help")
    pipeline.elements[2].query("Read", runtime, messages=[], extra_args={})
    assert not runtime.functions
    assert len(client.requests) == 1


def test_tool_filter_then_agent_usage_and_execution():
    from as_all.methods import compose
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.functions_runtime import FunctionsRuntime
    from types import SimpleNamespace

    fixture = load_module("as_all.filter_fixture", ROOT / "reference/tests/test_pipeline_integration.py")
    runtime = FunctionsRuntime()
    reads = []

    @runtime.register_function
    def read_file(filename: str) -> str:
        """Read a file.

        :param filename: File to read.
        """
        reads.append(filename)
        return "data"

    call = SimpleNamespace(id="c", function=SimpleNamespace(name="read_file", arguments='{"filename":"f"}'))
    client = fixture.ScriptedClient([
        fixture.reply("read_file"), fixture.reply(None, [call]), fixture.reply("done")])
    result = compose("tool_filter", OpenAILLM(client, "gemini-2.5-flash"), "Help").query("Read", runtime)
    assert reads == ["f"]
    assert len(client.requests) == 3
    assert result[-1]["token_usage"] == {
        "requests": 3, "input_tokens": 6, "output_tokens": 9, "total_tokens": 15}


def test_ipiguard_real_executor_fake_then_real():
    reference = load_module("as_all.integration_fixture", ROOT / "reference/tests/test_pipeline_integration.py")
    reference.test_ipiguard_public_pipeline_fake_then_real()


def test_detector_full_pipeline_without_network(monkeypatch):
    reference = load_module("as_all.detector_fixture", ROOT / "reference/tests/test_pipeline_integration.py")
    monkeypatch.setattr("as_all.methods.detector_path", lambda: "protectai/deberta-v3-base-prompt-injection-v2")
    from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector, TransformersBasedPIDetector
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.functions_runtime import FunctionsRuntime
    from types import SimpleNamespace
    import json

    detected = []
    def init(self, model_name, safe_label, threshold, mode):
        PromptInjectionDetector.__init__(self, mode)
        assert (model_name, safe_label, threshold, mode) == (
            "protectai/deberta-v3-base-prompt-injection-v2", "SAFE", .5, "message")
        self.safe_label, self.threshold = safe_label, threshold
        self.pipeline = lambda text: detected.append(text) or [{"label": "INJECTION", "score": .9}]
    monkeypatch.setattr(TransformersBasedPIDetector, "__init__", init)
    runtime = FunctionsRuntime()
    @runtime.register_function
    def read_file(filename: str) -> str:
        """Read a file.

        :param filename: File to read.
        """
        return "fixture text"
    call = SimpleNamespace(id="c", function=SimpleNamespace(name="read_file", arguments='{"filename":"f"}'))
    client = reference.ScriptedClient([reference.reply(None, [call]), reference.reply("done")])
    from as_all.methods import compose
    compose("transformers_pi_detector", OpenAILLM(client, "gemini-2.5-flash"), "Help").query("Read", runtime)
    assert detected == ["fixture text"]
    assert "Data omitted" in json.dumps(client.requests[-1]["messages"])
