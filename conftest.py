"""Keep the method and upstream-reference suites offline and independent."""
import httpx
import pytest

from agentdojo.agent_pipeline.llms import openai_llm

_ORIGINAL_REQUEST = openai_llm.chat_completion_request


@pytest.fixture(autouse=True)
def offline_runtime(monkeypatch):
    # Source builders unwrap this method's retry decorator. Reference tests also
    # inspect the original wrapper, so start each test from the imported source.
    monkeypatch.setattr(openai_llm, "chat_completion_request", _ORIGINAL_REQUEST)
    monkeypatch.setattr("as_all.methods.detector_path",
                        lambda: "protectai/deberta-v3-base-prompt-injection-v2")

    def no_network(*args, **kwargs):
        raise AssertionError("Offline tests require MockTransport, not real HTTP")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
