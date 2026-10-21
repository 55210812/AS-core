"""Original Qwen Chat Completions protocol with explicit stage accounting."""
import os
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit
from openai.types.chat import ChatCompletion
from as_all.transport import NativeClient, _is_not_given, _thinking_from_extra


class QwenClient(NativeClient):
    model = "qwen3.5-flash"
    provider = "qwen"
    usage_field = "usage"

    def __init__(self, **kwargs):
        kwargs.setdefault("key", os.environ.get("QWEN_API_KEY"))
        kwargs.setdefault("base_url", os.environ.get("QWEN_BASE_URL"))
        if not kwargs["key"] or not kwargs["base_url"]:
            raise ValueError("QWEN_API_KEY and QWEN_BASE_URL are required")
        super().__init__(**kwargs)

    @staticmethod
    def _generate_content_url(base_url):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("QWEN_BASE_URL must be an HTTP(S) API base without credentials or query")
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    def _headers(self):
        return {"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}

    def _request(self, kwargs):
        allowed = {"model", "messages", "temperature", "tools", "tool_choice", "extra_body", "max_tokens",
                   "max_completion_tokens", "response_format", "stream"}
        body = {key: deepcopy(value) for key, value in kwargs.items() if not _is_not_given(value)}
        if set(body) - allowed:
            raise ValueError("Unsupported Qwen request fields: " + str(sorted(set(body) - allowed)))
        if body.get("model") != self.model:
            raise ValueError("ModelIdentityMismatch: unexpected Qwen request model")
        if body.get("stream"):
            raise ValueError("Streaming/partial response resends are unsupported")
        if body.get("temperature", 0) != 0:
            raise ValueError("Qwen temperature must be0")
        supplied = _thinking_from_extra(body.pop("extra_body", None))
        if supplied is not None and supplied != self._thinking_budget:
            raise ValueError("Thinking configuration conflicts with explicit Qwen stage")
        body.update(temperature=0, enable_thinking=self.stage == "agent", stream=False)
        if not body.get("messages"):
            raise ValueError("Qwen messages cannot be empty")
        # Keep original tools, proxy envelope, content and argument bytes unchanged.
        return body

    def _completion(self, data):
        if data.get("model") != self.model:
            raise ValueError("ModelIdentityMismatch: unexpected Qwen response model")
        result = ChatCompletion.model_validate(data)
        if len(result.choices) != 1:
            raise ValueError("Qwen response must contain exactly one choice")
        choice = result.choices[0]
        message = choice.message
        if self.stage == "defense" and getattr(message, "reasoning_content", None):
            raise ValueError("ThinkingChannelMismatch: defense returned reasoning content")
        if not message.content and not message.tool_calls:
            raise ValueError("EmptyModelResponse: Qwen returned no text or tool call")
        if choice.finish_reason not in ("stop", "tool_calls", "length"):
            raise ValueError("QwenResponseRejected: " + str(choice.finish_reason))
        if message.tool_calls and choice.finish_reason == "length":
            raise ValueError("QwenResponseRejected: partial tool call")
        return result
