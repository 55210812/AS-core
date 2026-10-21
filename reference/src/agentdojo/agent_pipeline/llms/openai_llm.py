import json
from collections.abc import Sequence
from typing import overload

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionReasoningEffort,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _tool_call_to_openai(tool_call: FunctionCall) -> ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args),
        },
    )


_REASONING_MODELS = {"o1", "o3"}
_DEEPSEEK_MODELS = {"deepseek-v4-flash", "deepseek-v4-flash-0731"}
_OPENAI_COMPAT_SYSTEM_MODELS = _DEEPSEEK_MODELS | {"qwen3.5-flash"}


class ToolArgumentsFormatError(ValueError):
    """Raised when a model-produced tool call has invalid JSON arguments."""


def _is_reasoning_model(model_name: str) -> bool:
    return any(model in model_name for model in _REASONING_MODELS)


def _is_deepseek_model(model_name: str) -> bool:
    return model_name in _DEEPSEEK_MODELS


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatSystemMessage,
) -> list[ChatCompletionContentPartTextParam]: ...


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatAssistantMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None: ...


def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatAssistantMessage | ChatSystemMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None:
    if message["content"] is None:
        return None
    return [ChatCompletionContentPartTextParam(type="text", text=el["content"] or "") for el in message["content"]]


def _message_to_openai(message: ChatMessage, model_name: str) -> ChatCompletionMessageParam:
    """Pinned IPIGuard fork wire format with current text-block input types."""
    content = message["content"]
    if content is not None:
        content = get_text_content_as_str(content)
    match message["role"]:
        case "system":
            return ChatCompletionSystemMessageParam(role="system", content=content)
        case "user":
            return ChatCompletionUserMessageParam(role="user", content=content)
        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                return ChatCompletionAssistantMessageParam(
                    role="assistant", content=content,
                    tool_calls=[_tool_call_to_openai(call) for call in message["tool_calls"]],
                )
            return ChatCompletionAssistantMessageParam(role="assistant", content=content)
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or content,
                tool_call_id=message["tool_call_id"], role="tool",
                name=message["tool_call"].function,
            )
        case _:
            raise ValueError(f"Invalid message type: {message}")


def _openai_to_tool_call(tool_call: ChatCompletionMessageToolCall) -> FunctionCall:
    arguments = json.loads(tool_call.function.arguments)
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return FunctionCall(function=tool_call.function.name, args=arguments, id=tool_call.id)


def _assistant_message_to_content(message: ChatCompletionMessage) -> list[MessageContentBlock] | None:
    if message.content is None:
        return None
    return [text_content_block_from_string(message.content)]


def _openai_to_assistant_message(message: ChatCompletionMessage) -> ChatAssistantMessage:
    if message.tool_calls is not None:
        tool_calls = [_openai_to_tool_call(tool_call) for tool_call in message.tool_calls]
    else:
        tool_calls = None
    assistant_message = ChatAssistantMessage(
        role="assistant", content=_assistant_message_to_content(message), tool_calls=tool_calls
    )
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None:
        assistant_message["reasoning_content"] = reasoning_content
    return assistant_message


def _stream_to_assistant_message(completion) -> ChatAssistantMessage:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_call_buffers: dict[int, dict[str, str | None]] = {}
    for chunk in completion:
        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                reasoning_parts.append(reasoning_content)
            for tool_call in getattr(delta, "tool_calls", None) or []:
                index = getattr(tool_call, "index", None)
                if index is None:
                    index = len(tool_call_buffers)
                buffer = tool_call_buffers.setdefault(index, {"id": None, "name": "", "arguments": ""})
                tool_call_id = getattr(tool_call, "id", None)
                if tool_call_id:
                    buffer["id"] = tool_call_id
                function = getattr(tool_call, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    if name:
                        buffer["name"] = (buffer.get("name") or "") + name
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        buffer["arguments"] = (buffer.get("arguments") or "") + arguments

    tool_calls = []
    for buffer in tool_call_buffers.values():
        function_name = buffer.get("name") or ""
        if not function_name:
            continue
        raw_args = buffer.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError) as error:
            raise ToolArgumentsFormatError(
                f"Invalid JSON arguments for tool call {buffer.get('id')} ({function_name}): {error}"
            ) from error
        if not isinstance(arguments, dict):
            raise ToolArgumentsFormatError(
                f"Tool call {buffer.get('id')} ({function_name}) arguments must decode to a JSON object, "
                f"got {type(arguments).__name__}"
            )
        tool_calls.append(FunctionCall(function=function_name, args=arguments, id=buffer.get("id")))

    content = "".join(content_parts)
    content_blocks = [text_content_block_from_string(content)] if content else None
    assistant_message = ChatAssistantMessage(role="assistant", content=content_blocks, tool_calls=tool_calls or None)
    if reasoning_parts:
        assistant_message["reasoning_content"] = "".join(reasoning_parts)
    return assistant_message


def _completion_to_assistant_message(completion) -> ChatAssistantMessage:
    if hasattr(completion, "choices"):
        return _openai_to_assistant_message(completion.choices[0].message)
    if isinstance(completion, str):
        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(completion)] if completion else None,
            tool_calls=None,
        )
    return _stream_to_assistant_message(completion)


def _record_token_usage(extra_args: dict, completion) -> None:
    """Accumulate OpenAI-compatible token usage for benchmark accounting."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    token_usage = extra_args.setdefault(
        "token_usage",
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        },
    )
    token_usage["prompt_tokens"] = int(token_usage.get("prompt_tokens", 0) or 0) + prompt_tokens
    token_usage["completion_tokens"] = int(token_usage.get("completion_tokens", 0) or 0) + completion_tokens
    token_usage["total_tokens"] = int(token_usage.get("total_tokens", 0) or 0) + total_tokens
    token_usage["llm_calls"] = int(token_usage.get("llm_calls", 0) or 0) + 1


def _function_to_openai(f: Function) -> ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return ChatCompletionToolParam(type="function", function=function_definition)


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None = 0.0,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature,
        reasoning_effort=reasoning_effort or NOT_GIVEN,
    )


class OpenAILLM(BasePipelineElement):
    """LLM pipeline element that uses OpenAI's API.

    Args:
        client: The OpenAI client.
        model: The model name.
        temperature: The temperature to use for generation.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        if self.model == "gpt-5.5":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools or NOT_GIVEN,
                tool_choice="auto" if openai_tools else NOT_GIVEN,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort or NOT_GIVEN,
                stream=True,
            )
        else:
            completion = chat_completion_request(
                self.client, self.model, openai_messages, openai_tools, self.reasoning_effort, self.temperature
            )
        _record_token_usage(extra_args, completion)
        output = _completion_to_assistant_message(completion)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class OpenAILLMToolFilter(BasePipelineElement):
    def __init__(self, prompt: str, client: openai.OpenAI, model: str, temperature: float | None = 0.0) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        self.temperature = temperature

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        messages = [*messages, ChatUserMessage(role="user", content=[text_content_block_from_string(self.prompt)])]
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools or NOT_GIVEN,
            tool_choice="none",
            temperature=self.temperature,
            stream=True if self.model == "gpt-5.5" else NOT_GIVEN,
        )
        _record_token_usage(extra_args, completion)
        output = _completion_to_assistant_message(completion)

        new_tools = {}
        for tool_name, tool in runtime.functions.items():
            if output["content"] is not None and tool_name in get_text_content_as_str(output["content"]):
                new_tools[tool_name] = tool

        runtime.update_functions(new_tools)

        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
