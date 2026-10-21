from __future__ import annotations

import hashlib
import json
import os
import uuid
from ast import literal_eval
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from semantic_action_frame_guard import OpenAITransport, TraceRecorder
from semantic_action_frame_guard.v5_models import AuthorizationDecision
from semantic_action_frame_guard.v6_taskshield_history import (
    V6TaskShieldHistoryPipeline,
)
from semantic_action_frame_guard.v6_taskshield_history_models import (
    ConversationHistoryMessage,
    GuardInput,
)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _jsonable_tool_output(value: FunctionReturnType, fallback: str) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable_tool_output(item, str(item)) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable_tool_output(item, str(item)) for item in value]
    return fallback


def _resolved_message_text(message: ChatMessage) -> tuple[str, str]:
    content = message.get("content")
    visible = get_text_content_as_str(content) if content else ""
    if visible.strip():
        return visible, "content"
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning, "reasoning_content"
    return "", "none"


def _taskshield_history(
    messages: Sequence[ChatMessage],
    *,
    excluded_roles: frozenset[str] = frozenset(),
) -> tuple[list[ConversationHistoryMessage], list[dict[str, Any]]]:
    history: list[ConversationHistoryMessage] = []
    sources: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message["role"] in excluded_roles:
            continue
        text, source = _resolved_message_text(message)
        entry: dict[str, Any] = {"role": message["role"], "content": text}
        if message["role"] == "assistant" and message.get("tool_calls"):
            entry["tool_calls"] = [
                {"call_id": call.id, "tool_name": call.function, "tool_args": dict(call.args)}
                for call in message["tool_calls"]
            ]
        if message["role"] == "tool":
            entry["tool_call_id"] = message.get("tool_call_id")
            entry["error"] = message.get("error")
        history.append(ConversationHistoryMessage.model_validate(entry))
        sources.append({"index": index, "source": source})
    return history, sources


class SemanticFrameAlignmentV6TaskShieldHistoryToolsExecutor(BasePipelineElement):
    """Authorize calls with Task Shield-shaped conversation history and ID selection."""

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        *,
        client: Any = None,
        defense_name: str = "semantic_frame_alignment_v6_reference_selection_taskshield_history",
        excluded_history_roles: frozenset[str] = frozenset(),
    ) -> None:
        model = os.getenv("SAFG_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
        transport = OpenAITransport(
            model,
            api_key=os.getenv("SAFG_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("SAFG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            reasoning_effort=os.getenv("SAFG_REASONING_EFFORT"),
            client=client,
        )
        trace_root = Path(
            os.getenv("SAFG_TRACE_DIR", "semantic-action-frame-results/traces/agentdojo/v6-taskshield-history")
        )
        self.recorder = TraceRecorder(trace_root)
        self.guard = V6TaskShieldHistoryPipeline(
            transport,
            self.recorder,
            task_frame_cache_path=os.getenv("SAFG_TASK_FRAME_CACHE"),
        )
        self.output_formatter = tool_output_formatter
        self.defense_name = defense_name
        self.excluded_history_roles = frozenset(excluded_history_roles)

    @staticmethod
    def _task_id(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:20]

    def _guard_input(
        self,
        query: str,
        tool_call: Any,
        runtime: FunctionsRuntime,
        history: list[ConversationHistoryMessage],
        related_content: str,
        task_id: str,
    ) -> GuardInput:
        function = runtime.functions[tool_call.function]
        call_id = tool_call.id or str(uuid.uuid4())
        return GuardInput(
            task_id=task_id,
            trusted_user_task=query,
            proposed_tool_call={
                "tool_call_id": call_id,
                "tool_name": function.name,
                "tool_description": function.description,
                "arguments": dict(tool_call.args),
            },
            conversation_history=history,
            related_content=related_content,
        )

    def _feedback(self, authorization: Any) -> str:
        evidence = json.dumps(
            {
                "action_support": [item.model_dump(mode="json") for item in authorization.action_support],
                "parameter_grounding": {
                    key: value.model_dump(mode="json")
                    for key, value in authorization.parameter_grounding.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        prefix = "Repair required" if authorization.decision == AuthorizationDecision.REPAIR else "Blocked"
        return (
            f"{prefix} by {self.defense_name}. The proposed call was not executed. "
            "Authorization evidence: " + evidence
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        calls = messages[-1]["tool_calls"]
        if not calls:
            return query, runtime, env, messages, extra_args

        history, history_sources = _taskshield_history(
            messages[:-1], excluded_roles=self.excluded_history_roles
        )
        related_content, related_source = _resolved_message_text(messages[-1])
        task_id = self._task_id(query)
        results: list[ChatToolResultMessage] = []
        for tool_call in calls:
            if tool_call.function == EMPTY_FUNCTION_NAME or tool_call.function not in runtime.functions:
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_call.function} provided.",
                    )
                )
                continue
            for key, value in tool_call.args.items():
                if isinstance(value, str) and is_string_list(value):
                    tool_call.args[key] = literal_eval(value)
            guard_input = self._guard_input(
                query,
                tool_call,
                runtime,
                history,
                related_content,
                task_id,
            )
            try:
                _, authorization = self.guard.analyze(guard_input)
            except Exception as exc:
                authorization = None
                error = f"V6 authorization failed: {type(exc).__name__}: {exc}"
            else:
                error = None
            event: dict[str, Any] = {
                "defense_name": self.defense_name,
                "event_type": "runtime_action_authorization_decision",
                "tool_name": tool_call.function,
                "authorization": authorization.model_dump(mode="json") if authorization else None,
                "action": authorization.decision.value if authorization else "block",
                "conversation_history": [item.model_dump(mode="json") for item in history],
                "related_content": related_content,
                "history_content_sources": history_sources,
                "related_content_source": related_source,
                "reference_selection_mode": "all",
                "reference_catalog": getattr(self.guard, "last_reference_catalog", {}),
                "history_resolution": getattr(self.guard, "last_history_resolution", {}),
                "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
            }
            if error:
                event["error"] = error
            extra_args.setdefault("defense_events", []).append(event)
            if authorization is None or authorization.decision != AuthorizationDecision.ALLOW:
                feedback = error or self._feedback(authorization)
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=feedback,
                    )
                )
                self.recorder.record(
                    {
                        "stage": "agentdojo_execution",
                        "tool_call_id": guard_input.proposed_tool_call.tool_call_id,
                        "decision": authorization.model_dump(mode="json") if authorization else {"decision": "block"},
                        "feedback": feedback,
                        "executed": False,
                        "tool_name": tool_call.function,
                        "tool_args": dict(tool_call.args),
                    }
                )
                continue

            value, tool_error = runtime.run_function(env, tool_call.function, tool_call.args)
            formatted_output = self.output_formatter(value)
            results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(formatted_output)],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=tool_error,
                )
            )
            self.recorder.record(
                {
                    "stage": "agentdojo_execution",
                    "tool_call_id": guard_input.proposed_tool_call.tool_call_id,
                    "decision": authorization.model_dump(mode="json"),
                    "executed": True,
                    "tool_name": tool_call.function,
                    "tool_args": dict(tool_call.args),
                    "raw_tool_output": formatted_output,
                    "native_tool_output": _jsonable_tool_output(value, formatted_output),
                    "tool_error": tool_error,
                }
            )
        return query, runtime, env, [*messages, *results], extra_args


__all__ = ["SemanticFrameAlignmentV6TaskShieldHistoryToolsExecutor"]
