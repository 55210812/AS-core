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
from semantic_action_frame_guard import (
    OpenAITransport,
    TraceRecorder,
    V6AuthorizationDecision,
    V6ExecutionRecord,
    V6ExecutionStatus,
    V6GuardInput,
    V6SemanticPipeline,
    V6ToolCallSpec,
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


def _visible_plan(message: ChatMessage) -> str:
    content = message.get("content") or []
    return get_text_content_as_str(content)


class SemanticFrameAlignmentV6ToolsExecutor(BasePipelineElement):
    """Authorize calls with deterministic argument and action-context candidates."""

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        *,
        client: Any = None,
        defense_name: str = "semantic_frame_alignment_v6_action_context",
        selection_mode: str | None = None,
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
            os.getenv("SAFG_TRACE_DIR", "semantic-action-frame-results/traces/agentdojo/v6")
        )
        self.recorder = TraceRecorder(trace_root)
        self.guard = V6SemanticPipeline(
            transport,
            self.recorder,
            task_frame_cache_path=os.getenv("SAFG_TASK_FRAME_CACHE"),
            selection_mode=selection_mode,
        )
        self.output_formatter = tool_output_formatter
        self.defense_name = defense_name
        self.selection_mode = selection_mode

    @staticmethod
    def _task_id(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:20]

    def _reset_for_task(self, query: str, extra_args: dict) -> str:
        task_id = self._task_id(query)
        if extra_args.get("semantic_frame_v6_task_id") != task_id:
            extra_args.pop("semantic_frame_v6_execution_history", None)
            extra_args.pop("semantic_frame_v6_history_metadata", None)
            extra_args.pop("semantic_frame_v6_pending_calls", None)
        extra_args["semantic_frame_v6_task_id"] = task_id
        return task_id

    @staticmethod
    def _update_plans(messages: Sequence[ChatMessage], extra_args: dict) -> None:
        pending = list(extra_args.get("semantic_frame_v6_pending_calls", []))
        if not pending or not messages or messages[-1]["role"] != "assistant":
            return
        plan = _visible_plan(messages[-1])
        metadata = extra_args.setdefault("semantic_frame_v6_history_metadata", {})
        for call_id in pending:
            if call_id in metadata and not metadata[call_id].get("plan"):
                metadata[call_id]["plan"] = plan
        extra_args["semantic_frame_v6_pending_calls"] = []

    def _guard_input(
        self,
        query: str,
        tool_call,
        runtime: FunctionsRuntime,
        history_snapshot: list[dict[str, Any]],
        history_metadata: dict[str, dict[str, Any]],
        task_id: str,
        current_visible_plan: str,
    ) -> V6GuardInput:
        function = runtime.functions[tool_call.function]
        call_id = tool_call.id or str(uuid.uuid4())
        return V6GuardInput(
            task_id=task_id,
            trusted_user_task=query,
            proposed_tool_call=V6ToolCallSpec(
                tool_call_id=call_id,
                tool_name=function.name,
                tool_description=function.description,
                arguments=dict(tool_call.args),
            ),
            execution_tool_history=[V6ExecutionRecord.model_validate(item) for item in history_snapshot],
            history_metadata=history_metadata,
            current_visible_plan=current_visible_plan,
        )

    def _feedback(self, authorization) -> str:
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
        prefix = "Repair required" if authorization.decision == V6AuthorizationDecision.REPAIR else "Blocked"
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

        task_id = self._reset_for_task(query, extra_args)
        self._update_plans(messages, extra_args)
        current_visible_plan = _visible_plan(messages[-1])
        live_history = extra_args.setdefault("semantic_frame_v6_execution_history", [])
        history_metadata = extra_args.setdefault("semantic_frame_v6_history_metadata", {})
        history_snapshot = [dict(item) for item in live_history]
        results: list[ChatToolResultMessage] = []
        pending_for_next_turn: list[str] = []

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
                history_snapshot,
                dict(history_metadata),
                task_id,
                current_visible_plan,
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
                "argument_source_candidates": {
                    key: [item.model_dump(mode="json") for item in values]
                    for key, values in getattr(self.guard, "last_argument_source_candidates", {}).items()
                },
                "action_context_candidates": [
                    item.model_dump(mode="json")
                    for item in getattr(self.guard, "last_action_context_candidates", [])
                ],
                "reference_selection_mode": self.selection_mode,
                "reference_catalog": getattr(self.guard, "last_reference_catalog", {}),
                "retrieval_diagnostics": getattr(self.guard, "last_retrieval_diagnostics", {}),
                "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
            }
            if error:
                event["error"] = error
            extra_args.setdefault("defense_events", []).append(event)
            if authorization is None or authorization.decision != V6AuthorizationDecision.ALLOW:
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
            record = V6ExecutionRecord(
                call_id=guard_input.proposed_tool_call.tool_call_id,
                tool_name=tool_call.function,
                arguments=dict(tool_call.args),
                status=V6ExecutionStatus.SUCCESS if tool_error is None else V6ExecutionStatus.ERROR,
                output=_jsonable_tool_output(value, formatted_output),
                error=tool_error,
            )
            live_history.append(record.model_dump(mode="json"))
            history_metadata[record.call_id] = {
                "tool_description": runtime.functions[tool_call.function].description,
                "plan": "",
                "order": len(live_history) - 1,
            }
            if tool_error is None:
                pending_for_next_turn.append(record.call_id)
            self.recorder.record(
                {
                    "stage": "agentdojo_execution",
                    "tool_call_id": record.call_id,
                    "decision": authorization.model_dump(mode="json"),
                    "executed": True,
                    "tool_name": record.tool_name,
                    "tool_args": record.arguments,
                    "raw_tool_output": formatted_output,
                    "native_tool_output": record.output,
                    "tool_error": tool_error,
                }
            )
        extra_args["semantic_frame_v6_pending_calls"] = pending_for_next_turn
        return query, runtime, env, [*messages, *results], extra_args


__all__ = ["SemanticFrameAlignmentV6ToolsExecutor"]
