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
    V5AuthorizationDecision,
    V5ExecutionRecord,
    V5ExecutionStatus,
    V5GuardInput,
    V5RuntimeAuthorizationGuard,
    V5SemanticPipeline,
    V5ToolCallSpec,
)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)


def _jsonable_tool_output(value: FunctionReturnType, fallback: str) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {
            str(key): _jsonable_tool_output(item, str(item))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable_tool_output(item, str(item)) for item in value]
    return fallback


class SemanticFrameAlignmentV5ToolsExecutor(BasePipelineElement):
    """Authorize tool calls using V5 task frames and real execution history."""

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        *,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        guard_model = model or os.getenv("SAFG_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
        if guard_model == "qwen3.5-flash":
            api_key = os.getenv("SAFG_API_KEY") or os.getenv("QWEN_API_KEY")
            base_url = os.getenv("SAFG_BASE_URL") or os.getenv("QWEN_BASE_URL")
            if not api_key or not base_url:
                raise ValueError("QWEN_API_KEY and QWEN_BASE_URL must be set for qwen3.5-flash Guard")
        else:
            api_key = os.getenv("SAFG_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("SAFG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        transport = OpenAITransport(
            guard_model,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=os.getenv("SAFG_REASONING_EFFORT"),
            client=client,
        )
        trace_root = Path(
            os.getenv(
                "SAFG_TRACE_DIR",
                "semantic-action-frame-results/traces/agentdojo/v5_without_specific_root",
            )
        )
        self.recorder = TraceRecorder(trace_root)
        self.guard = V5RuntimeAuthorizationGuard(
            V5SemanticPipeline(transport, self.recorder)
        )
        self.output_formatter = tool_output_formatter

    @staticmethod
    def _task_id(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:20]

    def _reset_for_task(self, query: str, extra_args: dict) -> str:
        task_id = self._task_id(query)
        if extra_args.get("semantic_frame_v5_without_specific_root_task_id") != task_id:
            extra_args.pop("semantic_frame_v5_without_specific_root_execution_history", None)
        extra_args["semantic_frame_v5_without_specific_root_task_id"] = task_id
        return task_id

    def _guard_input(
        self,
        query: str,
        tool_call,
        runtime: FunctionsRuntime,
        history_snapshot: list[dict[str, Any]],
        task_id: str,
    ) -> V5GuardInput:
        function = runtime.functions[tool_call.function]
        call_id = tool_call.id or str(uuid.uuid4())
        return V5GuardInput(
            task_id=task_id,
            trusted_user_task=query,
            proposed_tool_call=V5ToolCallSpec(
                tool_call_id=call_id,
                tool_name=function.name,
                tool_description=function.description,
                arguments=dict(tool_call.args),
            ),
            execution_tool_history=[
                V5ExecutionRecord.model_validate(item) for item in history_snapshot
            ],
        )

    @staticmethod
    def _feedback(authorization) -> str:
        evidence = json.dumps(
            {
                "action_support": [
                    item.model_dump(mode="json")
                    for item in authorization.action_support
                ],
                "parameter_grounding": {
                    key: value.model_dump(mode="json")
                    for key, value in authorization.parameter_grounding.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if authorization.decision == V5AuthorizationDecision.REPAIR:
            return (
                "Repair required by semantic_frame_alignment_v5_without_specific_root. "
                "The proposed call was not executed. Authorization evidence: "
                + evidence
            )
        return (
            "Blocked by semantic_frame_alignment_v5_without_specific_root. "
            "The proposed call was not executed. Authorization evidence: "
            + evidence
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
        live_history = extra_args.setdefault(
            "semantic_frame_v5_without_specific_root_execution_history", []
        )
        history_snapshot = [dict(item) for item in live_history]
        results: list[ChatToolResultMessage] = []

        for tool_call in calls:
            if (
                tool_call.function == EMPTY_FUNCTION_NAME
                or tool_call.function not in runtime.functions
            ):
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
                query, tool_call, runtime, history_snapshot, task_id
            )
            authorization = self.guard.evaluate(guard_input)
            extra_args.setdefault("defense_events", []).append(
                {
                    "defense_name": "semantic_frame_alignment_v5_without_specific_root",
                    "event_type": "runtime_action_authorization_decision",
                    "action": authorization.decision.value,
                    "tool_name": tool_call.function,
                    "authorization": authorization.model_dump(mode="json"),
                    "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
                }
            )

            if authorization.decision != V5AuthorizationDecision.ALLOW:
                feedback = self._feedback(authorization)
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
                        "decision": authorization.model_dump(mode="json"),
                        "feedback": feedback,
                        "executed": False,
                        "tool_name": tool_call.function,
                        "tool_args": dict(tool_call.args),
                    }
                )
                continue

            value, error = runtime.run_function(
                env, tool_call.function, tool_call.args
            )
            formatted_output = self.output_formatter(value)
            results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(formatted_output)],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=error,
                )
            )
            record = V5ExecutionRecord(
                call_id=guard_input.proposed_tool_call.tool_call_id,
                tool_name=tool_call.function,
                arguments=dict(tool_call.args),
                status=(
                    V5ExecutionStatus.SUCCESS
                    if error is None
                    else V5ExecutionStatus.ERROR
                ),
                output=_jsonable_tool_output(value, formatted_output),
                error=error,
            )
            live_history.append(record.model_dump(mode="json"))
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
                    "tool_error": error,
                }
            )

        return query, runtime, env, [*messages, *results], extra_args
