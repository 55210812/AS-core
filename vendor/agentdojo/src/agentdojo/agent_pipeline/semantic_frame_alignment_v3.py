import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from semantic_action_frame_guard import (
    CompletedCallSummary,
    CompletedCallSummaryDraft,
    CompletedCallSummarySubmission,
    OpenAITransport,
    SUMMARY_TOOL_NAME,
    TASK_AGENT_TRACE_CONTRACT,
    TraceRecorder,
    V3AlignmentDecision,
    V3GuardInput,
    V3SemanticFrameAlignmentGuard,
    V3SemanticPipeline,
    V3ToolCallSpec,
)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import (
    EmptyEnv,
    Env,
    Function,
    FunctionCall,
    FunctionReturnType,
    FunctionsRuntime,
)
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

def submit_completed_call_summaries(
    summaries: list[CompletedCallSummaryDraft],
) -> str:
    """Submit semantic summaries for pending successful domain-tool results.

    This internal tool records trace metadata only. It does not access or modify
    the external environment and does not authorize later actions.

    :param summaries: One semantic summary for each pending successful result, in result order.
    """
    return ""


def _summary_tool_definition() -> Function:
    description = (
        "Submit semantic summaries for pending successful domain-tool results.\n"
        "This internal tool records trace metadata only. It does not access or modify "
        "the external environment and does not authorize later actions."
    )
    return Function(
        name=SUMMARY_TOOL_NAME,
        description=description,
        parameters=CompletedCallSummarySubmission,
        dependencies={},
        run=submit_completed_call_summaries,
        full_docstring=submit_completed_call_summaries.__doc__ or description,
        return_type=str,
    )


class SemanticFrameSummaryToolRegistrar(BasePipelineElement):
    """Expose the internal structured summary tool to the Task Agent."""

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if SUMMARY_TOOL_NAME not in runtime.functions:
            runtime.register_function(_summary_tool_definition())
        return query, runtime, env, messages, extra_args


class SemanticFrameTraceCollector(BasePipelineElement):
    """Bind structured Task-Agent summaries to exact successful call identities."""

    def __init__(self, recorder: TraceRecorder) -> None:
        self.recorder = recorder

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
        assistant = messages[-1]
        raw_content = get_text_content_as_str(assistant["content"] or [])
        calls = list(assistant["tool_calls"] or [])
        summary_calls = [call for call in calls if call.function == SUMMARY_TOOL_NAME]
        domain_calls = [call for call in calls if call.function != SUMMARY_TOOL_NAME]
        pending = list(extra_args.get("semantic_frame_unreported_call_ids", []))
        event: dict[str, Any] = {
            "stage": "task_agent_completed_call_summaries",
            "tool_call_id": pending[-1] if pending else "assistant-message",
            "raw_assistant_content": raw_content,
            "expected_call_ids": pending,
            "accepted": False,
            "validation_errors": [],
        }
        accepted: list[CompletedCallSummary] = []
        if pending and not summary_calls:
            event["validation_errors"].append(
                "missing submit_completed_call_summaries call for pending successful results"
            )
        elif not pending and summary_calls:
            event["validation_errors"].append(
                "submit_completed_call_summaries was called with no pending successful results"
            )
        elif len(summary_calls) > 1:
            event["validation_errors"].append(
                "submit_completed_call_summaries must be called exactly once per pending result batch"
            )
        elif summary_calls:
            try:
                summary_call = summary_calls[0]
                if calls.index(summary_call) > 0:
                    raise ValueError(
                        "submit_completed_call_summaries must precede domain-tool calls in the assistant response"
                    )
                submission = CompletedCallSummarySubmission.model_validate(summary_call.args)
                if len(submission.summaries) > len(pending):
                    raise ValueError(
                        "summary count exceeds pending successful calls"
                    )
                executed = extra_args.get("semantic_frame_executed_calls", {})
                already_reported = {
                    item["call_id"]
                    for item in extra_args.get("semantic_frame_completed_call_summaries", [])
                }
                accepted_ids = pending[: len(submission.summaries)]
                for call_id, draft in zip(
                    accepted_ids, submission.summaries, strict=True
                ):
                    if call_id in already_reported:
                        raise ValueError(f"duplicate completed call summary: {call_id}")
                    record = executed.get(call_id)
                    if record is None:
                        raise ValueError(f"unknown or unexecuted call_id: {call_id}")
                    accepted.append(
                        CompletedCallSummary(
                            call_id=call_id,
                            tool_name=record["tool_name"],
                            tool_args=record["tool_args"],
                            action_summary=draft.action_summary,
                            result_summary=draft.result_summary,
                            outputs=draft.outputs,
                            raw_output=record["raw_output"],
                        )
                    )
                event["accepted"] = True
            except Exception as exc:
                event["validation_errors"].append(f"{type(exc).__name__}: {exc}")

        if accepted:
            stored = extra_args.setdefault("semantic_frame_completed_call_summaries", [])
            stored.extend(item.model_dump(mode="json") for item in accepted)
            extra_args["semantic_frame_unreported_call_ids"] = pending[len(accepted) :]
        elif pending:
            extra_args["semantic_frame_unreported_call_ids"] = pending
        summary_results = extra_args.setdefault("semantic_frame_summary_call_results", {})
        for summary_call in summary_calls:
            summary_results[summary_call.id] = {
                "accepted": bool(accepted),
                "error": (
                    None
                    if accepted
                    else "; ".join(event["validation_errors"])
                    or "invalid summary submission"
                ),
                "accepted_call_ids": [item.call_id for item in accepted],
            }
        if pending and not summary_calls and not domain_calls:
            retry_call = FunctionCall(
                function=SUMMARY_TOOL_NAME,
                args={"summaries": []},
                id=f"summary-required-{uuid.uuid4().hex}",
            )
            summary_results[retry_call.id] = {
                "accepted": False,
                "error": "; ".join(event["validation_errors"]),
                "accepted_call_ids": [],
            }
            assistant_with_retry = ChatAssistantMessage(
                role="assistant",
                content=assistant["content"],
                tool_calls=[retry_call],
                reasoning_content=assistant.get("reasoning_content"),
            )
            messages = [*messages[:-1], assistant_with_retry]
            event["runtime_retry_tool_call_id"] = retry_call.id
        event["raw_summary_tool_calls"] = [
            {"id": call.id, "function": call.function, "args": dict(call.args)}
            for call in summary_calls
        ]
        event["proposed_domain_tool_calls"] = [
            {"id": call.id, "function": call.function, "args": dict(call.args)}
            for call in domain_calls
        ]
        event["structured_output"] = [item.model_dump(mode="json") for item in accepted]
        event["sanitized_assistant_content"] = raw_content
        self.recorder.record(event)
        extra_args.setdefault("semantic_frame_task_agent_messages", []).append(event)
        return query, runtime, env, messages, extra_args


class SemanticFrameAlignmentToolsExecutor(BasePipelineElement):
    """V3 executor: allow original calls and return repair/block feedback without execution."""

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
    ) -> None:
        transport = OpenAITransport(
            os.getenv("SAFG_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
            api_key=os.getenv("SAFG_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("SAFG_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
            reasoning_effort=os.getenv("SAFG_REASONING_EFFORT"),
        )
        trace_root = Path(
            os.getenv(
                "SAFG_TRACE_DIR",
                "semantic-action-frame-results/traces/agentdojo/v3_general_semantic_frames",
            )
        )
        self.recorder = TraceRecorder(trace_root)
        self.guard = V3SemanticFrameAlignmentGuard(
            V3SemanticPipeline(transport, self.recorder)
        )
        self.output_formatter = tool_output_formatter

    @staticmethod
    def _task_id(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:20]

    def _reset_for_task(self, query: str, extra_args: dict) -> str:
        task_id = self._task_id(query)
        if extra_args.get("semantic_frame_task_id") != task_id:
            for key in (
                "semantic_frame_completed_call_summaries",
                "semantic_frame_executed_calls",
                "semantic_frame_unreported_call_ids",
                "semantic_frame_denied_calls",
                "semantic_frame_task_agent_messages",
                "semantic_frame_summary_call_results",
                "semantic_frame_deferred_calls",
            ):
                extra_args.pop(key, None)
        extra_args["semantic_frame_task_id"] = task_id
        return task_id

    def _guard_input(self, query, tool_call, runtime, extra_args) -> V3GuardInput:
        function = runtime.functions[tool_call.function]
        task_id = self._reset_for_task(query, extra_args)
        return V3GuardInput(
            task_id=task_id,
            tool_call_id=tool_call.id or str(uuid.uuid4()),
            user_task=query,
            tool=V3ToolCallSpec(
                name=function.name,
                description=function.description,
                json_schema=function.parameters.model_json_schema(),
                args=dict(tool_call.args),
            ),
            completed_call_summaries=[
                CompletedCallSummary.model_validate(item)
                for item in extra_args.get("semantic_frame_completed_call_summaries", [])
            ],
        )

    @staticmethod
    def _feedback(decision) -> str:
        if decision.decision == V3AlignmentDecision.REPAIR:
            return (
                "Repair required by semantic_frame_alignment: "
                f"{decision.reason} Do not retry the unchanged call; obtain the missing dependency "
                "or correct the grounded arguments before proposing another call."
            )
        return (
            "Blocked by semantic_frame_alignment: "
            f"{decision.reason} Do not retry the denied call."
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

        results: list[ChatToolResultMessage] = []
        blocking_pending = list(
            extra_args.get("semantic_frame_unreported_call_ids", [])
        )
        for tool_call in calls:
            if tool_call.function == SUMMARY_TOOL_NAME:
                result = extra_args.get("semantic_frame_summary_call_results", {}).get(
                    tool_call.id,
                    {
                        "accepted": False,
                        "error": "summary submission was not validated by the trace collector",
                        "accepted_call_ids": [],
                    },
                )
                accepted = bool(result["accepted"])
                deferred = list(extra_args.get("semantic_frame_deferred_calls", []))
                feedback = (
                    "Accepted structured summaries for: "
                    + ", ".join(result["accepted_call_ids"])
                    if accepted
                    else "Invalid structured summary: " + str(result["error"])
                )
                if accepted and deferred and not blocking_pending:
                    deferred_text = "; ".join(
                        f"{item['tool_name']}({json.dumps(item['tool_args'], sort_keys=True)})"
                        for item in deferred
                    )
                    feedback += (
                        ". The following earlier domain calls were not executed: "
                        f"{deferred_text}. Re-propose any call that is still required."
                    )
                    extra_args["semantic_frame_deferred_calls"] = []
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(feedback if accepted else "")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=None if accepted else feedback,
                    )
                )
                self.recorder.record(
                    {
                        "stage": "task_agent_summary_submission",
                        "tool_call_id": tool_call.id,
                        "accepted": accepted,
                        "accepted_call_ids": result["accepted_call_ids"],
                        "deferred_calls_reported": deferred if accepted else [],
                        "error": result["error"],
                        "executed": False,
                    }
                )
                continue
            if blocking_pending:
                deferred = extra_args.setdefault("semantic_frame_deferred_calls", [])
                deferred_record = {
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.function,
                    "tool_args": dict(tool_call.args),
                }
                if deferred_record not in deferred:
                    deferred.append(deferred_record)
                feedback = (
                    "A successful prior domain-tool result still requires a structured summary. "
                    "Call submit_completed_call_summaries before proposing another domain action. "
                    "Do not repeat the already executed domain tool."
                )
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
                        "stage": "task_agent_summary_required",
                        "tool_call_id": tool_call.id,
                        "pending_call_ids": blocking_pending,
                        "proposed_tool_name": tool_call.function,
                        "proposed_tool_args": dict(tool_call.args),
                        "executed": False,
                    }
                )
                continue
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
            original_args = dict(tool_call.args)
            guard_input = self._guard_input(query, tool_call, runtime, extra_args)
            trajectory_fingerprint = [
                item.call_id for item in guard_input.completed_call_summaries
            ]
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "tool_name": tool_call.function,
                        "tool_args": original_args,
                        "completed_call_ids": trajectory_fingerprint,
                    },
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            denied = extra_args.setdefault("semantic_frame_denied_calls", {})
            if signature in denied:
                origin = denied[signature]
                reason = "Repeated unchanged call under the same accepted execution trace."
                extra_args.setdefault("defense_events", []).append(
                    {
                        "defense_name": "semantic_frame_alignment",
                        "event_type": "semantic_frame_alignment_repeat_denial",
                        "action": "block",
                        "tool_name": tool_call.function,
                        "reason": reason,
                        "origin_denied_tool_call_id": origin,
                        "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
                    }
                )
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Blocked by semantic_frame_alignment: {reason}",
                    )
                )
                continue

            decision = self.guard.evaluate(guard_input)
            extra_args.setdefault("defense_events", []).append(
                {
                    "defense_name": "semantic_frame_alignment",
                    "event_type": "semantic_frame_alignment_decision",
                    "action": decision.decision.value,
                    "tool_name": tool_call.function,
                    "reason": decision.reason,
                    "alignment_result": decision.model_dump(mode="json"),
                    "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
                }
            )
            if decision.decision != V3AlignmentDecision.ALLOW:
                denied[signature] = guard_input.tool_call_id
                feedback = self._feedback(decision)
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
                        "tool_call_id": guard_input.tool_call_id,
                        "decision": decision.model_dump(mode="json"),
                        "feedback": feedback,
                        "executed": False,
                    }
                )
                continue

            value, error = runtime.run_function(env, tool_call.function, original_args)
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
            if error is None:
                executed = extra_args.setdefault("semantic_frame_executed_calls", {})
                executed[guard_input.tool_call_id] = {
                    "call_id": guard_input.tool_call_id,
                    "tool_name": tool_call.function,
                    "tool_args": original_args,
                    "raw_output": formatted_output,
                }
                extra_args.setdefault("semantic_frame_unreported_call_ids", []).append(
                    guard_input.tool_call_id
                )
            self.recorder.record(
                {
                    "stage": "agentdojo_execution",
                    "tool_call_id": guard_input.tool_call_id,
                    "decision": decision.model_dump(mode="json"),
                    "executed": error is None,
                    "tool_name": tool_call.function,
                    "tool_args": original_args,
                    "raw_tool_output": formatted_output,
                    "tool_error": error,
                }
            )
        return query, runtime, env, [*messages, *results], extra_args
