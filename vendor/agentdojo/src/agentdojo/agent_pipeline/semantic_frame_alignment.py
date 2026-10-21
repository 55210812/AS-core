from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from semantic_action_frame_guard import (
    AlignmentDecision,
    AvailableToolSpec,
    EvidenceLedger,
    GuardInput,
    OpenAITransport,
    SemanticFrameAlignmentGuard,
    SemanticPipeline,
    ToolCallSpec,
    TraceRecorder,
    TrajectoryObservation,
)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


class SemanticFrameAlignmentToolsExecutor(BasePipelineElement):
    """Execute only tool calls aligned with the trusted task's semantic frames."""

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
    ) -> None:
        transport = OpenAITransport(
            os.getenv("SAFG_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
            api_key=os.getenv("SAFG_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("SAFG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            reasoning_effort=os.getenv("SAFG_REASONING_EFFORT"),
        )
        trace_root = Path(
            os.getenv("SAFG_TRACE_DIR", "semantic-action-frame-results/traces/agentdojo/semantic_frame_alignment")
        )
        self.evidence_ledger_enabled = os.getenv("SAFG_EVIDENCE_LEDGER") == "1"
        self.recorder = TraceRecorder(trace_root)
        self.guard = SemanticFrameAlignmentGuard(SemanticPipeline(transport, self.recorder))
        self.output_formatter = tool_output_formatter

    @staticmethod
    def _observations(messages: Sequence[ChatMessage]) -> list[TrajectoryObservation]:
        observations = []
        for index, message in enumerate(messages):
            if message["role"] != "tool":
                continue
            observations.append(
                TrajectoryObservation(
                    observation_id=message.get("tool_call_id") or f"observation-{index}",
                    tool_name=message["tool_call"].function,
                    content=get_text_content_as_str(message["content"]),
                )
            )
        return observations

    @staticmethod
    def _available_tools(runtime: FunctionsRuntime) -> list[AvailableToolSpec]:
        return [
            AvailableToolSpec(
                name=function.name,
                description=function.description,
                parameters=function.parameters.model_json_schema(),
            )
            for function in sorted(
                runtime.functions.values(), key=lambda item: item.name
            )
        ]

    def _guard_input(self, query, tool_call, runtime, messages, extra_args) -> GuardInput:
        function = runtime.functions[tool_call.function]
        task_id = hashlib.sha256(query.encode()).hexdigest()[:20]
        previous_task_id = extra_args.get("semantic_frame_task_id")
        if previous_task_id is not None and previous_task_id != task_id:
            extra_args.pop("semantic_frame_evidence_ledger", None)
            extra_args.pop("semantic_frame_denied_calls", None)
        extra_args["semantic_frame_task_id"] = task_id
        evidence_ledger = None
        if getattr(self, "evidence_ledger_enabled", False):
            raw_ledger = extra_args.get("semantic_frame_evidence_ledger")
            evidence_ledger = (
                EvidenceLedger.model_validate(raw_ledger)
                if raw_ledger is not None
                else EvidenceLedger(task_id=task_id)
            )
        call_id = tool_call.id or str(uuid.uuid4())
        return GuardInput(
            task_id=task_id,
            tool_call_id=call_id,
            user_task=query,
            tool=ToolCallSpec(
                name=function.name,
                description=function.description,
                json_schema=function.parameters.model_json_schema(),
                args=dict(tool_call.args),
            ),
            available_tools=self._available_tools(runtime),
            observations=self._observations(messages),
            evidence_ledger=evidence_ledger,
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

        results = []
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

            original_args = dict(tool_call.args)
            guard_input = self._guard_input(query, tool_call, runtime, messages, extra_args)
            signature_payload = json.dumps(
                {
                    "tool_name": tool_call.function,
                    "tool_args": original_args,
                    "evidence_ledger": (
                        guard_input.evidence_ledger.model_dump(mode="json")
                        if guard_input.evidence_ledger is not None
                        else None
                    ),
                },
                sort_keys=True,
                default=str,
            )
            signature = hashlib.sha256(signature_payload.encode()).hexdigest()
            denied_calls = extra_args.setdefault("semantic_frame_denied_calls", {})
            if signature in denied_calls:
                origin_call_id = denied_calls[signature]
                reason = "Repeated submission of an identical call that was already denied."
                extra_args.setdefault("defense_events", []).append(
                    {
                        "defense_name": "semantic_frame_alignment",
                        "event_type": "semantic_frame_alignment_repeat_denial",
                        "action": "block",
                        "tool_name": tool_call.function,
                        "reason": reason,
                        "origin_denied_tool_call_id": origin_call_id,
                        "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
                    }
                )
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Denied by semantic_frame_alignment. Do not retry the denied call.",
                    )
                )
                self.recorder.record(
                    {
                        "stage": "agentdojo_execution",
                        "tool_call_id": tool_call.id,
                        "decision": {
                            "decision": "BLOCK",
                            "reason": reason,
                            "origin_denied_tool_call_id": origin_call_id,
                        },
                        "executed": False,
                    }
                )
                continue
            decision = self.guard.evaluate(guard_input)
            extra_args.setdefault("defense_events", []).append(
                {
                    "defense_name": "semantic_frame_alignment",
                    "event_type": "semantic_frame_alignment_decision",
                    "action": decision.decision.value.lower(),
                    "tool_name": tool_call.function,
                    "reason": decision.reason,
                    "alignment_result": decision.model_dump(mode="json"),
                    "task_agent_token_usage": dict(extra_args.get("token_usage", {})),
                }
            )
            if decision.decision != AlignmentDecision.ALLOW:
                denied_calls[signature] = guard_input.tool_call_id
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Denied by semantic_frame_alignment. Do not retry the denied call.",
                    )
                )
                self.recorder.record(
                    {
                        "stage": "agentdojo_execution",
                        "tool_call_id": guard_input.tool_call_id,
                        "decision": decision.model_dump(mode="json"),
                        "executed": False,
                    }
                )
                continue

            value, error = runtime.run_function(env, tool_call.function, original_args)
            formatted_output = self.output_formatter(value)
            ledger_update = None
            if error is None and getattr(self, "evidence_ledger_enabled", False):
                ledger_update = self.guard.update_evidence(
                    guard_input,
                    decision,
                    formatted_output,
                )
                extra_args["semantic_frame_evidence_ledger"] = ledger_update.ledger.model_dump(
                    mode="json"
                )
                extra_args.setdefault("defense_events", []).append(
                    {
                        "defense_name": "semantic_frame_alignment",
                        "event_type": "semantic_frame_evidence_ledger_update",
                        "action": "record",
                        "tool_name": tool_call.function,
                        "applied": ledger_update.applied,
                        "new_observation_ids": [
                            observation.observation_id
                            for observation in ledger_update.new_observations
                        ],
                        "reason": ledger_update.error,
                    }
                )
            results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(formatted_output)],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=error,
                )
            )
            self.recorder.record(
                {
                    "stage": "agentdojo_execution",
                    "tool_call_id": guard_input.tool_call_id,
                    "decision": decision.model_dump(mode="json"),
                    "executed": error is None,
                    "evidence_ledger_update": (
                        ledger_update.model_dump(mode="json")
                        if ledger_update is not None
                        else None
                    ),
                }
            )
        return query, runtime, env, [*messages, *results], extra_args
