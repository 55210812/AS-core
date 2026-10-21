from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError
from semantic_action_frame_guard import OpenAITransport
from semantic_action_frame_guard.joint_witness_runtime_guard import (
    JointWitnessGuardInput,
    JointWitnessRuntimeGuard,
)
from semantic_action_frame_guard.llm import StageFailure

from agentdojo.agent_pipeline.semantic_frame_alignment_agent_witness_no_guard import (
    PENDING_TASK_AGENT_USAGE_KEY,
    REPAIR_STATE_KEY,
    AgentWitnessNoGuardToolsExecutor,
    AgentWitnessProtocolError,
    JointProxyToolArguments,
    NoGuardWitnessOpenAILLM,
    _exclude_pending_task_agent_usage,
    _extract_proposed_calls,
    _frames_with_inline_refs,
)
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import (
    EmptyEnv,
    Env,
    FunctionCall,
    FunctionReturnType,
    FunctionsRuntime,
)
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string


class AuditedGuardTransport(OpenAITransport):
    def __init__(self, *args: Any, recorder: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    def complete(self, namespace: str, messages: list[dict[str, str]]):
        self.recorder.record(
            {
                "stage": "outgoing_request_audit",
                "request_stage": namespace,
                "model": self.model,
                "temperature": 0,
                "enable_thinking": self.enable_thinking,
            }
        )
        response = super().complete(namespace, messages)
        if os.getenv("SAFG_VERIFY_RESPONSE_MODEL") == "1":
            reported_model = (response.response or {}).get("model")
            if reported_model != self.model:
                raise RuntimeError(
                    f"ModelIdentityMismatch: requested {self.model}, received {reported_model}"
                )
            usage = response.usage or {}
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if self.enable_thinking is False and (reasoning or response.reasoning_content):
                raise RuntimeError("ThinkingConfigurationError: Guard returned reasoning with enable_thinking=false")
        return response


def _proposed_calls(validated: Sequence[tuple[Any, Any, Any]]) -> list[dict[str, Any]]:
    return [
        item[1].proposed_tool_call.model_dump(mode="json")
        for item in validated
    ]


class JointWitnessRuntimeGuardToolsExecutor(AgentWitnessNoGuardToolsExecutor):
    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        *,
        client: Any,
        model: str,
    ) -> None:
        super().__init__(
            tool_output_formatter,
            redact_completed_witnesses=False,
            joint_parameter_grounding=True,
        )
        transport = AuditedGuardTransport(
            model,
            api_key=os.getenv("SAFG_API_KEY") or os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("SAFG_BASE_URL") or os.getenv("QWEN_BASE_URL"),
            enable_thinking=False,
            client=client,
            recorder=self.context.recorder,
        )
        self.guard = JointWitnessRuntimeGuard(transport, self.context.recorder)

    def _resolved_guard_input(
        self,
        envelope: JointProxyToolArguments,
        task_frames: Any,
    ) -> JointWitnessGuardInput:
        witness = envelope.authorization_witness
        evidence_ids: list[str] = []
        for output_id in witness.action_support.authorization_evidence or []:
            if output_id not in evidence_ids:
                evidence_ids.append(output_id)
        prior_tool_outputs = []
        for output_id in evidence_ids:
            record = self.context.output_registry[output_id]
            output = record.get("output")
            if not isinstance(output, str):
                raise ValueError(f"referenced output is unavailable: {output_id}")
            prior_tool_outputs.append(
                {
                    "id": output_id,
                    "tool_name": record["tool_name"],
                    "output": output,
                }
            )
        frames_payload, _ = _frames_with_inline_refs(task_frames)
        grounding_roots = [] if witness.parameter_grounding is None else witness.parameter_grounding.roots
        referenced_root_ids = [witness.action_support.root, *grounding_roots]
        referenced_root_ids = list(dict.fromkeys(referenced_root_ids))
        root_to_frame: dict[str, str] = {}
        frame_roots: dict[str, set[str]] = {}
        for frame in frames_payload:
            frame_id = frame.get("frame_id")
            refs = frame.get("refs") or {}
            if isinstance(frame_id, str) and isinstance(refs, dict):
                frame_roots[frame_id] = set(refs)
                root_to_frame.update({root_id: frame_id for root_id in refs})
        missing_root_ids = [root_id for root_id in referenced_root_ids if root_id not in root_to_frame]
        if missing_root_ids:
            raise ValueError(
                "witness references task frame IDs that cannot be resolved: "
                + ", ".join(missing_root_ids)
            )
        selected_ids = {root_to_frame[root_id] for root_id in referenced_root_ids}
        selected_frames = [
            frame for frame in frames_payload
            if frame.get("frame_id") in selected_ids
        ]
        self._last_guard_frame_selection = {
            "referenced_root_ids": referenced_root_ids,
            "selected_frame_ids": [frame["frame_id"] for frame in selected_frames],
            "excluded_frame_ids": [
                frame["frame_id"] for frame in frames_payload
                if frame.get("frame_id") not in selected_ids
            ],
            "missing_root_ids": [],
        }
        return JointWitnessGuardInput.model_validate(
            {
                "trusted_task_frames": selected_frames,
                "proposed_tool_call": envelope.proposed_tool_call.model_dump(mode="json"),
                "authorization_witness": witness.model_dump(mode="json"),
                "prior_tool_outputs": prior_tool_outputs,
            }
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        calls = messages[-1].get("tool_calls") or []
        if not calls:
            return query, runtime, env, messages, extra_args

        task_frames = self.context.frame_cache.get(query)
        _, valid_root_ids = _frames_with_inline_refs(task_frames)
        try:
            validated = self._validate_batch(
                calls,
                valid_root_ids,
                runtime,
                self.context.available_output_ids(),
            )
        except (ValidationError, ValueError) as exc:
            _exclude_pending_task_agent_usage(
                extra_args,
                self.context.recorder,
                f"schema_format_repair_required: {exc}",
            )
            if REPAIR_STATE_KEY in extra_args:
                self.context.recorder.record(
                    {
                        "stage": "agent_witness_protocol_error",
                        "error": str(exc),
                        "repair_exhausted": True,
                    }
                )
                raise AgentWitnessProtocolError(f"format repair failed: {exc}") from exc
            expected_calls, verifiable = _extract_proposed_calls(calls)
            extra_args[REPAIR_STATE_KEY] = {
                "expected_calls": expected_calls,
                "verifiable": verifiable,
                "history_prefix_length": len(messages) - 1,
            }
            self.context.recorder.record(
                {
                    "stage": "agent_witness_format_repair",
                    "error": str(exc),
                    "expected_calls": expected_calls,
                    "verifiable": verifiable,
                }
            )
            feedback = (
                "FORMAT REPAIR REQUIRED. Return the same proposed tool calls in the same order "
                "with unchanged tool names and arguments. Repair only the JSON/witness structure. "
                f"Validation error: {exc}. Available prior output IDs: "
                + json.dumps(sorted(self.context.available_output_ids()))
            )
            results = [self._tool_error(call, feedback) for call in calls]
            return query, runtime, env, [*messages, *results], extra_args

        proposed = _proposed_calls(validated)
        format_repair = extra_args.pop(REPAIR_STATE_KEY, None)
        if format_repair is not None:
            if format_repair["verifiable"] and proposed != format_repair["expected_calls"]:
                _exclude_pending_task_agent_usage(
                    extra_args,
                    self.context.recorder,
                    "schema_format_repair_changed_proposed_tool_calls",
                )
                raise AgentWitnessProtocolError(
                    "format repair changed the proposed tool calls"
                )
            self.context.recorder.record(
                {
                    "stage": "agent_witness_format_repair_result",
                    "success": True,
                    "unverifiable_format_repair": not format_repair["verifiable"],
                    "proposed_tool_calls": proposed,
                }
            )
        extra_args.pop(PENDING_TASK_AGENT_USAGE_KEY, None)

        results: list[ChatToolResultMessage] = []
        for proxy_call, envelope, arguments in validated:
            actual_call = FunctionCall(
                function=envelope.proposed_tool_call.tool,
                args=arguments,
                id=proxy_call.id or str(uuid.uuid4()),
            )
            guard_input = self._resolved_guard_input(envelope, task_frames)
            event: dict[str, Any] = {
                "stage": "joint_witness_guard_decision",
                "proxy_tool_call_id": proxy_call.id,
                "guard_input": guard_input.model_dump(mode="json"),
                "guard_frame_selection": self._last_guard_frame_selection,
                "witness_variant": os.environ["WITNESS_VARIANT"],
                "executed": False,
            }
            try:
                decision = self.guard.authorize(guard_input, actual_call.id)
            except StageFailure as exc:
                event.update(
                    {
                        "effective_decision": "block",
                        "invalid_guard_output": True,
                        "error": str(exc),
                    }
                )
                results.append(
                    self._tool_error(
                        proxy_call,
                        f"block: invalid Guard output; call was not executed: {exc}",
                    )
                )
                self.context.recorder.record(event)
                continue

            event["decision"] = decision.model_dump(mode="json")
            event["effective_decision"] = decision.decision
            event["guard_reason"] = decision.reason
            if decision.decision == "allow":
                value, error = runtime.run_function(
                    env,
                    actual_call.function,
                    actual_call.args,
                )
                display_output = self.output_formatter(value)
                output_id = None
                if error is None:
                    output_id = self.context.register_output(
                        tool_call_id=actual_call.id,
                        tool_name=actual_call.function,
                        message_index=len(messages) + len(results),
                        output=display_output,
                    )
                content = display_output
                if output_id is not None:
                    content = (
                        f'<RUNTIME_TOOL_OUTPUT id="{output_id}">\n'
                        f"{display_output}\n"
                        "</RUNTIME_TOOL_OUTPUT>"
                    )
                results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(content)],
                        tool_call_id=proxy_call.id,
                        tool_call=proxy_call,
                        error=error,
                    )
                )
                event.update(
                    {
                        "executed": error is None,
                        "tool_output": display_output,
                        "tool_error": error,
                        "output_id": output_id,
                    }
                )
            else:
                feedback = (
                    "Blocked by runtime authorization Guard. The proposed call was not executed. "
                    f"Guard reason: {decision.reason}"
                )
                reminder_hook = getattr(self, "_block_task_reminder", None)
                if reminder_hook is not None:
                    feedback += reminder_hook(event)
                results.append(
                    self._tool_error(
                        proxy_call,
                        feedback,
                    )
                )
                event["effective_decision"] = "block"
                event["feedback_to_task_agent"] = feedback
            self.context.recorder.record(event)

        return query, runtime, env, [*messages, *results], extra_args


__all__ = [
    "JointWitnessRuntimeGuardToolsExecutor",
    "NoGuardWitnessOpenAILLM",
]
