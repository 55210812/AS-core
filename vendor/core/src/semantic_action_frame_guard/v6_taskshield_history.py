from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from semantic_action_frame_guard.llm import LLMTransport, TraceRecorder
from semantic_action_frame_guard.v3_models import TaskFrameSet
from semantic_action_frame_guard.v3_pipeline import V3StructuredStage
from semantic_action_frame_guard.v3_prompts import TASK_FRAME_SYSTEM, task_frame_request
from semantic_action_frame_guard.v5_models import AuthorizationDecision, RuntimeAuthorization
from semantic_action_frame_guard.v6_reference_selection import (
    ReferenceOption,
    ReferenceSelectionCatalog,
    ReferenceSelectionMode,
    V6ReferenceSelectionStage,
    _task_options,
)
from semantic_action_frame_guard.v6_taskshield_history_models import (
    ConversationHistoryMessage,
    GuardInput,
)
from semantic_action_frame_guard.v6_taskshield_history_prompts import (
    RUNTIME_AUTHORIZATION_SYSTEM,
    taskshield_history_request,
    taskshield_history_selection_system_prompt,
)


def build_taskshield_history_catalog(
    task_frames: TaskFrameSet,
    conversation_history: list[ConversationHistoryMessage],
    related_content: str,
) -> ReferenceSelectionCatalog:
    """Build a per-request catalog for Task Frame and Task Shield history IDs."""

    task_options = _task_options(task_frames)
    history_options: list[ReferenceOption] = []
    for index, message in enumerate(conversation_history):
        # Error tool results remain visible as Task Shield context but cannot be
        # selected as evidence for authorization.
        if message.role == "tool" and message.error:
            continue
        if not message.content.strip() and not message.tool_calls:
            continue
        history_options.append(
            ReferenceOption(f"h{len(history_options) + 1}", f"conversation_history[{index}]")
        )
    if related_content.strip():
        history_options.append(ReferenceOption(f"h{len(history_options) + 1}", "related_content"))
    return ReferenceSelectionCatalog(task_options, history_options, ReferenceSelectionMode.ALL)


def _history_payload_refs(catalog: ReferenceSelectionCatalog) -> set[str]:
    return {item.ref for item in catalog.history_reference_options}


def validate_taskshield_history_authorization(
    result: RuntimeAuthorization,
    guard_input: GuardInput,
    task_frames: TaskFrameSet,
    catalog: ReferenceSelectionCatalog,
) -> None:
    task_refs = {item.ref for item in catalog.task_reference_options}
    history_refs = _history_payload_refs(catalog)
    argument_names = set(guard_input.proposed_tool_call.arguments)
    unknown_arguments = set(result.parameter_grounding) - argument_names
    if unknown_arguments:
        raise ValueError(
            "parameter_grounding contains unknown arguments: " + ", ".join(sorted(unknown_arguments))
        )

    for support in result.action_support:
        if support.root not in task_refs:
            raise ValueError(f"invalid resolved task reference: {support.root}")
        for reference in support.via:
            if reference not in history_refs:
                raise ValueError(f"invalid resolved history reference: {reference}")
    for grounding in result.parameter_grounding.values():
        for reference in grounding.sources:
            if reference not in task_refs and reference not in history_refs:
                raise ValueError(f"invalid resolved source reference: {reference}")
    if result.decision == AuthorizationDecision.ALLOW:
        if not result.action_support:
            raise ValueError("allow requires non-empty action_support")
        if any(not item.sources for item in result.parameter_grounding.values()):
            raise ValueError("allow cannot contain an ungrounded parameter")


class V6TaskShieldHistorySelectionStage(V6ReferenceSelectionStage):
    """All-reference selection stage with an isolated cache namespace."""

    def __init__(self, system_prompt: str, transport: LLMTransport, recorder: TraceRecorder) -> None:
        super().__init__(
            system_prompt,
            transport,
            recorder,
            ReferenceSelectionMode.ALL,
            cache_namespace="v6_taskshield_history_reference_selection_all_1",
            cache_enabled=False,
        )


class V6TaskShieldHistoryPipeline:
    def __init__(
        self,
        transport: LLMTransport,
        recorder: TraceRecorder | None = None,
        *,
        task_frame_cache_path: str | Path | None = None,
    ) -> None:
        self.recorder = recorder or TraceRecorder()
        self.task_frame = V3StructuredStage(
            "task_frame",
            TASK_FRAME_SYSTEM,
            TaskFrameSet,
            transport,
            self.recorder,
            cache_namespace="v6_taskshield_history_task_frame_1",
        )
        self.runtime_authorization = V6TaskShieldHistorySelectionStage(
            taskshield_history_selection_system_prompt(),
            transport,
            self.recorder,
        )
        self.task_frame_cache: dict[str, Any] = {}
        self.require_task_frame_cache = task_frame_cache_path is not None
        if task_frame_cache_path:
            path = Path(task_frame_cache_path)
            if path.exists():
                self.task_frame_cache = json.loads(path.read_text(encoding="utf-8"))
        self.last_reference_catalog: dict[str, Any] = {}
        self.last_history_resolution: dict[str, Any] = {}

    @staticmethod
    def _task_frame_cache_key(trusted_user_task: str) -> str:
        return hashlib.sha256(trusted_user_task.encode("utf-8")).hexdigest()

    def _cached_task_frame(self, trusted_user_task: str, call_id: str) -> TaskFrameSet | None:
        digest = self._task_frame_cache_key(trusted_user_task)
        entry = self.task_frame_cache.get(digest)
        if entry is None and isinstance(self.task_frame_cache.get("frames_by_task_hash"), dict):
            entry = self.task_frame_cache["frames_by_task_hash"].get(digest)
        if entry is None:
            return None
        cache_source = "task_frame_cache"
        origin_tool_call_id = call_id
        if isinstance(entry, dict):
            stored_hash = entry.get("user_task_sha256")
            stored_task = entry.get("trusted_user_task")
            if stored_hash is not None and stored_hash != digest:
                raise ValueError("task frame cache hash does not match trusted task")
            if stored_task is not None and stored_task != trusted_user_task:
                raise ValueError("task frame cache trusted task does not match request")
            cache_source = entry.get("cache_source", cache_source)
            origin_tool_call_id = entry.get("origin_tool_call_id", origin_tool_call_id)
        if isinstance(entry, dict) and "task_frames" in entry:
            entry = entry["task_frames"]
        frames = TaskFrameSet.model_validate(entry)
        self.recorder.record(
            {
                "stage": "task_frame",
                "tool_call_id": call_id,
                "cache_hit": True,
                "cache_source": cache_source,
                "task_frame_cache_key": digest,
                "origin_tool_call_id": origin_tool_call_id,
                "structured_output": frames.model_dump(mode="json"),
            }
        )
        return frames

    def analyze(self, guard_input: GuardInput) -> tuple[TaskFrameSet, RuntimeAuthorization]:
        call_id = guard_input.proposed_tool_call.tool_call_id
        task_frames = self._cached_task_frame(guard_input.trusted_user_task, call_id)
        if task_frames is None:
            if self.require_task_frame_cache:
                raise ValueError("task frame cache has no frame for trusted task")
            task_frames = self.task_frame.run(
                task_frame_request(guard_input.trusted_user_task), call_id
            )
            digest = self._task_frame_cache_key(guard_input.trusted_user_task)
            self.task_frame_cache[digest] = {
                "user_task_sha256": digest,
                "trusted_user_task": guard_input.trusted_user_task,
                "task_frames": task_frames.model_dump(mode="json"),
                "cache_source": "generated_in_memory",
                "origin_tool_call_id": call_id,
            }
            self.recorder.record(
                {
                    "stage": "task_frame_cache_store",
                    "tool_call_id": call_id,
                    "task_frame_cache_key": digest,
                    "cache_source": "generated_in_memory",
                    "origin_tool_call_id": call_id,
                    "structured_output": task_frames.model_dump(mode="json"),
                }
            )

        catalog = build_taskshield_history_catalog(
            task_frames,
            guard_input.conversation_history,
            guard_input.related_content,
        )
        self.last_reference_catalog = catalog.model_dump(mode="json")
        self.last_history_resolution = {
            "history_message_count": len(guard_input.conversation_history),
            "selectable_history_ids": [item.id for item in catalog.history_reference_options],
            "content_source": [
                {"index": index, "source": "content" if message.content.strip() else "reasoning_content"}
                for index, message in enumerate(guard_input.conversation_history)
                if message.content.strip() or message.tool_calls
            ],
        }
        payload = taskshield_history_request(
            task_frames,
            guard_input.proposed_tool_call,
            guard_input,
            [item.model_dump() for item in catalog.task_reference_options],
            [item.model_dump() for item in catalog.history_reference_options],
        )
        self.recorder.record(
            {
                "stage": "v6_taskshield_history_catalog",
                "tool_call_id": call_id,
                "reference_catalog": self.last_reference_catalog,
                "history_resolution": self.last_history_resolution,
                "conversation_history": payload["conversation_history"],
                "related_content": payload["related_content"],
            }
        )
        authorization = self.runtime_authorization.run(
            payload,
            call_id,
            guard_input,
            task_frames,
            catalog,
            lambda result: validate_taskshield_history_authorization(
                result, guard_input, task_frames, catalog
            ),
        )
        return task_frames, authorization


__all__ = [
    "GuardInput",
    "ConversationHistoryMessage",
    "RUNTIME_AUTHORIZATION_SYSTEM",
    "V6TaskShieldHistoryPipeline",
    "V6TaskShieldHistorySelectionStage",
    "build_taskshield_history_catalog",
    "taskshield_history_request",
    "taskshield_history_selection_system_prompt",
    "validate_taskshield_history_authorization",
]
