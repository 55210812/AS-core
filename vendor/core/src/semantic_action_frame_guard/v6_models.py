from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from semantic_action_frame_guard.v5_models import (
    ActionSupportEntry,
    AuthorizationDecision,
    ExecutionRecord,
    ExecutionStatus,
    ParameterGroundingEntry,
    RuntimeAuthorization,
    StrictModel,
    ToolCallSpec,
)


class SourceEvidence(StrictModel):
    """A value copied verbatim by the deterministic trajectory retriever."""

    source_path: str = Field(min_length=1)
    value: Any


class CandidateProvenance(StrictModel):
    ref: str = Field(min_length=1)
    tool_description: str
    plan: str = ""


class ArgumentSourceCandidate(StrictModel):
    """One deterministic, untrusted candidate for a concrete argument."""

    candidate_id: str = Field(min_length=1)
    value: Any
    provenance: list[CandidateProvenance] = Field(default_factory=list)


class ActionContextCandidate(StrictModel):
    """A deterministic, untrusted candidate explaining why a prior result matters."""

    candidate_id: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    tool_description: str
    plan: str = ""


class GuardInput(StrictModel):
    # task_id and trusted_user_task are orchestration metadata; the serialized
    # authorization payload intentionally contains only the three V6 fields.
    task_id: str = Field(min_length=1)
    trusted_user_task: str = Field(min_length=1)
    proposed_tool_call: ToolCallSpec
    argument_source_candidates: dict[str, list[ArgumentSourceCandidate]] = Field(
        default_factory=dict
    )
    action_context_candidates: list[ActionContextCandidate] = Field(default_factory=list)
    execution_tool_history: list[ExecutionRecord] = Field(default_factory=list)
    history_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Adapter-only context for deterministic retrieval; never serialized into
    # the Runtime Authorization payload.
    current_visible_plan: str = ""

    @model_validator(mode="after")
    def unique_call_ids(self) -> GuardInput:
        call_ids = [item.call_id for item in self.execution_tool_history]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("execution history call IDs must be unique")
        if self.proposed_tool_call.tool_call_id in call_ids:
            raise ValueError("the proposed call cannot already be in execution history")
        unknown_arguments = set(self.argument_source_candidates) - set(
            self.proposed_tool_call.arguments
        )
        if unknown_arguments:
            raise ValueError(
                "argument_source_candidates contains unknown arguments: "
                + ", ".join(sorted(unknown_arguments))
            )
        return self


__all__ = [
    "ActionContextCandidate",
    "ActionSupportEntry",
    "ArgumentSourceCandidate",
    "AuthorizationDecision",
    "CandidateProvenance",
    "ExecutionRecord",
    "ExecutionStatus",
    "GuardInput",
    "ParameterGroundingEntry",
    "RuntimeAuthorization",
    "SourceEvidence",
    "ToolCallSpec",
]
