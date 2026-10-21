from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticActionFrame(StrictModel):
    frame_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | list[str] | dict[str, Any]
    constraints: dict[str, Any]
    source: str
    output: str
    sink: str
    side_effect: str


class TaskFrameSet(RootModel[list[SemanticActionFrame]]):
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def stable_ids(self) -> TaskFrameSet:
        if not self.root:
            raise ValueError("task frame set must not be empty")
        expected = [f"task_{index:02d}" for index in range(1, len(self.root) + 1)]
        actual = [frame.frame_id for frame in self.root]
        if actual != expected:
            raise ValueError(f"task frame IDs must be stable and ordered: {expected}")
        return self


class CompletedCallSummaryDraft(StrictModel):
    action_summary: str = Field(min_length=1)
    result_summary: str = Field(min_length=1)
    outputs: dict[str, Any]


class CompletedCallSummarySubmission(StrictModel):
    summaries: list[CompletedCallSummaryDraft] = Field(min_length=1)


class CompletedCallSummary(StrictModel):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_args: dict[str, Any]
    action_summary: str = Field(min_length=1)
    result_summary: str = Field(min_length=1)
    outputs: dict[str, Any]
    raw_output: str = ""


class CompactExecutionTrace(StrictModel):
    compact_execution_trace: list[str]


class AlignmentDecision(StrEnum):
    ALLOW = "allow"
    REPAIR = "repair"
    BLOCK = "block"


class AlignmentErrorType(StrEnum):
    NONE = "none"
    ACTION_WRONG = "action_wrong"
    GROUNDING_WRONG = "grounding_wrong"
    DEPENDENCY_WRONG = "dependency_wrong"


class AlignmentJudgment(StrictModel):
    decision: AlignmentDecision
    error_type: AlignmentErrorType
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> AlignmentJudgment:
        if self.decision == AlignmentDecision.ALLOW and self.error_type != AlignmentErrorType.NONE:
            raise ValueError("allow requires error_type=none")
        if self.decision != AlignmentDecision.ALLOW and self.error_type == AlignmentErrorType.NONE:
            raise ValueError("repair and block require a non-none error_type")
        return self


class ToolCallSpec(StrictModel):
    name: str = Field(min_length=1)
    description: str
    json_schema: dict[str, Any]
    args: dict[str, Any]


class GuardInput(StrictModel):
    task_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    user_task: str = Field(min_length=1)
    tool: ToolCallSpec
    completed_call_summaries: list[CompletedCallSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_completed_calls(self) -> GuardInput:
        ids = [item.call_id for item in self.completed_call_summaries]
        if len(ids) != len(set(ids)):
            raise ValueError("completed call summaries must have unique call_id values")
        if self.tool_call_id in ids:
            raise ValueError("the proposed call cannot already be a completed call")
        return self


class RelevantPriorResult(StrictModel):
    call_id: str
    record_summary: str
    grounds: list[str]
    match_reason: str


class AlignmentResult(StrictModel):
    decision: AlignmentDecision
    error_type: AlignmentErrorType
    reason: str
    task_frames: TaskFrameSet | None
    compact_execution_trace: CompactExecutionTrace | None
    action_frame: SemanticActionFrame | None


def primitive_leaves(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key, item in value.items():
            leaves.extend(primitive_leaves(item, f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(primitive_leaves(item, f"{path}[{index}]"))
        return leaves
    return [(path, value)]


def relevant_prior_results(
    summaries: list[CompletedCallSummary],
    current_args: dict[str, Any],
) -> list[RelevantPriorResult]:
    argument_leaves = primitive_leaves(current_args)
    results: list[RelevantPriorResult] = []
    for summary in summaries:
        output_leaves = primitive_leaves(summary.outputs)
        grounds: list[str] = []
        for argument_path, argument_value in argument_leaves:
            if any(argument_value == output_value for _, output_value in output_leaves):
                grounds.append(f"{argument_path}={argument_value!r}")
        if grounds:
            results.append(
                RelevantPriorResult(
                    call_id=summary.call_id,
                    record_summary=f"{summary.action_summary} {summary.result_summary}",
                    grounds=grounds,
                    match_reason="exact primitive value equality between current arguments and validated prior outputs",
                )
            )
    return results
