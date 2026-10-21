from __future__ import annotations

from semantic_action_frame_guard.llm import StageFailure
from semantic_action_frame_guard.v3_models import (
    AlignmentDecision,
    AlignmentErrorType,
    AlignmentResult,
    GuardInput,
)
from semantic_action_frame_guard.v3_pipeline import V3SemanticPipeline


class V3SemanticFrameAlignmentGuard:
    def __init__(self, semantic_pipeline: V3SemanticPipeline) -> None:
        self.semantic_pipeline = semantic_pipeline

    def evaluate(self, guard_input: GuardInput) -> AlignmentResult:
        try:
            task_frames, compact_trace, action_frame, judgment = (
                self.semantic_pipeline.analyze(guard_input)
            )
            result = AlignmentResult(
                decision=judgment.decision,
                error_type=judgment.error_type,
                reason=judgment.reason,
                task_frames=task_frames,
                compact_execution_trace=compact_trace,
                action_frame=action_frame,
            )
        except StageFailure as exc:
            result = AlignmentResult(
                decision=AlignmentDecision.BLOCK,
                error_type=AlignmentErrorType.ACTION_WRONG,
                reason=f"Fail-closed V3 structured semantic stage error: {exc}",
                task_frames=None,
                compact_execution_trace=None,
                action_frame=None,
            )
        self.semantic_pipeline.recorder.record(
            {
                "stage": "alignment_decision",
                "tool_call_id": guard_input.tool_call_id,
                "decision": result.model_dump(mode="json"),
                "executed": False,
            }
        )
        return result
