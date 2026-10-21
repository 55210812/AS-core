from __future__ import annotations

from semantic_action_frame_guard.llm import SemanticPipeline, StageFailure
from semantic_action_frame_guard.models import (
    AlignmentDecision,
    AlignmentResult,
    EvidenceLedger,
    EvidenceObservation,
    EvidenceLedgerUpdate,
    GuardInput,
    RefinementJudgment,
    RefinementReason,
    RefinementType,
)


class SemanticFrameAlignmentGuard:
    def __init__(self, semantic_pipeline: SemanticPipeline) -> None:
        self.semantic_pipeline = semantic_pipeline

    def evaluate(self, guard_input: GuardInput) -> AlignmentResult:
        try:
            frames, effect, refinement, dependency_validation = self.semantic_pipeline.analyze(guard_input)
            if refinement.refinement_type == RefinementType.NOT_REFINEMENT:
                decision = AlignmentDecision.BLOCK
                reason = refinement.reason.explanation
            elif self.semantic_pipeline.dependency_chain_enabled and (
                dependency_validation is None or dependency_validation.valid_path is None
            ):
                decision = AlignmentDecision.BLOCK
                reason = "Dependency chain rejected: no valid dependency path from the refined task frame."
            else:
                decision = AlignmentDecision.ALLOW
                reason = refinement.reason.explanation
            result = AlignmentResult(
                decision=decision,
                task_frames=frames,
                tool_effect=effect,
                refinement=refinement,
                reason=reason,
            )
        except StageFailure as exc:
            result = AlignmentResult(
                decision=AlignmentDecision.BLOCK,
                task_frames=None,
                tool_effect=None,
                refinement=RefinementJudgment(
                    answer="no",
                    refined_frame_id=None,
                    refinement_type=RefinementType.NOT_REFINEMENT,
                    reason=RefinementReason(
                        task_frame_predicate=None,
                        action_frame_predicate="unknown",
                        explanation=f"Fail-closed structured semantic stage error: {exc}",
                    ),
                ),
                reason=f"Fail-closed structured semantic stage error: {exc}",
            )
        self.semantic_pipeline.recorder.record(
            {
                "stage": "alignment_decision",
                "tool_call_id": guard_input.tool_call_id,
                "dependency_chain_enabled": self.semantic_pipeline.dependency_chain_enabled,
                "evidence_ledger_enabled": self.semantic_pipeline.evidence_ledger_enabled,
                "decision": result.model_dump(mode="json"),
                "executed": False,
            }
        )
        return result

    def update_evidence(
        self,
        guard_input: GuardInput,
        alignment_result: AlignmentResult,
        executed_output: str,
    ) -> EvidenceLedgerUpdate:
        ledger = guard_input.evidence_ledger or EvidenceLedger(
            task_id=guard_input.task_id
        )
        error = None
        new_observations: list[EvidenceObservation] = []
        if not self.semantic_pipeline.evidence_ledger_enabled:
            error = "evidence ledger is not enabled"
        elif ledger.task_id != guard_input.task_id:
            error = "evidence ledger task_id does not match the guard input"
        elif alignment_result.decision != AlignmentDecision.ALLOW:
            error = "evidence is updated only after an allowed tool call"
        else:
            observation_id = f"observation:{guard_input.tool_call_id}"
            existing_ids = {
                item.observation_id for item in ledger.observations
            }
            if observation_id in existing_ids:
                error = "evidence observation ID already exists"
            else:
                new_observations = [
                    EvidenceObservation(
                        observation_id=observation_id,
                        tool_call_id=guard_input.tool_call_id,
                        tool_name=guard_input.tool.name,
                        tool_args=guard_input.tool.args,
                        output=executed_output,
                    )
                ]

        updated_ledger = (
            ledger.model_copy(
                update={
                    "observations": [
                        *ledger.observations,
                        *new_observations,
                    ]
                }
            )
            if error is None
            else ledger
        )
        update = EvidenceLedgerUpdate(
            ledger=updated_ledger,
            new_observations=new_observations if error is None else [],
            applied=error is None,
            error=error,
        )
        self.semantic_pipeline.recorder.record(
            {
                "stage": "evidence_ledger_commit",
                "tool_call_id": guard_input.tool_call_id,
                "evidence_ledger_enabled": (
                    self.semantic_pipeline.evidence_ledger_enabled
                ),
                "ledger_before": ledger.model_dump(mode="json"),
                "accepted_observation_ids": [
                    item.observation_id
                    for item in update.new_observations
                ],
                "ledger_after": update.ledger.model_dump(mode="json"),
                "applied": update.applied,
                "error": update.error,
            }
        )
        return update
