from __future__ import annotations

from semantic_action_frame_guard.llm import StageFailure
from semantic_action_frame_guard.v4_models import (
    AuthorizationDecision,
    GuardInput,
    RuntimeAuthorization,
)
from semantic_action_frame_guard.v4_pipeline import V4SemanticPipeline


class V4RuntimeAuthorizationGuard:
    def __init__(self, semantic_pipeline: V4SemanticPipeline) -> None:
        self.semantic_pipeline = semantic_pipeline

    def evaluate(self, guard_input: GuardInput) -> RuntimeAuthorization:
        try:
            _, authorization = self.semantic_pipeline.analyze(guard_input)
            return authorization
        except StageFailure as exc:
            self.semantic_pipeline.recorder.record(
                {
                    "stage": "runtime_action_authorization_failure",
                    "tool_call_id": guard_input.proposed_tool_call.tool_call_id,
                    "error": str(exc),
                    "fail_closed": True,
                }
            )
            return RuntimeAuthorization(
                action_support=[],
                parameter_grounding={},
                decision=AuthorizationDecision.BLOCK,
            )
