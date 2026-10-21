from __future__ import annotations

from semantic_action_frame_guard.llm import LLMTransport, TraceRecorder
from semantic_action_frame_guard.v6_models import (
    AuthorizationDecision,
    GuardInput,
    RuntimeAuthorization,
)
from semantic_action_frame_guard.v6_pipeline import V6SemanticPipeline


class V6RuntimeAuthorizationGuard:
    """Fail closed at the current tool call boundary, without blacklisting tools."""

    def __init__(self, pipeline: V6SemanticPipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def from_transport(
        cls, transport: LLMTransport, recorder: TraceRecorder | None = None, **kwargs
    ) -> V6RuntimeAuthorizationGuard:
        return cls(V6SemanticPipeline(transport, recorder, **kwargs))

    def evaluate(self, guard_input: GuardInput) -> RuntimeAuthorization:
        try:
            _, result = self.pipeline.analyze(guard_input)
            return result
        except Exception:
            # Keep the public decision model valid while preventing a failed
            # authorization request from executing the proposed tool call.
            return RuntimeAuthorization(
                action_support=[],
                parameter_grounding={},
                decision=AuthorizationDecision.BLOCK,
            )


__all__ = ["V6RuntimeAuthorizationGuard"]
