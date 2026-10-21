from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from semantic_action_frame_guard.v4_models import (
    ActionSupportEntry,
    AuthorizationDecision,
    ExecutionRecord,
    ExecutionStatus,
    ParameterGroundingEntry,
    RuntimeAuthorization,
    StrictModel,
)


class ToolCallSpec(StrictModel):
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_description: str
    arguments: dict[str, Any]


class GuardInput(StrictModel):
    task_id: str = Field(min_length=1)
    trusted_user_task: str = Field(min_length=1)
    proposed_tool_call: ToolCallSpec
    execution_tool_history: list[ExecutionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_call_ids(self) -> GuardInput:
        call_ids = [item.call_id for item in self.execution_tool_history]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("execution history call IDs must be unique")
        if self.proposed_tool_call.tool_call_id in call_ids:
            raise ValueError("the proposed call cannot already be in execution history")
        return self


__all__ = [
    "ActionSupportEntry",
    "AuthorizationDecision",
    "ExecutionRecord",
    "ExecutionStatus",
    "GuardInput",
    "ParameterGroundingEntry",
    "RuntimeAuthorization",
    "ToolCallSpec",
]
