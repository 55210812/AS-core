from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ExecutionRecord(StrictModel):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    status: ExecutionStatus
    output: Any = None
    error: str | None = None

    @model_validator(mode="after")
    def coherent_status(self) -> ExecutionRecord:
        if self.status == ExecutionStatus.SUCCESS and self.error is not None:
            raise ValueError("successful execution records cannot contain an error")
        if self.status == ExecutionStatus.ERROR and not self.error:
            raise ValueError("failed execution records require an error")
        return self


class ActionSupportEntry(StrictModel):
    root: str = Field(min_length=1)
    via: list[str]
    relation: str = Field(min_length=1)


class ParameterGroundingEntry(StrictModel):
    sources: list[str]
    derivation: str = Field(min_length=1)


class AuthorizationDecision(StrEnum):
    ALLOW = "allow"
    REPAIR = "repair"
    BLOCK = "block"


class RuntimeAuthorization(StrictModel):
    action_support: list[ActionSupportEntry]
    parameter_grounding: dict[str, ParameterGroundingEntry]
    decision: AuthorizationDecision


class ToolCallSpec(StrictModel):
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_description: str
    tool_json_schema: dict[str, Any]
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
