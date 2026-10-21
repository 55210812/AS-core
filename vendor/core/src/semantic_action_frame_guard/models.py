from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticActionFrame(StrictModel):
    frame_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | list[str] | dict[str, Any]
    constraints: dict[str, Any] = Field(default_factory=dict)
    source: str
    output: str
    sink: str
    side_effect: str


class TaskFrameSet(RootModel[list[SemanticActionFrame]]):
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def unique_frame_ids(self) -> TaskFrameSet:
        if not self.root:
            raise ValueError("task frame set must not be empty")
        frame_ids = [frame.frame_id for frame in self.root]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("task frame ids must be unique")
        return self


class GroundingType(StrEnum):
    SOURCE_INSTRUCTION = "source_instruction"
    ARGUMENT_BINDING = "argument_binding"
    OBJECT_BINDING = "object_binding"
    BRANCH_RESOLUTION = "branch_resolution"


class GroundingStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


_GROUNDED_FIELD_KEYS = {
    "predicate",
    "object",
    "argument",
    "sink",
    "side_effect",
    "abstract_object",
}


class ObservationEvidence(StrictModel):
    observation_id: str = Field(min_length=1)
    grounding_type: GroundingType
    authorized_source: str | None = None
    dependency_source: str | None = None
    excerpt: str = Field(min_length=1)
    grounded_fields: dict[str, Any]
    grounding_status: GroundingStatus

    @model_validator(mode="after")
    def coherent_grounding(self) -> ObservationEvidence:
        if not self.authorized_source and not self.dependency_source:
            raise ValueError("observation evidence requires authorized_source or dependency_source")
        if not self.grounded_fields:
            raise ValueError("grounded_fields must not be empty")
        unknown = set(self.grounded_fields) - _GROUNDED_FIELD_KEYS
        if unknown:
            raise ValueError(f"unsupported grounded_fields keys: {sorted(unknown)}")
        return self


class EvidenceObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_args: dict[str, Any]
    output: str


class EvidenceLedger(StrictModel):
    task_id: str = Field(min_length=1)
    observations: list[EvidenceObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_observation_ids(self) -> EvidenceLedger:
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("evidence ledger observation ids must be unique")
        tool_call_ids = [item.tool_call_id for item in self.observations]
        if len(tool_call_ids) != len(set(tool_call_ids)):
            raise ValueError("evidence ledger tool call ids must be unique")
        return self


class EvidenceLedgerUpdate(StrictModel):
    ledger: EvidenceLedger
    new_observations: list[EvidenceObservation] = Field(default_factory=list)
    applied: bool
    error: str | None = None


class ToolSemanticEffect(SemanticActionFrame):
    @model_validator(mode="after")
    def valid_supporting_observation_ids_shape(self) -> ToolSemanticEffect:
        raw_ids = self.constraints.get("supporting_observation_ids")
        raw_evidence = self.constraints.get("observation_evidence")
        if raw_ids is not None and raw_evidence is not None:
            raise ValueError(
                "constraints may contain supporting_observation_ids or observation_evidence, not both"
            )
        if raw_ids is None:
            return self
        if not isinstance(raw_ids, list):
            raise ValueError("constraints.supporting_observation_ids must be a list")
        if any(
            not isinstance(observation_id, str) or not observation_id
            for observation_id in raw_ids
        ):
            raise ValueError(
                "constraints.supporting_observation_ids entries must be non-empty strings"
            )
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(
                "constraints.supporting_observation_ids must not contain duplicates"
            )
        return self

    @model_validator(mode="after")
    def valid_observation_evidence_shape(self) -> ToolSemanticEffect:
        raw_evidence = self.constraints.get("observation_evidence")
        if raw_evidence is None:
            return self
        if not isinstance(raw_evidence, list):
            raise ValueError("constraints.observation_evidence must be a list")
        for item in raw_evidence:
            ObservationEvidence.model_validate(item)
        return self


class RefinementType(StrEnum):
    DIRECT = "direct_refinement"
    IMPLEMENTATION_SUPPORT = "implementation_support_refinement"
    NOT_REFINEMENT = "not_refinement"


class RefinementReason(StrictModel):
    task_frame_predicate: str | None
    action_frame_predicate: str
    explanation: str


class DerivedBinding(StrictModel):
    abstract_object: str = Field(min_length=1)
    bound_value: Any
    supporting_observation_ids: list[str] = Field(min_length=1)
    derivation: str = Field(min_length=1)

    @model_validator(mode="after")
    def concrete_and_unique(self) -> DerivedBinding:
        if self.bound_value is None:
            raise ValueError("derived binding bound_value must be concrete")
        if len(self.supporting_observation_ids) != len(
            set(self.supporting_observation_ids)
        ):
            raise ValueError(
                "derived binding observation ids must be unique"
            )
        if any(not item for item in self.supporting_observation_ids):
            raise ValueError(
                "derived binding observation ids must be non-empty"
            )
        return self


class RefinementJudgment(StrictModel):
    answer: Literal["yes", "no"]
    refined_frame_id: str | None
    refinement_type: RefinementType
    reason: RefinementReason
    derived_bindings: list[DerivedBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self) -> RefinementJudgment:
        if self.refinement_type == RefinementType.NOT_REFINEMENT:
            if self.answer != "no" or self.refined_frame_id is not None:
                raise ValueError("not_refinement requires answer=no and refined_frame_id=null")
        elif self.answer != "yes" or self.refined_frame_id is None:
            raise ValueError("a refinement requires answer=yes and an existing refined_frame_id")
        return self


class DependencyLink(StrictModel):
    from_id: str = Field(min_length=1)
    to_id: str = Field(min_length=1)
    evidence_observation_id: str | None = None
    evidence_excerpt: str | None = None
    arg_bindings: dict[str, Any] = Field(default_factory=dict)


class ChainRefinementJudgment(RefinementJudgment):
    dependency_links: list[DependencyLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def links_match_outcome(self) -> ChainRefinementJudgment:
        is_not_refinement = self.refinement_type == RefinementType.NOT_REFINEMENT
        if is_not_refinement and self.dependency_links:
            raise ValueError("not_refinement requires an empty dependency_links list")
        if not is_not_refinement and not self.dependency_links:
            raise ValueError("a refinement requires at least one dependency link")
        return self


class AlignmentDecision(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class AvailableToolSpec(StrictModel):
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, Any]


class ToolCallSpec(StrictModel):
    name: str = Field(min_length=1)
    description: str
    json_schema: dict[str, Any]
    args: dict[str, Any]


class TrajectoryObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    tool_name: str = ""
    content: str


class GuardInput(StrictModel):
    task_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    user_task: str = Field(min_length=1)
    tool: ToolCallSpec
    available_tools: list[AvailableToolSpec] | None = None
    observations: list[TrajectoryObservation] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger | None = None

    @model_validator(mode="after")
    def complete_tool_catalog_is_consistent(self) -> GuardInput:
        if self.available_tools is None:
            return self
        names = [item.name for item in self.available_tools]
        if len(names) != len(set(names)):
            raise ValueError("available tool names must be unique")
        if self.tool.name not in names:
            raise ValueError("available tools must contain the current tool")
        return self


class AlignmentResult(StrictModel):
    decision: AlignmentDecision
    task_frames: TaskFrameSet | None
    tool_effect: ToolSemanticEffect | None
    refinement: RefinementJudgment
    reason: str
