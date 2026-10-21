from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from semantic_action_frame_guard.models import (
    ChainRefinementJudgment,
    DependencyLink,
    GuardInput,
    RefinementType,
    TaskFrameSet,
)


@dataclass(frozen=True)
class DependencyValidation:
    accepted_links: tuple[DependencyLink, ...]
    rejected_links: tuple[dict[str, Any], ...]
    valid_path: tuple[str, ...] | None

    def trace_fields(self) -> dict[str, Any]:
        return {
            "accepted_links": [link.model_dump(mode="json") for link in self.accepted_links],
            "rejected_links": list(self.rejected_links),
            "valid_path": list(self.valid_path) if self.valid_path else None,
        }


def _node(node_id: str) -> tuple[str, str] | None:
    kind, separator, value = node_id.partition(":")
    if separator and value and kind in {"task_frame", "observation", "tool_call"}:
        return kind, value
    return None


def validate_dependency_chain(
    judgment: ChainRefinementJudgment,
    frames: TaskFrameSet,
    guard_input: GuardInput,
) -> DependencyValidation:
    frame_ids = {frame.frame_id for frame in frames.root}
    observation_counts = Counter(item.observation_id for item in guard_input.observations)
    observations = {item.observation_id: item for item in guard_input.observations}
    observation_order = {
        item.observation_id: index for index, item in enumerate(guard_input.observations)
    }
    accepted: list[DependencyLink] = []
    rejected: list[dict[str, Any]] = []

    def reject(link: DependencyLink, reason: str) -> None:
        rejected.append({"link": link.model_dump(mode="json"), "reason": reason})

    for link in judgment.dependency_links:
        source = _node(link.from_id)
        target = _node(link.to_id)
        if source is None or target is None:
            reject(link, "node IDs must use task_frame, observation, or tool_call prefixes")
            continue
        source_kind, source_value = source
        target_kind, target_value = target

        if source_kind == "task_frame" and source_value not in frame_ids:
            reject(link, f"unknown task frame: {source_value}")
            continue
        if source_kind == "observation" and source_value not in observations:
            reject(link, f"unknown observation: {source_value}")
            continue
        if target_kind == "observation" and target_value not in observations:
            reject(link, f"unknown observation: {target_value}")
            continue
        if source_kind == "tool_call":
            reject(link, "a tool_call cannot be a dependency source")
            continue
        if target_kind == "task_frame":
            reject(link, "a task_frame cannot be a dependency target")
            continue
        if target_kind == "tool_call" and target_value != guard_input.tool_call_id:
            reject(link, f"tool_call:{target_value} is not the current tool call")
            continue
        if target_kind not in {"observation", "tool_call"}:
            reject(link, "dependency target must be an observation or the current tool call")
            continue

        involved_observations = {
            value
            for kind, value in (source, target)
            if kind == "observation"
        }
        if any(observation_counts[item] != 1 for item in involved_observations):
            reject(link, "an observation ID used by a link must be unique")
            continue
        if source_kind == "observation" and target_kind == "observation":
            if observation_order[source_value] >= observation_order[target_value]:
                reject(link, "observation links must move forward in trajectory order")
                continue

        if involved_observations:
            if not link.evidence_observation_id or not link.evidence_excerpt:
                reject(link, "a link involving an observation requires evidence ID and excerpt")
                continue
            if link.evidence_observation_id not in involved_observations:
                reject(link, "evidence observation must be an observation endpoint of the link")
                continue
            content = observations[link.evidence_observation_id].content
            if link.evidence_excerpt not in content:
                reject(link, "evidence excerpt is not an exact substring of the observation")
                continue
        elif link.evidence_observation_id is not None or link.evidence_excerpt is not None:
            reject(link, "a direct task-to-call link cannot cite an unrelated observation")
            continue

        if link.arg_bindings and target_kind != "tool_call":
            reject(link, "argument bindings are allowed only on a link to the current tool call")
            continue
        mismatched_argument = next(
            (
                name
                for name, value in link.arg_bindings.items()
                if name not in guard_input.tool.args or guard_input.tool.args[name] != value
            ),
            None,
        )
        if mismatched_argument is not None:
            reject(
                link,
                f"arg binding {mismatched_argument!r} does not equal the original tool argument",
            )
            continue
        accepted.append(link)

    path: tuple[str, ...] | None = None
    if (
        judgment.refinement_type != RefinementType.NOT_REFINEMENT
        and judgment.refined_frame_id is not None
    ):
        start = f"task_frame:{judgment.refined_frame_id}"
        goal = f"tool_call:{guard_input.tool_call_id}"
        adjacency: dict[str, list[str]] = {}
        for link in accepted:
            adjacency.setdefault(link.from_id, []).append(link.to_id)
        queue = deque([(start, (start,))])
        visited = {start}
        while queue:
            current, current_path = queue.popleft()
            if current == goal:
                path = current_path
                break
            for next_node in adjacency.get(current, []):
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, (*current_path, next_node)))

    return DependencyValidation(
        accepted_links=tuple(accepted),
        rejected_links=tuple(rejected),
        valid_path=path,
    )
