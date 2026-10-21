from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from semantic_action_frame_guard.v3_models import TaskFrameSet
from semantic_action_frame_guard.v5_models import ExecutionRecord, ExecutionStatus
from semantic_action_frame_guard.v6_models import (
    ActionContextCandidate,
    ArgumentSourceCandidate,
    CandidateProvenance,
    SourceEvidence,
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "from",
    "by",
    "is",
    "be",
    "this",
    "that",
    "it",
    "as",
}


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _same_json_scalar(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    # bool is a subclass of int in Python but not in JSON.
    if type(left) is not type(right):
        return False
    return left == right


def _path_for_key(path: str, key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
        return f"{path}.{key}"
    return f'{path}[{json.dumps(key, ensure_ascii=False)}]'


def iter_primitive_nodes(value: Any, path: str) -> Iterable[tuple[str, Any]]:
    """Yield deterministic JSON paths and primitive leaves (including null)."""

    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_primitive_nodes(item, _path_for_key(path, str(key)))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_primitive_nodes(item, f"{path}[{index}]")
        return
    yield path, value


def _iter_output_evidence(
    value: Any, path: str
) -> Iterable[tuple[str, Any, str, str]]:
    """Yield keys, primitive values, and complete source lines.

    The paths are internal retrieval details and are never serialized into the
    Runtime Authorization payload.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            child_path = _path_for_key(path, str(key))
            yield f"{child_path}#key", str(key), "key", path
            yield from _iter_output_evidence(item, child_path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_output_evidence(item, f"{path}[{index}]")
        return
    yield path, value, "value", path.rsplit(".", 1)[0] if "." in path else path
    if isinstance(value, str):
        for index, row in enumerate(value.splitlines()):
            yield f"{path}#line[{index}]", row, "line", path


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _words(value: Any) -> set[str]:
    return {
        token.lower()
        for token in _WORD.findall(_render(value))
        if token.lower() not in _STOP_WORDS
    }


def _task_frame_matches(task_frames: TaskFrameSet, target: Any) -> list[SourceEvidence]:
    matches: list[SourceEvidence] = []
    if target is None:
        return matches
    for frame in task_frames.root:
        for path, value in iter_primitive_nodes(
            frame.model_dump(mode="json"), frame.frame_id
        ):
            if _same_json_scalar(value, target):
                matches.append(SourceEvidence(source_path=path, value=value))
    return matches


def _output_matches(output: Any, target: Any, root: str) -> list[SourceEvidence]:
    matches: list[SourceEvidence] = []
    if target is None:
        return matches
    matched_value_paths: set[str] = set()
    for path, value, kind, _ in _iter_output_evidence(output, root):
        if kind == "value" and _same_json_scalar(value, target):
            matches.append(SourceEvidence(source_path=path, value=value))
            matched_value_paths.add(path)
        elif (
            (kind == "key" and _same_json_scalar(value, target))
            or (
                kind == "line"
                and path.rsplit("#line[", 1)[0] not in matched_value_paths
                and _same_json_scalar(value, target)
            )
        ):
            matches.append(SourceEvidence(source_path=path, value=value))
    return matches


def _argument_leaves(value: Any, path: str) -> list[tuple[str, Any]]:
    return [(p, v) for p, v in iter_primitive_nodes(value, path) if v is not None]


def _record_text(record: ExecutionRecord, metadata: dict[str, Any]) -> str:
    description = str(metadata.get("tool_description", ""))
    plan = str(metadata.get("plan", ""))
    output = _render(record.output)
    return f"{description} {plan} {output}"


def _candidate_for_argument(
    argument_name: str,
    argument_value: Any,
    history: list[ExecutionRecord],
    history_metadata: dict[str, dict[str, Any]],
    task_frames: TaskFrameSet,
) -> tuple[ArgumentSourceCandidate | None, dict[str, Any]]:
    leaves = _argument_leaves(argument_value, argument_name)
    provenances: list[CandidateProvenance] = []
    diagnostics: dict[str, Any] = {
        "exact_task_frame_matches": 0,
        "exact_history_matches": 0,
        "text_candidates": 0,
        "source_call_ids": [],
    }

    # A candidate can be grounded directly in a trusted frame.  This is a
    # candidate only: the Guard still decides whether the frame field has the
    # same semantic role as the argument.
    task_evidence: list[SourceEvidence] = []
    for _, value in leaves:
        task_evidence.extend(_task_frame_matches(task_frames, value))
    if task_evidence:
        unique = {(item.source_path, json.dumps(item.value, ensure_ascii=False, default=str)): item for item in task_evidence}
        evidence = list(unique.values())
        for item in evidence:
            provenances.append(
                CandidateProvenance(
                    ref=item.source_path,
                    tool_description="trusted task frame",
                    plan="",
                )
            )
        diagnostics["exact_task_frame_matches"] = len(evidence)

    # Search successful history newest-first.  Keep all calls tied for the
    # strongest exact coverage so deterministic retrieval never guesses among
    # equal candidates.
    history_hits: list[tuple[int, ExecutionRecord, list[SourceEvidence]]] = []
    for position, record in enumerate(history):
        if record.status != ExecutionStatus.SUCCESS:
            continue
        evidence: list[SourceEvidence] = []
        for _, value in leaves:
            evidence.extend(_output_matches(record.output, value, f"{record.call_id}.output"))
        if evidence:
            deduped = {
                (item.source_path, json.dumps(item.value, ensure_ascii=False, default=str)): item
                for item in evidence
            }
            history_hits.append((position, record, list(deduped.values())))

    if history_hits:
        max_coverage = max(len(items) for _, _, items in history_hits)
        for _, record, evidence in history_hits:
            if len(evidence) != max_coverage:
                continue
            metadata = history_metadata.get(record.call_id, {})
            provenances.append(
                CandidateProvenance(
                    ref=f"{record.call_id}.output",
                    tool_description=str(metadata.get("tool_description", "")),
                    plan=str(metadata.get("plan", "")),
                )
            )
            diagnostics["source_call_ids"].append(record.call_id)
            diagnostics["exact_history_matches"] += len(evidence)

    # If exact data-flow matching found nothing, offer a deterministic textual
    # candidate.  This is deliberately only a hint for the Guard: no value is
    # synthesized and no authorization decision is made here.
    if not provenances:
        query_words = _words(argument_name) | _words(argument_value)
        scored: list[tuple[int, int, ExecutionRecord]] = []
        for position, record in enumerate(history):
            if record.status != ExecutionStatus.SUCCESS:
                continue
            metadata = history_metadata.get(record.call_id, {})
            overlap = len(query_words & _words(_record_text(record, metadata)))
            if overlap:
                scored.append((overlap, position, record))
        if scored:
            best = max(score for score, _, _ in scored)
            for score, _, record in sorted(scored, key=lambda item: item[1]):
                if score != best:
                    continue
                metadata = history_metadata.get(record.call_id, {})
                provenances.append(
                    CandidateProvenance(
                        ref=f"{record.call_id}.output",
                        tool_description=str(metadata.get("tool_description", "")),
                        plan=str(metadata.get("plan", "")),
                    )
                )
                diagnostics["text_candidates"] += 1
                diagnostics["source_call_ids"].append(record.call_id)

    if not provenances:
        return None, diagnostics
    # Preserve first-seen order and remove duplicate provenance entries.
    seen: set[tuple[str, str]] = set()
    unique_provenances: list[CandidateProvenance] = []
    for provenance in provenances:
        key = (provenance.ref, provenance.plan)
        if key in seen:
            continue
        seen.add(key)
        unique_provenances.append(provenance)
    return (
        ArgumentSourceCandidate(
            candidate_id=f"c_{argument_name}_1",
            value=argument_value,
            provenance=unique_provenances,
        ),
        diagnostics,
    )


def build_argument_source_candidates(
    arguments: dict[str, Any],
    history: list[ExecutionRecord],
    task_frames: TaskFrameSet,
    history_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, list[ArgumentSourceCandidate]], dict[str, Any]]:
    """Build candidate sources without an LLM or case/tool-specific rules."""

    metadata = history_metadata or {}
    candidates: dict[str, list[ArgumentSourceCandidate]] = {}
    diagnostics: dict[str, Any] = {}
    for name, value in arguments.items():
        candidate, detail = _candidate_for_argument(
            name, value, history, metadata, task_frames
        )
        candidates[name] = [candidate] if candidate is not None else []
        diagnostics[name] = detail
    return candidates, diagnostics


def _argument_exact_evidence(
    arguments: dict[str, Any], record: ExecutionRecord
) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    for value in arguments.values():
        for _, leaf in _argument_leaves(value, "argument"):
            evidence.extend(_output_matches(record.output, leaf, f"{record.call_id}.output"))
    unique: dict[tuple[str, str], SourceEvidence] = {}
    for item in evidence:
        unique[(item.source_path, json.dumps(item.value, ensure_ascii=False, default=str))] = item
    return list(unique.values())


def build_action_context_candidates(
    tool_name: str,
    tool_description: str,
    arguments: dict[str, Any],
    history: list[ExecutionRecord],
    task_frames: TaskFrameSet,
    history_metadata: dict[str, dict[str, Any]] | None = None,
    current_visible_plan: str = "",
    argument_candidates: dict[str, list[ArgumentSourceCandidate]] | None = None,
) -> tuple[list[ActionContextCandidate], dict[str, Any]]:
    """Retrieve untrusted action-context candidates without an LLM.

    Retrieval is intentionally generic: it scores exact argument coverage and
    token overlap over real prior descriptions, visible plans, task-frame
    text, and outputs.  The Guard remains responsible for all semantic
    authorization decisions.
    """

    metadata = history_metadata or {}
    argument_candidates = argument_candidates or {}
    query_words = (
        _words(tool_name)
        | _words(tool_description)
        | _words(arguments)
        | _words(task_frames.model_dump(mode="json"))
        | _words(current_visible_plan)
    )
    prioritized_calls = {
        provenance.ref.split(".", 1)[0]
        for values in argument_candidates.values()
        for candidate in values
        for provenance in candidate.provenance
        if provenance.ref
    }
    scored_records: list[tuple[int, int, ExecutionRecord]] = []
    diagnostics: dict[str, Any] = {
        "history_considered": 0,
        "selected_call_ids": [],
        "candidate_count": 0,
    }
    for position, record in enumerate(history):
        if record.status != ExecutionStatus.SUCCESS:
            continue
        diagnostics["history_considered"] += 1
        record_meta = metadata.get(record.call_id, {})
        description = str(record_meta.get("tool_description", ""))
        plan = str(record_meta.get("plan", ""))
        overlap = len(query_words & _words(" ".join((description, plan, _render(record.output)))))
        exact_count = len(_argument_exact_evidence(arguments, record))
        plan_overlap = len(query_words & _words(plan))
        score = exact_count * 100 + overlap + plan_overlap * 2
        if record.call_id in prioritized_calls:
            score += 50
        if score:
            scored_records.append((score, position, record))

    selected_records = sorted(scored_records, key=lambda item: (-item[0], item[1]))[:3]
    candidates: list[ActionContextCandidate] = []
    for _, _, record in sorted(selected_records, key=lambda item: item[1]):
        record_meta = metadata.get(record.call_id, {})
        candidates.append(
            ActionContextCandidate(
                candidate_id=f"action_{record.call_id}",
                ref=f"{record.call_id}.output",
                tool_description=str(record_meta.get("tool_description", "")),
                plan=str(record_meta.get("plan", "")),
            )
        )
        diagnostics["selected_call_ids"].append(record.call_id)
    diagnostics["candidate_count"] = len(candidates)
    return candidates, diagnostics


__all__ = [
    "build_action_context_candidates",
    "build_argument_source_candidates",
    "iter_primitive_nodes",
]
