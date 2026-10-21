from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import requests

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.recent_defenses import defense_events
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str, text_content_block_from_string


DEFAULT_GUARD_URL = "http://127.0.0.1:18100"
DEFAULT_REFUSAL_MESSAGE = "I cannot continue because a safety guard classified this step as unsafe."


def _event(
    extra_args: dict,
    defense_name: str,
    event_type: str,
    action: str,
    reason: str,
    stage: str,
    detector: str | None = None,
    tool_name: str | None = None,
    score: float | None = None,
    threshold: float | None = None,
    label: str | None = None,
    category: str | None = None,
) -> None:
    item: dict[str, Any] = {
        "defense_name": defense_name,
        "event_type": event_type,
        "action": action,
        "reason": reason,
        "stage": stage,
    }
    if detector is not None:
        item["detector"] = detector
    if tool_name is not None:
        item["tool_name"] = tool_name
    if score is not None:
        item["score"] = score
    if threshold is not None:
        item["threshold"] = threshold
    if label is not None:
        item["label"] = label
    if category is not None:
        item["category"] = category
    defense_events(extra_args).append(item)


def _text_from_message(message: ChatMessage) -> str:
    content = message.get("content")
    if content is None:
        return ""
    return get_text_content_as_str(content)


def _blocked_tool_content(defense_name: str) -> list[dict[str, str]]:
    return [text_content_block_from_string(f"<Tool output blocked by {defense_name}: unsafe content detected>")]


def _refusal_content(defense_name: str) -> list[dict[str, str]]:
    return [text_content_block_from_string(f"{DEFAULT_REFUSAL_MESSAGE} Defense: {defense_name}.")]


class PaperSafetyGuardClient:
    """HTTP client for the paper safety sidecar."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or os.getenv("PAPER_SAFETY_GUARD_URL") or DEFAULT_GUARD_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else float(os.getenv("PAPER_SAFETY_GUARD_TIMEOUT", "60"))

    def classify(
        self,
        prompt: str,
        response: str | None,
        detectors: Sequence[str],
        stage: str | None = None,
    ) -> dict[str, Any]:
        payload = {"items": [{"prompt": prompt, "response": response, "stage": stage}], "detectors": list(detectors)}
        try:
            response_obj = requests.post(f"{self.base_url}/classify", json=payload, timeout=self.timeout)
            response_obj.raise_for_status()
            data = response_obj.json()
            return data["results"][0]
        except Exception as exc:
            return {
                "source": "sidecar_unavailable",
                "is_parsing_error": True,
                "error": repr(exc),
                "prompt_harmfulness": None,
                "response_harmfulness": None,
                "response_refusal": None,
            }


class PaperSafetyElement(BasePipelineElement):
    """Common runtime guard used to expose paper-derived defenses in AgentDojo."""

    defense_name = "paper_safety"
    detectors: tuple[str, ...] = ()
    block_user_prompt = True
    block_tool_output = True
    replace_harmful_response = True

    def __init__(self, client: PaperSafetyGuardClient | None = None) -> None:
        self.client = client or PaperSafetyGuardClient()

    def _result_requests_block(self, result: dict[str, Any], stage: str) -> tuple[bool, str]:
        if "wildguard" in self.detectors:
            if result.get("prompt_harmfulness") == "harmful" and stage in {"user", "tool"}:
                return True, "wildguard_prompt_harmfulness"
            if result.get("response_harmfulness") == "harmful" and stage == "assistant":
                return True, "wildguard_response_harmfulness"
        sdgo_result = result.get("sdgo_self_discrimination")
        if isinstance(sdgo_result, dict) and not sdgo_result.get("skipped"):
            label = sdgo_result.get("label")
            score = sdgo_result.get("score")
            threshold = sdgo_result.get("threshold", 0.65)
            over_threshold = isinstance(score, (int, float)) and float(score) >= float(threshold)
            if sdgo_result.get("should_block") or (label in {"unsafe", "harmful"} and over_threshold):
                return True, f"sdgo_self_discrimination_{label or 'unsafe'}"
        for detector in ("latent_harmfulness", "refusal_direction"):
            detector_result = result.get(detector)
            if not isinstance(detector_result, dict) or detector_result.get("skipped"):
                continue
            label = detector_result.get("label")
            if label in {"harmful", "unsafe", "refusal_signal_high"}:
                return True, f"{detector}_{label}"
        return False, "no_unsafe_signal"

    def _record_result_events(
        self,
        extra_args: dict,
        result: dict[str, Any],
        stage: str,
        action: str,
        reason: str,
        tool_name: str | None = None,
    ) -> None:
        if result.get("source") == "sidecar_unavailable":
            _event(
                extra_args,
                self.defense_name,
                "paper_guard_unavailable",
                "skipped",
                result.get("error", "sidecar_unavailable"),
                stage,
                tool_name=tool_name,
            )
            return

        if "wildguard" in self.detectors:
            label = result.get("prompt_harmfulness") or result.get("response_harmfulness")
            _event(
                extra_args,
                self.defense_name,
                "wildguard_classification",
                action,
                reason,
                stage,
                detector="wildguard",
                tool_name=tool_name,
                label=label,
            )

        if "sdgo_self_discrimination" in self.detectors:
            sdgo_result = result.get("sdgo_self_discrimination", {})
            if sdgo_result.get("skipped"):
                _event(
                    extra_args,
                    self.defense_name,
                    "paper_guard_skipped",
                    "skipped",
                    sdgo_result.get("reason", "sdgo_judge_unavailable"),
                    stage,
                    detector="sdgo_self_discrimination",
                    tool_name=tool_name,
                )
            else:
                _event(
                    extra_args,
                    self.defense_name,
                    "sdgo_self_discrimination",
                    action,
                    reason,
                    stage,
                    detector="sdgo_self_discrimination",
                    tool_name=tool_name,
                    score=sdgo_result.get("score"),
                    threshold=sdgo_result.get("threshold"),
                    label=sdgo_result.get("label"),
                    category=sdgo_result.get("category"),
                )

        for detector in ("refusal_direction", "latent_harmfulness"):
            if detector not in self.detectors:
                continue
            detector_result = result.get(detector, {})
            if detector_result.get("skipped"):
                _event(
                    extra_args,
                    self.defense_name,
                    "paper_guard_skipped",
                    "skipped",
                    detector_result.get("reason", "artifact_or_model_unavailable"),
                    stage,
                    detector=detector,
                    tool_name=tool_name,
                )
                continue
            _event(
                extra_args,
                self.defense_name,
                f"{detector}_score",
                action,
                reason,
                stage,
                detector=detector,
                tool_name=tool_name,
                score=detector_result.get("score"),
                threshold=detector_result.get("threshold"),
                label=detector_result.get("label"),
            )

    def _classify(self, prompt: str, response: str | None = None, stage: str | None = None) -> dict[str, Any]:
        return self.client.classify(prompt=prompt, response=response, detectors=self.detectors, stage=stage)

    def _handle_user_message(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: Sequence[ChatMessage],
        extra_args: dict,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        text = _text_from_message(messages[-1])
        result = self._classify(text, stage="user")
        should_block, reason = self._result_requests_block(result, "user")
        self._record_result_events(extra_args, result, "user", "blocked" if should_block else "allowed", reason)
        if should_block and self.block_user_prompt:
            raise AbortAgentError(f"{DEFAULT_REFUSAL_MESSAGE} Defense: {self.defense_name}.", list(messages), env)
        return query, runtime, env, messages, extra_args

    def _handle_tool_messages(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: Sequence[ChatMessage],
        extra_args: dict,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        updated_messages = list(messages)
        for index in range(len(updated_messages) - 1, -1, -1):
            message = updated_messages[index]
            if message["role"] != "tool":
                break
            text = _text_from_message(message)
            tool_name = message["tool_call"].function if "tool_call" in message else None
            result = self._classify(text, stage="tool")
            should_block, reason = self._result_requests_block(result, "tool")
            action = "blocked" if should_block and self.block_tool_output else "allowed"
            self._record_result_events(extra_args, result, "tool", action, reason, tool_name)
            if should_block and self.block_tool_output:
                updated_messages[index] = {**message, "content": _blocked_tool_content(self.defense_name)}
        return query, runtime, env, updated_messages, extra_args

    def _handle_assistant_message(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env,
        messages: Sequence[ChatMessage],
        extra_args: dict,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        last_message = messages[-1]
        if last_message.get("tool_calls"):
            return query, runtime, env, messages, extra_args
        response_text = _text_from_message(last_message)
        if not response_text:
            return query, runtime, env, messages, extra_args
        result = self._classify(query, response_text, stage="assistant")
        should_block, reason = self._result_requests_block(result, "assistant")
        action = "replaced" if should_block and self.replace_harmful_response else "allowed"
        self._record_result_events(extra_args, result, "assistant", action, reason)
        if should_block and self.replace_harmful_response:
            messages = [*messages[:-1], {**last_message, "content": _refusal_content(self.defense_name)}]
        return query, runtime, env, messages, extra_args

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        role = messages[-1]["role"]
        if role == "user":
            return self._handle_user_message(query, runtime, env, messages, extra_args)
        if role == "tool":
            return self._handle_tool_messages(query, runtime, env, messages, extra_args)
        if role == "assistant":
            return self._handle_assistant_message(query, runtime, env, messages, extra_args)
        return query, runtime, env, messages, extra_args


class WildGuardModerator(PaperSafetyElement):
    defense_name = "wildguard"
    detectors = ("wildguard",)


class RefusalDirectionGuard(PaperSafetyElement):
    defense_name = "refusal_direction_guard"
    detectors = ("refusal_direction",)


class LatentHarmfulnessGuard(PaperSafetyElement):
    defense_name = "latent_harmfulness_guard"
    detectors = ("latent_harmfulness",)


class DeRTaRuntimeGuard(PaperSafetyElement):
    defense_name = "derta_runtime_guard"
    detectors = ("wildguard", "latent_harmfulness", "refusal_direction")


class SDGOSelfDiscriminationGuard(PaperSafetyElement):
    defense_name = "sdgo_self_discrimination_guard"
    detectors = ("sdgo_self_discrimination",)


class PaperSafetyGuard(PaperSafetyElement):
    defense_name = "paper_safety_guard"
    detectors = ("wildguard", "latent_harmfulness", "refusal_direction", "sdgo_self_discrimination")
