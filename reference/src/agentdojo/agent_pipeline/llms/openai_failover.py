from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai


def _is_retryable_api_error(error: Exception) -> bool:
    if isinstance(
        error,
        openai.APIConnectionError
        | openai.APITimeoutError
        | openai.InternalServerError
        | openai.RateLimitError,
    ):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    message = str(error).lower()
    return "model_not_found" in message or "no available channel" in message


class FailoverOpenAI:
    """OpenAI-compatible client with primary/fallback routing for chat completions."""

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        *,
        failure_threshold: int = 3,
        primary_probe_interval_seconds: float = 600.0,
        event_log: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._primary = primary
        self._fallback = fallback
        self._failure_threshold = failure_threshold
        self._primary_probe_interval_seconds = primary_probe_interval_seconds
        self._event_log = Path(event_log) if event_log else None
        self._clock = clock
        self._sleep = sleep
        self._active_backend = "primary"
        self._next_primary_probe = 0.0
        self._lock = threading.RLock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def _write_event(self, event: str, **details: Any) -> None:
        if self._event_log is None:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "active_backend": self._active_backend,
            **details,
        }
        self._event_log.parent.mkdir(parents=True, exist_ok=True)
        with self._event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _request(self, backend: str, **kwargs: Any) -> Any:
        client = self._primary if backend == "primary" else self._fallback
        return client.chat.completions.create(**kwargs)

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        with self._lock:
            now = self._clock()
            if self._active_backend == "fallback":
                if now >= self._next_primary_probe:
                    try:
                        completion = self._request("primary", **kwargs)
                    except Exception as error:
                        if not _is_retryable_api_error(error):
                            raise
                        self._next_primary_probe = now + self._primary_probe_interval_seconds
                        self._write_event(
                            "primary_probe_failed",
                            error_type=type(error).__name__,
                            status_code=getattr(error, "status_code", None),
                        )
                    else:
                        self._active_backend = "primary"
                        self._write_event("switched_to_primary")
                        return completion
                return self._request("fallback", **kwargs)

            for attempt in range(1, self._failure_threshold + 1):
                try:
                    return self._request("primary", **kwargs)
                except Exception as error:
                    if not _is_retryable_api_error(error):
                        raise
                    self._write_event(
                        "primary_request_failed",
                        attempt=attempt,
                        error_type=type(error).__name__,
                        status_code=getattr(error, "status_code", None),
                    )
                    if attempt < self._failure_threshold:
                        self._sleep(min(0.5 * attempt, 1.0))

            self._active_backend = "fallback"
            self._next_primary_probe = self._clock() + self._primary_probe_interval_seconds
            self._write_event("switched_to_fallback")
            return self._request("fallback", **kwargs)


def build_openai_client() -> openai.OpenAI | FailoverOpenAI:
    primary_api_key = os.getenv("OPENAI_PRIMARY_API_KEY") or os.getenv("OPENAI_API_KEY")
    primary_base_url = os.getenv("OPENAI_PRIMARY_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    fallback_api_key = os.getenv("OPENAI_FALLBACK_API_KEY")
    fallback_base_url = os.getenv("OPENAI_FALLBACK_BASE_URL")

    primary = openai.OpenAI(api_key=primary_api_key, base_url=primary_base_url, max_retries=0)
    if not fallback_api_key or not fallback_base_url:
        return primary

    fallback = openai.OpenAI(api_key=fallback_api_key, base_url=fallback_base_url, max_retries=0)
    failure_threshold = int(os.getenv("OPENAI_FAILOVER_FAILURE_THRESHOLD", "3"))
    probe_interval = float(os.getenv("OPENAI_PRIMARY_PROBE_INTERVAL_SECONDS", "600"))
    return FailoverOpenAI(
        primary,
        fallback,
        failure_threshold=failure_threshold,
        primary_probe_interval_seconds=probe_interval,
        event_log=os.getenv("OPENAI_FAILOVER_EVENT_LOG"),
    )
