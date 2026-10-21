"""Supplier usage and request-duration accounting by explicit stage."""
import json


def cost(path):
    result = {
        stage: dict(
            requests=0,
            responses=0,
            raw_input=0,
            output=0,
            cached_input=0,
            noncached_input=0,
            missing_usage=0,
            unknown_cache=0,
            request_seconds=0.0,
        )
        for stage in ("agent", "defense")
    }
    if not path.exists():
        return result
    seen = set()
    for line in path.read_text().splitlines():
        event = json.loads(line)
        identity = (event["request_id"], event["event"])
        if identity in seen:
            raise ValueError("Duplicate request ledger event")
        seen.add(identity)
        bucket = result[event["stage"]]
        if event["event"] == "request_started":
            bucket["requests"] += 1
        elif event["event"] in ("request_completed", "request_error"):
            bucket["request_seconds"] += event.get("elapsed_seconds", 0)
            # Each attempt is a real HTTP request, including rejected429/failed connects.
            bucket["requests"] += max(0, len(event.get("transport_attempts", [])) - 1)
            attempts = event.get("transport_attempts", [])
            # The last successful attempt is represented by the top-level usage.
            # Earlier retries can have their own billed usage; never discard it.
            usages = [a.get("raw_usage") for a in attempts if a.get("retry")]
            usages.append(event.get("raw_usage"))
            for usage in usages:
                _add_usage(bucket, usage, event.get("provider"))
    return result


def _add_usage(bucket, usage, provider=None):
    if not usage:
        bucket["missing_usage"] += 1
        return
    is_qwen = provider == "qwen" or any(
        key in usage for key in ("prompt_tokens", "completion_tokens", "prompt_tokens_details")
    )
    if is_qwen:
        if not all(k in usage for k in ("prompt_tokens", "completion_tokens")):
            bucket["missing_usage"] += 1
            return
        raw = usage["prompt_tokens"]
        output = usage["completion_tokens"]
        if type(raw) is not int or type(output) is not int or raw < 0 or output < 0:
            raise ValueError("Invalid supplier token count")
        bucket["responses"] += 1
        bucket["raw_input"] += raw
        # OpenAI-compatible completion_tokens already includes reasoning tokens.
        bucket["output"] += output
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if cached is None:
            bucket["unknown_cache"] += 1
        else:
            if type(cached) is not int or not 0 <= cached <= raw:
                raise ValueError("Invalid supplier cache count")
            bucket["cached_input"] += cached
            bucket["noncached_input"] += raw - cached
        return
    if not all(k in usage for k in ("promptTokenCount", "candidatesTokenCount")):
        bucket["missing_usage"] += 1
        return
    bucket["responses"] += 1
    raw = usage["promptTokenCount"]
    bucket["raw_input"] += raw
    # Candidate + thought is counted once, not total + thought.
    thoughts = usage.get("thoughtsTokenCount")
    if thoughts is None:
        thoughts = (
            usage.get("totalTokenCount", usage["promptTokenCount"] + usage["candidatesTokenCount"])
            - raw
            - usage["candidatesTokenCount"]
        )
    if thoughts < 0:
        raise ValueError("Invalid supplier reasoning count")
    bucket["output"] += usage["candidatesTokenCount"] + thoughts
    cached = usage.get("cachedContentTokenCount")
    if cached is None:
        bucket["unknown_cache"] += 1
    else:
        if not 0 <= cached <= raw:
            raise ValueError("Invalid supplier cache count")
        bucket["cached_input"] += cached
        bucket["noncached_input"] += raw - cached
