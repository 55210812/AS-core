"""Source integrity and library API checks; no model or detector downloads."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from as_all.methods import ALL_METHODS, METHODS, ROOT, WITNESS_FORM_METHODS, build


def test_frozen_files_and_adapted_modules_match_manifest():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    assert manifest["baseline"]["commit"] == "f30245d4c4e8b73c03a45b637a4242bd199b3d11"
    for section in ("unchanged_sha256", "adapted_sha256"):
        for relative, expected in manifest[section].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


def test_original_algorithm_symbols_are_preserved():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for relative, symbols in manifest["original_symbol_sha256"].items():
        text = (ROOT / relative).read_text()
        nodes = {node.name: node for node in ast.parse(text).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for name, expected in symbols.items():
            segment = ast.get_source_segment(text, nodes[name])
            assert hashlib.sha256(segment.encode()).hexdigest() == expected, (relative, name)


def test_independent_source_checkout_and_removed_orchestration():
    import agentdojo
    import as_all
    import semantic_action_frame_guard
    assert Path(as_all.__file__).resolve().is_relative_to(ROOT / "src")
    for module in (agentdojo, semantic_action_frame_guard):
        assert Path(module.__file__).resolve().is_relative_to(ROOT / "vendor")
    assert not any((ROOT / name).exists() for name in ("run.sh", "feedback.sh", "tools", ".github", "uv.lock"))
    removed = {"bootstrap", "cli", "experiments", "feedback", "report"}
    for path in (ROOT / "src/as_all").glob("*.py"):
        assert path.stem not in removed
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {"as_all." + name for name in removed}


def test_all_fifteen_methods_and_frozen_tasks():
    assert len(METHODS) == 11 and len(WITNESS_FORM_METHODS) == 4
    assert len(ALL_METHODS) == len(set(ALL_METHODS)) == 15
    frames = json.loads((ROOT / "data/manual_frames97.json").read_text())
    tasks = json.loads((ROOT / "data/task_texts97.json").read_text())
    assert len(frames) == len(tasks) == 97 and frames.keys() == tasks.keys()
    for filename in ("attacks949.json", "qwen_important949.json", "qwen_tool_knowledge949.json",
                     "gemini_tool_knowledge949.json"):
        assert len(json.loads((ROOT / "data" / filename).read_text())) == 949


@pytest.mark.parametrize("method", ALL_METHODS)
def test_build_one_case_without_configuration_or_model_request(method, tmp_path, monkeypatch):
    from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector, TransformersBasedPIDetector
    def detector(self, model_name, safe_label, threshold, mode):
        PromptInjectionDetector.__init__(self, mode)
        self.safe_label, self.threshold = safe_label, threshold
        self.pipeline = lambda text: [{"label": "SAFE", "score": 1.0}]
    monkeypatch.setattr(TransformersBasedPIDetector, "__init__", detector)
    monkeypatch.setattr("as_all.methods.detector_path", lambda: "mock-pinned-detector")
    pipeline = build(method, tmp_path, provider="qwen", key="test-only",
                     base_url="https://example.invalid/v1", task_id="slack/user_task_1")
    assert pipeline.name == "qwen3.5-flash" and pipeline.max_final_result_attempts == 1
    assert not (tmp_path / "case.json").exists()
    assert not (tmp_path / "requests.jsonl").exists()


@pytest.mark.parametrize("kwargs,match", [
    ({"method": "unknown"}, "Unknown method"),
    ({"provider": "unknown"}, "Unknown provider"),
    ({"method": "as_only", "task_id": "unknown"}, "Unknown frozen task_id"),
    ({"method": "as_pg", "provider": "gemini"}, "require the Qwen interface"),
    ({"key": ""}, "required"),
    ({"base_url": ""}, "required"),
])
def test_invalid_explicit_inputs_fail_before_connection(tmp_path, kwargs, match):
    args = dict(method="no_defense", provider="qwen", key="test-only", base_url="https://example.invalid/v1")
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build(case_dir=tmp_path, **args)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("provider,raw,output", [
    ("gemini", {"promptTokenCount": 100, "candidatesTokenCount": 12, "thoughtsTokenCount": 8,
                "cachedContentTokenCount": 40}, 20),
    ("qwen", {"prompt_tokens": 100, "completion_tokens": 20,
              "completion_tokens_details": {"reasoning_tokens": 8},
              "prompt_tokens_details": {"cached_tokens": 40}}, 20),
])
def test_stage_usage_and_cache_arithmetic(provider, raw, output, tmp_path):
    from as_all.usage import cost
    path = tmp_path / "requests.jsonl"
    events = []
    for stage in ("agent", "defense"):
        events.extend([
            dict(event="request_started", request_id=stage, provider=provider, stage=stage),
            dict(event="request_completed", request_id=stage, provider=provider, stage=stage,
                 raw_usage=raw, elapsed_seconds=1.5),
        ])
    path.write_text("\n".join(json.dumps(event) for event in events))
    for bucket in cost(path).values():
        assert bucket["requests"] == 1 and bucket["raw_input"] == 100
        assert bucket["output"] == output and bucket["cached_input"] == 40
        assert bucket["noncached_input"] == 60 and bucket["request_seconds"] == 1.5
