"""Offline differential tests against the frozen official detector.

Only framework imports/model construction are stubbed. Detector functions and
content conversion helpers are executed from their actual saved source.
Run: /opt/anaconda3/bin/python -m unittest discover -s outputs/baseline-official-aligned-20261014/tests -p test_detector_official_parity.py -v
"""
import ast
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "reference/ipiguard/pi_detector.py"
LOCAL = ROOT / "src/agentdojo/agent_pipeline/pi_detector.py"


class AbortAgentError(Exception):
    pass


def load_detector(path, name):
    # Execute actual helper bodies without importing unrelated SDK/runtime packages.
    helper_tree = ast.parse((ROOT / "src/agentdojo/types.py").read_text())
    wanted = {"get_text_content_as_str", "text_content_block_from_string"}
    helper_module = ast.Module(body=[n for n in helper_tree.body
                                    if isinstance(n, ast.FunctionDef) and n.name in wanted],
                               type_ignores=[])
    helpers = {"TextContentBlock": dict, "MessageContentBlock": dict}
    exec(compile(helper_module, "types.py", "exec"), helpers)
    dependencies = {
        "agentdojo.agent_pipeline.base_pipeline_element": {"BasePipelineElement": object},
        "agentdojo.agent_pipeline.errors": {"AbortAgentError": AbortAgentError},
        "agentdojo.functions_runtime": {"EmptyEnv": dict, "Env": dict, "FunctionsRuntime": object},
        "agentdojo.types": {"ChatMessage": dict, "MessageContentBlock": dict,
                            **{k: helpers[k] for k in wanted}},
    }
    modules = {}
    for module_name, attrs in dependencies.items():
        module = types.ModuleType(module_name)
        module.__dict__.update(attrs)
        modules[module_name] = module
    result = types.ModuleType(name)
    modules[name] = result
    with patch.dict(sys.modules, modules):
        exec(compile(path.read_text(), str(path), "exec"), result.__dict__)
    return result


def blocks(messages):
    converted = copy.deepcopy(messages)
    for message in converted:
        if message["content"] is not None:
            message["content"] = [{"type": "text", "content": message["content"]}]
    return converted


def flatten(messages):
    converted = copy.deepcopy(messages)
    for message in converted:
        if isinstance(message["content"], list):
            message["content"] = "\n".join(b["content"] for b in message["content"])
    return converted


class OfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = load_detector(OFFICIAL, "frozen_official_detector")
        cls.local = load_detector(LOCAL, "aligned_local_detector")

    def compare(self, messages, mode="message", raise_on_injection=False):
        outcomes = []
        for module, inputs in ((self.official, copy.deepcopy(messages)),
                               (self.local, blocks(messages))):
            detector = module.PromptInjectionDetector(mode, raise_on_injection)
            seen = []
            def detect(text):
                seen.append(text)
                return "ATTACK" in text, .1
            detector.detect = detect
            runtime, env, extra = object(), {}, {"sentinel": 1}
            try:
                result = detector.query("query", runtime, env, inputs, extra)
                self.assertIs(result[1], runtime)
                self.assertIs(result[2], env)
                self.assertIs(result[3], inputs)  # official mutates original sequence
                self.assertIs(result[4], extra)
                outcomes.append((seen, flatten(inputs), None))
            except AbortAgentError as exc:
                outcomes.append((seen, flatten(inputs), str(exc.args[0])))
        self.assertEqual(*outcomes)
        return outcomes[0]

    def test_error_only_is_not_detected_or_cleared(self):
        seen, result, _ = self.compare([{"role": "tool", "content": "normal", "error": "ATTACK"}])
        self.assertEqual(seen, ["normal"])
        self.assertEqual(result[0]["error"], "ATTACK")

    def test_content_blocked_error_still_preserved(self):
        _, result, _ = self.compare([{"role": "tool", "content": "ATTACK", "error": "original error"}])
        self.assertEqual(result[0]["content"], "<Data omitted because a prompt injection was detected>")
        self.assertEqual(result[0]["error"], "original error")

    def test_reverse_detection_original_message_order(self):
        seen, result, _ = self.compare([
            {"role": "tool", "content": "older ATTACK", "error": None},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "first", "error": None},
            {"role": "tool", "content": "second ATTACK", "error": None},
        ])
        self.assertEqual(seen, ["second ATTACK", "first"])
        self.assertEqual(result[0]["content"], "older ATTACK")
        self.assertEqual(result[2]["content"], "first")

    def test_full_conversation(self):
        seen, _, _ = self.compare([
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "first", "error": "ATTACK"},
            {"role": "tool", "content": "second ATTACK", "error": None},
        ], mode="full_conversation")
        self.assertEqual(seen, ["user: request\ntool: first\ntool: second ATTACK",
                                "user: request\ntool: first"])

    def test_abort_and_early_returns(self):
        self.compare([{"role": "tool", "content": "ATTACK", "error": None}], raise_on_injection=True)
        self.compare([])
        self.compare([{"role": "user", "content": "ATTACK"}])
        self.compare([{"role": "tool", "content": "", "error": None}])

    def test_classifier_defaults_threshold_mapping_and_no_truncation(self):
        import inspect
        long_text = "token " * 10000
        for module in (self.official, self.local):
            defaults = inspect.signature(module.TransformersBasedPIDetector).parameters
            self.assertEqual(defaults["safe_label"].default, "SAFE")
            self.assertEqual(defaults["threshold"].default, .5)
            self.assertEqual(defaults["mode"].default, "message")
            detector = object.__new__(module.TransformersBasedPIDetector)
            detector.safe_label, detector.threshold = "SAFE", .5
            for label, score, expected in [("SAFE", .9, False), ("SAFE", .1, True),
                                            ("INJECTION", .9, True), ("INJECTION", .1, False),
                                            ("SAFE", .5, False), ("INJECTION", .5, False)]:
                calls = []
                def pipeline(*args, **kwargs):
                    calls.append((args, kwargs))
                    return [{"label": label, "score": score}]
                detector.pipeline = pipeline
                decision, safety = detector.detect(long_text)
                self.assertEqual(decision, expected)
                self.assertAlmostEqual(safety, score if label == "SAFE" else 1 - score)
                self.assertEqual(calls, [((long_text,), {})])

    def test_pipeline_configuration_remains_official(self):
        tree = ast.parse((ROOT / "src/agentdojo/agent_pipeline/agent_pipeline.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "TransformersBasedPIDetector"]
        self.assertEqual(len(calls), 1)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
        self.assertEqual(kwargs, {"model_name": "protectai/deberta-v3-base-prompt-injection-v2",
                                  "safe_label": "SAFE", "threshold": .5, "mode": "message"})


if __name__ == "__main__":
    unittest.main()
