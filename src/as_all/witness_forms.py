"""Portable wiring of existing Qwen methods; frozen prompts and executor bodies."""
import copy
import hashlib
import json
import os

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, tool_result_to_str
from agentdojo.types import get_text_content_as_str, text_content_block_from_string
from semantic_action_frame_guard import TraceRecorder
from semantic_action_frame_guard.joint_witness_runtime_guard import JointWitnessGuardInput, JointWitnessRuntimeGuard
from semantic_action_frame_guard.llm import LLMResponse
from semantic_action_frame_guard.v3_pipeline import V3StructuredStage, normalize_strict_json_code_fence

from as_all.methods import ROOT, load_module

REFERENCE = ROOT / "reference/witness_forms"


def source_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def component(method):
    kind = "history" if method == "no_witness_guard" else "reasoning"
    return load_module("as_all._forms_" + kind, REFERENCE / (kind + "_guard.py"))


def guard_payload(method, frames, messages, call):
    return component(method).payload(frames, messages, call)


class GuardTransport:
    """Adapt the package's existing isolated, audited Qwen client to the old stage interface."""
    def __init__(self, client, model):
        self.client, self.model = client, model

    def complete(self, namespace, messages):
        result = self.client.chat.completions.create(model=self.model, messages=messages,
            temperature=0, max_tokens=2048, extra_body={"enable_thinking": False})
        data = result.model_dump(mode="json")
        usage = data.get("usage") or {}
        message = data["choices"][0]["message"]
        return LLMResponse(raw=message.get("content") or "", input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0), usage=usage or None,
            response=data, reasoning_content=message.get("reasoning_content"))


def build(method, case_dir, agent_client, defense_client, model, *, frozen_frames=None):
    if frozen_frames is None:
        frames_by_task = json.loads((ROOT / "data/manual_frames97.json").read_text())
        case = json.loads((case_dir / "case.json").read_text())
        frozen_frames = frames_by_task[f"{case['domain']}/{case['task']}"]
    frames = copy.deepcopy(frozen_frames)
    recorder = TraceRecorder(case_dir / "semantic")
    transport = GuardTransport(defense_client, model)
    if method in ("as_pg", "as_pg_guard"):
        return build_pg(method, frames, recorder, agent_client, transport, model)
    source = component(method)
    source.stopped = lambda: False
    prompt_name = "history" if method == "no_witness_guard" else "reasoning"
    stage = V3StructuredStage("guard_no_witness" if method == "no_witness_guard" else "guard_reasoning_context",
        (REFERENCE / (prompt_name + "_prompt.txt")).read_text(), source.Decision, transport, recorder,
        cache_namespace="witness_forms_original_v1", cache_enabled=False,
        response_normalizer=normalize_strict_json_code_fence)
    llm = source.FrameAgent(OpenAILLM(agent_client, model, temperature=0), frames, recorder)
    executor = source.GuardExecutor(frames, recorder, stage)
    pipeline = AgentPipeline([SystemMessage(load_system_message()), InitQuery(), llm,
                              ToolsExecutionLoop([executor, llm], max_iters=18)])
    pipeline.name = model
    return pipeline


def build_pg(method, frames, recorder, client, transport, model):
    os.environ["WITNESS_VARIANT"] = method
    # Load independent module globals so the old methods' adapters remain untouched.
    base = load_module("as_all._aspg_base", ROOT / "vendor/agentdojo/src/agentdojo/agent_pipeline/semantic_frame_alignment_agent_witness_no_guard.py")
    guarded = load_module("as_all._aspg_guard", ROOT / "vendor/agentdojo/src/agentdojo/agent_pipeline/semantic_frame_alignment_joint_witness_guard.py")
    root_ids = {frame["id"] for frame in frames}
    def frame_refs(_):
        return copy.deepcopy(frames), set(root_ids)
    base._frames_with_inline_refs = guarded._frames_with_inline_refs = frame_refs
    original_augment = base._augment_messages
    def augment(*args, **kwargs):
        messages, roots = original_augment(*args, **kwargs)
        for message in messages:
            if message["role"] == "system":
                text = get_text_content_as_str(message.get("content") or [])
                text = text.replace("trusted_task_frames[*].refs, such as t1 or t1_c1", "trusted_task_frames[*].id, such as t1 or t2")
                message["content"] = [text_content_block_from_string(text)]
        return messages, roots
    base._augment_messages = augment
    context = base.NoGuardWitnessContext(recorder, type("FrozenCache", (), {"get": lambda _, query: copy.deepcopy(frames)})())
    if method == "as_pg":
        executor = object.__new__(base.AgentWitnessNoGuardToolsExecutor)
    else:
        executor = object.__new__(guarded.JointWitnessRuntimeGuardToolsExecutor)
        executor.guard = JointWitnessRuntimeGuard(transport, recorder)
        def resolved(envelope, _):
            witness = envelope.authorization_witness
            roots = list(dict.fromkeys([witness.action_support.root,
                *(witness.parameter_grounding.roots if witness.parameter_grounding else [])]))
            selected = [frame for frame in frames if frame["id"] in roots]
            if set(roots) - root_ids:
                raise ValueError("unknown task root before Guard request")
            executor._last_guard_frame_selection = {"referenced_root_ids": roots,
                "selected_frame_ids": [frame["id"] for frame in selected],
                "excluded_frame_ids": [frame["id"] for frame in frames if frame not in selected]}
            evidence = []
            for oid in witness.action_support.authorization_evidence or []:
                record = context.output_registry[oid]
                evidence.append({"id": oid, "tool_name": record["tool_name"], "output": record["output"]})
            return JointWitnessGuardInput.model_validate({"trusted_task_frames": selected,
                "proposed_tool_call": envelope.proposed_tool_call.model_dump(mode="json"),
                "authorization_witness": witness.model_dump(mode="json"), "prior_tool_outputs": evidence})
        executor._resolved_guard_input = resolved
    executor.context = context
    executor.output_formatter = tool_result_to_str
    executor.redact_completed_witnesses = False
    executor.require_output_evidence = False
    executor.joint_parameter_grounding = True
    llm = base.NoGuardWitnessOpenAILLM(client, model, temperature=0, witness_context=context,
        explicit_cache=False, redact_completed_witnesses=False, joint_parameter_grounding=True)
    pipeline = AgentPipeline([SystemMessage(load_system_message()), InitQuery(), llm,
                              ToolsExecutionLoop([executor, llm], max_iters=18)])
    pipeline.name = model
    return pipeline
