"""A-version native domain tools; unchanged Witness/Guard execution boundary."""
import copy
import json
import os

from as_all.transport import NativeClient
from as_all.methods import MODEL, ROOT, load_module
from as_all import a_schema

PROXY = "submit_tool_call_with_authorization_witness"


class AClient(NativeClient):
    holistic = False

    def _messages(self, messages):
        for message in messages:
            if message.get("role") == "assistant" and self._history_key(message) not in self._native_history:
                raise ValueError("ConfigurationAuditError: native response history missing or modified")
        return super()._messages(messages)

    def _request(self, kwargs):
        body = super()._request(kwargs)
        context = a_schema.context(body)
        original = copy.deepcopy(body)
        original["tools"] = [{"functionDeclarations": [
            {"name": f["name"], "description": f["description"], "parametersJsonSchema": f["parameters"]}
            for f in context["available_domain_tools"]]}]
        return a_schema.transform(original, body, holistic=self.holistic)

    def _completion(self, data):
        result = super()._completion(data)
        payload = result.model_dump(mode="json")
        for call in payload["choices"][0]["message"].get("tool_calls") or []:
            function = call["function"]
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("AgentWitnessProtocolError: native arguments not an object")
            envelope = {"proposed_tool_call": {"tool": function["name"],
                        "arguments": {k: v for k, v in arguments.items() if k != "authorization_witness"}}}
            if "authorization_witness" in arguments:
                envelope["authorization_witness"] = arguments["authorization_witness"]
            call["function"] = {"name": PROXY, "arguments": json.dumps(envelope, ensure_ascii=False)}
        result = type(result).model_validate(payload)
        self._native_history[self._history_key(payload["choices"][0]["message"])] = copy.deepcopy(
            data["candidates"][0]["content"])
        return result


def build(method, case_dir, defense_client, client_args, *, model=MODEL, client_factory=None,
          variant_reference="native_a", frozen_frames=None):
    client_factory = client_factory or AClient
    from agentdojo.agent_pipeline import semantic_frame_alignment_agent_witness_no_guard as base
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
    from agentdojo.agent_pipeline.basic_elements import SystemMessage, InitQuery
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
    from agentdojo.agent_pipeline.llms import openai_llm
    from semantic_action_frame_guard import TraceRecorder

    base.chat_completion_request = openai_llm.chat_completion_request
    variants = load_module("ablation_variants", ROOT / "reference" / variant_reference / "ablation_variants.py")
    if frozen_frames is None:
        frames = json.loads((ROOT / "data/manual_frames97.json").read_text())
        case = json.loads((case_dir / "case.json").read_text())
        frozen_frames = frames[f"{case['domain']}/{case['task']}"]
    frozen = copy.deepcopy(frozen_frames)
    def frame_refs(_original):
        return copy.deepcopy(frozen), {f["id"] for f in frozen}
    def initialize(self, formatter):
        self.mode = "as_only"
        self.context = base.NoGuardWitnessContext(TraceRecorder(case_dir / "semantic"),
                                                type("FrozenCache", (), {"get": lambda _, q: frozen})())
        self.output_formatter = formatter
        self.redact_completed_witnesses = False
        self.require_output_evidence = False
        self.joint_parameter_grounding = True
    def guard_input(self, envelope, _original):
        support = envelope.authorization_witness.action_support
        selected = [f for f in frozen if f["id"] == support.root]
        if len(selected) != 1:
            raise ValueError("unknown task root before Guard request")
        self._last_guard_frame_selection = {"referenced_root_ids": [support.root],
            "selected_frame_ids": [support.root], "excluded_frame_ids": [f["id"] for f in frozen if f not in selected]}
        evidence = [{"id": oid, "tool_name": self.context.output_registry[oid]["tool_name"],
                     "output": self.context.output_registry[oid]["output"]}
                    for oid in support.authorization_evidence or []]
        return variants.GuardPayload(trusted_task_frames=selected, proposed_tool_call=envelope.proposed_tool_call,
                                     authorization_witness=envelope.authorization_witness, prior_tool_outputs=evidence)
    variants.ACTION_CONTRACT = variants.ACTION_CONTRACT.replace(
        "trusted_task_frames[*].refs, such as t1 or t1_c1", "trusted_task_frames[*].id, such as t1 or t2")
    variants.frame_refs = frame_refs
    variants.VariantMixin.initialize = initialize
    variants.VariantMixin._resolved_guard_input = guard_input
    base._augment_messages, base._proxy_function, base._frames_with_inline_refs = (
        variants.augment, variants.proxy_function, frame_refs)
    variants.guarded._frames_with_inline_refs = frame_refs
    os.environ["WITNESS_VARIANT"] = "as_only"
    if method == "as_only":
        executor = variants.VariantNoGuardExecutor()
    else:
        executor = variants.VariantGuardExecutor.__new__(variants.VariantGuardExecutor)
        initialize(executor, variants.guarded.tool_result_to_str)
        transport = variants.guarded.AuditedGuardTransport(model, client=defense_client,
                     enable_thinking=False, recorder=executor.context.recorder)
        executor.guard = variants.VariantGuard(transport, executor.context.recorder, False)
        if method == "as_only_guard_frame_reminder":
            def reminder(event):
                assert event["effective_decision"] == "block" and not event.get("invalid_guard_output")
                content = "\n\nReminder: trusted task frames (Please address incomplete tasks):\n" + "\n".join(
                    f"- [{f['id']}] {f['task']}" for f in frozen)
                event["block_task_reminder"] = {"content": content, "frame_ids": [f["id"] for f in frozen]}
                return content
            executor._block_task_reminder = reminder
    llm = variants.VariantLLM(client_factory(**client_args, stage="agent"), model, temperature=0,
        witness_context=executor.context, explicit_cache=False, redact_completed_witnesses=False,
        joint_parameter_grounding=True)
    pipeline = AgentPipeline([SystemMessage(load_system_message()), InitQuery(), llm,
                              ToolsExecutionLoop([executor, llm], max_iters=18)])
    pipeline.name = model
    return pipeline
