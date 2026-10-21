"""Existing method components; only portable imports and callback bindings added."""
import copy
import hashlib
import json
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.types import text_content_block_from_string, get_text_content_as_str
from semantic_action_frame_guard.llm import StageFailure
from as_all.witness_forms import source_digest as digest

FRAME_MARKER = '\n\nPROGRAM-GENERATED CONTEXT:\n'
stopped: Callable[[], bool]
from agentdojo.agent_pipeline.task_shield_defense import TaskShieldJudge


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    root: str | None
    via: str | None
    decision: Literal['allow', 'block']
    reason: str

    @model_validator(mode='after')
    def consistent(self):
        if not self.reason.strip():
            raise ValueError('reason must be non-empty for allow and block')
        if self.decision == 'allow':
            if not self.root or not self.root.strip() or not self.via or not self.via.strip():
                raise ValueError('allow requires non-empty root and via')
        elif self.root is not None or self.via is not None:
            raise ValueError('block requires root=null and via=null')
        return self


def history(messages):
    judge = TaskShieldJudge()
    judge.bind(copy.deepcopy(messages), {}, len(messages) - 1)
    return copy.deepcopy(judge.history)


def payload(frames, messages, call):
    return {'trusted_task_frames': copy.deepcopy(frames), 'conversation_history': history(messages[:-1]),
            'proposed_tool_call': {'tool': call.function, 'arguments': copy.deepcopy(call.args)},
            'related_content': get_text_content_as_str(messages[-1].get('content') or [])}


def validate_root(result, frames):
    if result.decision == 'allow' and result.root not in {f['id'] for f in frames}:
        raise ValueError('root is not an existing trusted_task_frames id')


class FrameAgent(BasePipelineElement):
    def __init__(self, llm, frames, recorder):
        self.llm, self.frames, self.recorder = llm, copy.deepcopy(frames), recorder

    def query(self, query, runtime, env, messages, extra_args):
        outgoing = copy.deepcopy(messages)
        system = next(m for m in outgoing if m['role'] == 'system')
        system['content'] = [text_content_block_from_string(get_text_content_as_str(system.get('content') or [])
             + FRAME_MARKER + json.dumps({'trusted_task_frames': self.frames}, ensure_ascii=False))]
        self.recorder.record({'stage': 'frame_agent_input', 'trusted_task_frames': self.frames,
                              'frame_sha256': digest(self.frames), 'messages': outgoing,
                              'native_tools': list(runtime.functions), 'witness_supplied': False})
        q, rt, e, updated, args = self.llm.query(query, runtime, env, outgoing, extra_args)
        # Keep Guard history in Task Shield's native visible-message format; do
        # not accumulate injected context or model-generated root/via there.
        return q, rt, e, [*messages, updated[-1]], args


class GuardExecutor(BasePipelineElement):
    def __init__(self, frames, recorder, stage, executor=None):
        self.frames, self.recorder, self.stage = copy.deepcopy(frames), recorder, stage
        self.executor = executor or ToolsExecutor()
        self.seen_call_ids = set()

    def query(self, query, runtime, env, messages, extra_args):
        if not messages or messages[-1]['role'] != 'assistant' or not messages[-1].get('tool_calls'):
            return query, runtime, env, messages, extra_args
        calls = messages[-1]['tool_calls']
        # Snapshot the complete batch's pre-execution history before any sibling
        # tool can run; all siblings receive this same visible boundary.
        inputs = [payload(self.frames, messages, c) for c in calls]
        results = []
        for call, packet in zip(calls, inputs):
            if stopped():
                raise RuntimeError('ConfigurationAuditError: STOP present')
            event = {'stage': 'guard_no_witness_decision', 'tool_call_id': call.id,
                     'guard_input': packet, 'frame_ids': [f['id'] for f in self.frames],
                     'executed': False, 'invalid_guard_output': False}
            if call.id in self.seen_call_ids:
                event.update(effective_decision='program_reject', feedback='Duplicate tool call ID; tool not executed.')
            elif call.function not in runtime.functions:
                event.update(effective_decision='program_reject', feedback='Invalid tool ' + call.function)
            else:
                try:
                    result = self.stage.run(packet, call.id, validator=lambda value: validate_root(value, self.frames))
                except StageFailure as exc:
                    event.update(effective_decision='program_reject', invalid_guard_output=True,
                                 feedback='Guard output invalid; tool not executed: ' + str(exc))
                else:
                    event.update(decision=result.model_dump(mode='json'), effective_decision=result.decision)
                    if result.decision == 'allow':
                        current = copy.deepcopy(messages[-1]); current['tool_calls'] = [call]
                        _, _, env, executed_messages, _ = self.executor.query(
                            query, runtime, env, [*messages[:-1], current], extra_args)
                        tool_message = executed_messages[-1]
                        assert tool_message['role'] == 'tool' and tool_message['tool_call_id'] == call.id
                        results.append(tool_message)
                        event.update(executed=True, tool_output=tool_message.get('content'), tool_error=tool_message.get('error'))
                    else:
                        event['feedback'] = 'Tool call blocked by runtime authorization guard: ' + result.reason
            if not event['executed']:
                results.append({'role': 'tool', 'content': [text_content_block_from_string('')],
                                'tool_call_id': call.id, 'tool_call': call, 'error': event['feedback']})
            self.recorder.record(event)
            self.seen_call_ids.add(call.id)
        return query, runtime, env, [*messages, *results], extra_args
