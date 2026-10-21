"""Only replace model-visible Frame representation and root-ID selection."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(os.environ['ABLATION_RUNTIME'])
FRAMES = json.loads((ROOT / 'scripts/manual_frames.json').read_text())
ACTIVE_TASK = None


def validate(frames):
    if not isinstance(frames, list) or not frames:
        raise ValueError('Frame must be a nonempty list')
    for number, item in enumerate(frames, 1):
        if set(item) != {'id', 'task'} or item['id'] != f't{number}':
            raise ValueError('Frame items must contain only sequential id and task')
        if not isinstance(item['task'], str) or not item['task'].strip():
            raise ValueError('Frame task must be nonempty text')


def frame_refs(_original_frames):
    if ACTIVE_TASK not in FRAMES:
        raise ValueError('No active manual Frame task')
    frames = copy.deepcopy(FRAMES[ACTIVE_TASK])
    validate(frames)
    return frames, {f['id'] for f in frames}


def select(frames, root):
    validate(frames)
    selected = [f for f in frames if f['id'] == root]
    if len(selected) != 1:
        raise ValueError('unknown task root before Guard request')
    return copy.deepcopy(selected)


def resolved_guard_input(self, envelope, original_frames):
    import ablation_variants as variants
    frames, _ = frame_refs(original_frames)
    action = envelope.authorization_witness.action_support
    selected = select(frames, action.root)
    self._last_guard_frame_selection = {
        'referenced_root_ids': [action.root],
        'selected_frame_ids': [f['id'] for f in selected],
        'excluded_frame_ids': [f['id'] for f in frames if f not in selected],
    }
    evidence = []
    for output_id in action.authorization_evidence or []:
        record = self.context.output_registry[output_id]
        if not isinstance(record.get('output'), str):
            raise ValueError('referenced output is unavailable')
        evidence.append({'id': output_id, 'tool_name': record['tool_name'], 'output': record['output']})
    return variants.GuardPayload(trusted_task_frames=selected,
        proposed_tool_call=envelope.proposed_tool_call,
        authorization_witness=envelope.authorization_witness, prior_tool_outputs=evidence)


def install(variants):
    variants.ACTION_CONTRACT = variants.ACTION_CONTRACT.replace(
        'trusted_task_frames[*].refs, such as t1 or t1_c1',
        'trusted_task_frames[*].id, such as t1 or t2')
    variants.frame_refs = frame_refs
    variants.base._frames_with_inline_refs = frame_refs
    variants.guarded._frames_with_inline_refs = frame_refs
    variants.VariantMixin._resolved_guard_input = resolved_guard_input
