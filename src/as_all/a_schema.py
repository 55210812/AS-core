import copy
import json

def resolve(ref, root):
    assert ref.startswith('#/$defs/'), 'unsupported schema reference'
    return root['$defs'][ref.removeprefix('#/$defs/')]

def check_schema(schema, root=None):
    """Reject unsupported keywords; covers the exact frozen schemas, not general JSON Schema."""
    root = schema if root is None else root
    allowed = {'$defs', '$ref', 'additionalProperties', 'anyOf', 'default', 'description', 'enum',
               'items', 'minItems', 'minLength', 'minimum', 'maximum', 'properties', 'required', 'title', 'type'}
    assert not set(schema)-allowed, 'unsupported schema keyword'
    if '$ref' in schema:
        resolve(schema['$ref'], root)
    for field in ['properties', '$defs']:
        for child in schema.get(field, {}).values():
            check_schema(child, root)
    for field in ['items', 'additionalProperties']:
        if isinstance(schema.get(field), dict):
            check_schema(schema[field], root)
    for child in schema.get('anyOf', []):
        check_schema(child, root)

def text(body):
    return '\n'.join(p['text'] for p in body['systemInstruction']['parts'])

def context(body):
    return json.loads(text(body).split('PROGRAM-GENERATED CONTEXT:\n', 1)[1])

def witness_tools(no_witness, proxy):
    envelope = proxy['tools'][0]['functionDeclarations'][0]['parametersJsonSchema']
    witness = envelope['properties']['authorization_witness']
    needed = set()
    def refs(v):
        if isinstance(v, dict):
            if '$ref' in v:
                name = v['$ref'].removeprefix('#/$defs/')
                if name not in needed:
                    needed.add(name)
                    refs(envelope['$defs'][name])
            for k, child in v.items():
                if k != '$ref':
                    refs(child)
        elif isinstance(v, list):
            for child in v:
                refs(child)
    refs(witness)
    def rename(v):
        if isinstance(v, dict):
            return {k: '#/$defs/witness__'+x.removeprefix('#/$defs/') if k == '$ref' else rename(x)
                    for k, x in v.items()}
        if isinstance(v, list):
            return [rename(x) for x in v]
        return v
    tools = copy.deepcopy(no_witness['tools'])
    for tool in tools[0]['functionDeclarations']:
        schema = tool['parametersJsonSchema']
        assert schema['type'] == 'object' and 'authorization_witness' not in schema.get('properties', {})
        schema['properties']['authorization_witness'] = rename(witness)
        schema.setdefault('required', []).append('authorization_witness')
        definitions = schema.setdefault('$defs', {})
        for name in needed:
            key = 'witness__'+name
            assert key not in definitions
            definitions[key] = rename(envelope['$defs'][name])
        check_schema(schema)
    return tools

def transform(no_witness, proxy, *, holistic=False):
    body = copy.deepcopy(proxy)
    system = text(body)
    old = ['For every domain action, call only submit_tool_call_with_authorization_witness.',
           'In the same proxy call provide one proposed_tool_call and one authorization_witness.']
    new = ['For every domain action, call the corresponding declared domain function and include authorization_witness in its arguments.',
           'In each call provide the original domain arguments and one authorization_witness.']
    if holistic:
        old = ['For each new domain action call only submit_tool_call_with_authorization_witness.',
               'In that same proxy call provide proposed_tool_call and authorization_witness.']
        new = ['For each new domain action call the corresponding declared domain function.',
               'In that same call provide the original domain arguments and authorization_witness.']
    for before, after in zip(old, new):
        assert system.count(before) == 1
        system = system.replace(before, after)
    system = system.replace('without calling the proxy.', 'without calling a tool.')
    before, payload = system.split('PROGRAM-GENERATED CONTEXT:\n', 1)
    data = json.loads(payload)
    authority_key = 'trusted_user_task' if holistic else 'trusted_task_frames'
    assert set(data) == {authority_key, 'available_domain_tools'}
    data.pop('available_domain_tools')
    body['systemInstruction'] = {'parts': [{'text': before+'PROGRAM-GENERATED CONTEXT:\n'+json.dumps(data, ensure_ascii=False)}]}
    body['tools'] = witness_tools(no_witness, proxy)
    return body

