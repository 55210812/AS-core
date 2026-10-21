"""Offline differential tests against pinned upstream AST, no API or SDK needed."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from dataclasses import dataclass, field
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

class Call:
    def __init__(self, function, args, id):
        self.function, self.args, self.id = function, args, id

class Base:
    def __init__(self, client=None, model='qwen3.5-flash', temperature=0.0):
        self.client, self.model, self.temperature = client, model, temperature

class Client:
    def __init__(self, responses):
        self.responses, self.requests = iter(responses), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))],
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3))

def load(path, allowed=None):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom)) and
            (allowed is None or getattr(n, 'name', None) in allowed)]
    ns = dict(json=json, nx=nx, uuid=uuid, dataclass=dataclass, field=field,
              BasePipelineElement=Base, OpenAILLM=Base, EmptyEnv=lambda: None,
              FunctionCall=Call, text_content_block_from_string=lambda s: {'type':'text','content':s},
              tool_result_to_str=str, record_defense_event=lambda *a, **k: None,
              Logger=lambda: SimpleNamespace(get=lambda: SimpleNamespace(log=lambda *a: None)),
              topological_sort=nx.topological_sort)
    for name in ['ChatAssistantMessage','ChatToolResultMessage','ChatSystemMessage','ChatUserMessage',
                 'ChatCompletionSystemMessageParam','ChatCompletionUserMessageParam',
                 'ChatCompletionAssistantMessageParam','ChatCompletionToolMessageParam',
                 'ChatCompletionMessageToolCallParam']:
        ns[name] = dict
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns

local = load(ROOT/'src/agentdojo/agent_pipeline/ipiguard_official_defense.py')
wire = load(ROOT/'reference/ipiguard/llms/openai_llm.py', {'_tool_call_to_openai','_message_to_openai'})
up = load(ROOT/'reference/ipiguard/llms/ipiguard_llm.py')
up['_message_to_openai'] = wire['_message_to_openai']
execution = load(ROOT/'reference/ipiguard/tool_execution.py', {'DagToolsExecutor','DagToolsExecutionLoop'})
execution['whitelist'] = local['OFFICIAL_QUERY_TOOL_WHITELIST']

def messages():
    return [{'role':'system','content':'system'}, {'role':'user','content':'task'}]

def blocks(ms):
    return [{**m, 'content': [{'type':'text','content':m['content']}] if m['content'] is not None else None} for m in ms]

class Parity(unittest.TestCase):
    def test_query_whitelist(self):
        tree=ast.parse((ROOT/'reference/ipiguard/agentdojo/src/agentdojo/default_suites/v1/tools/tool_white_list.py').read_text())
        expected=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='whitelist'))
        self.assertEqual(expected,local['OFFICIAL_QUERY_TOOL_WHITELIST'])

    def test_construct_requests_and_dag(self):
        payload = json.dumps({'tool_calls':[{'id':1,'function_name':'read','args':{},'depends_on':[]}]})
        runtime = SimpleNamespace(functions={'read':SimpleNamespace(name='read',description='Read',parameters=SimpleNamespace(model_json_schema=lambda: {'type':'object'}))})
        a,b = Client(['{}',payload]),Client(['{}',payload])
        old = up['OpenAIConstructLLM'](a)
        new = local['OfficialIPIGuardConstructLLM'](b, 'qwen3.5-flash')
        ro=old.query('task',runtime,messages=messages(),extra_args={})
        rn=new.query('task',runtime,messages=blocks(messages()),extra_args={})
        self.assertEqual(a.requests,b.requests)
        self.assertEqual(up['OpenAITraverseLLM']()._dag_to_json_str(ro[4]['dag']),rn[4]['ipiguard_official_dag'].to_json())
        self.assertEqual(ro[3],local['_simple_chat_messages'](rn[3],'qwen3.5-flash'))

    def test_update_and_reflection_requests(self):
        for reflect in (False, True):
            a,b=Client(['{"args":{"x":"<unknown>: string","reason":"why"},"new_tool_calls":[]}']),Client(['{"args":{"x":"<unknown>: string","reason":"why"},"new_tool_calls":[]}'])
            old,new=up['OpenAITraverseLLM'](a),local['OfficialIPIGuardTraverseLLM'](b,'qwen3.5-flash')
            ca,cb=Call('read',{'x':'<unknown>: string'},'1'),Call('read',{'x':'<unknown>: string'},'1')
            extra={'current_tool_call':ca,'new_tool_calls':[],'error_messages':[(ca,'bad')],'input_tokens':0,'output_tokens':0}
            if reflect:
                old.query_reflection('task',None,messages=messages(),extra_args=extra)
                new.reflect_args(cb,[(cb,'bad')],blocks(messages()),{})
            else:
                old.query_args_update('task',None,messages=messages(),extra_args=extra)
                new.update_args('task',cb,blocks(messages()),{})
            self.assertEqual(a.requests,b.requests)
            self.assertEqual(ca.args,cb.args)
            self.assertEqual(cb.args['x'],'<unknown>: string')

    def test_strict_parse_and_api_no_fallback(self):
        for raw in ('```json\n{}\n```','prefix {}'):
            with self.assertRaises(json.JSONDecodeError): local['extract_json_object'](raw)
            with self.assertRaises(json.JSONDecodeError): up['OpenAIConstructLLM']().construct_dag(raw)
        c=Client([RuntimeError('provider failure'),'{}'])
        with self.assertRaises(RuntimeError): local['_chat_text'](c,'qwen3.5-flash',[],json_format=True)
        self.assertEqual(len(c.requests),1)

    def test_dag_duplicates_missing_and_cycle(self):
        for payload in [
            {'tool_calls':[{'id':1,'function_name':'a','args':{},'depends_on':[99]}]},
            {'tool_calls':[{'id':1,'function_name':'a','args':{},'depends_on':[2]},{'id':2,'function_name':'b','args':{},'depends_on':[1]}]},
            {'tool_calls':[{'id':1,'function_name':'a','args':{},'depends_on':[2]},{'id':1,'function_name':'b','args':{},'depends_on':[]},{'id':2,'function_name':'c','args':{},'depends_on':[]}]},
        ]:
            a=up['OpenAIConstructLLM']().construct_dag(json.dumps(payload))
            b=local['OfficialIPIGuardDAG'].from_payload(payload)
            self.assertEqual(list(a.edges),list(b.graph.edges))
            if nx.is_directed_acyclic_graph(a): self.assertEqual(list(nx.topological_sort(a)),list(b.topological_order()))
            else:
                with self.assertRaises(nx.NetworkXUnfeasible): list(b.topological_order())
        with self.assertRaises(KeyError): local['OfficialIPIGuardDAG'].from_payload({'tool_calls':[{'function':'alias'}]})

    def test_reflection_three_times_no_coercion_or_skips(self):
        ca,cb=Call('missing',{'x':"['a']"},'1'),Call('missing',{'x':"['a']"},'1')
        attempts=[[],[]]
        def runtime(i):
            def run(env,name,args):
                attempts[i].append(copy.deepcopy(args)); return '', 'bad'
            return SimpleNamespace(run_function=run,functions={})
        old=execution['DagToolsExecutor'](SimpleNamespace(query_reflection=lambda q,r,e,m,x:(q,r,e,m,x)))
        new=local['OfficialIPIGuardDagExecutor'](SimpleNamespace(reflect_args=lambda c,e,m,x:c))
        a=old._run_tool_call_with_reflection('task',runtime(0),messages=messages(),extra_args={'current_tool_call':ca})
        b=new._run_tool_call_with_reflection('task',runtime(1),None,blocks(messages()),cb,{})
        self.assertEqual(attempts[0],attempts[1]); self.assertEqual(len(attempts[1]),4)
        self.assertEqual(a[3][-1]['error'],b[1][-1]['error'])

    def test_complete_execution_requests_fake_then_real(self):
        payload = {'tool_calls':[
            {'id':1,'function_name':'get_webpage','args':{'url':'fixture'},'depends_on':[]},
            {'id':2,'function_name':'write','args':{'text':'value'},'depends_on':[1]}]}
        responses = ['{"new_tool_calls":[{"function_name":"write","args":{"text":"value"}}]}',
                     '{"new_tool_calls":[]}', 'done']
        a,b=Client(responses),Client(responses)
        upstream=up['OpenAITraverseLLM'](a)
        adapted=local['OfficialIPIGuardTraverseLLM'](b,'qwen3.5-flash')
        calls=[[],[]]
        def runtime(i):
            def run(env,name,args):
                calls[i].append((name,copy.deepcopy(args))); return 'success',None
            return SimpleNamespace(run_function=run,functions={})
        execution['uuid']=local['uuid']=SimpleNamespace(uuid4=lambda:'new-id')
        oe={'dag':up['OpenAIConstructLLM']().construct_dag(json.dumps(payload)), 'input_tokens':0,'output_tokens':0}
        ne={'ipiguard_official_dag':local['OfficialIPIGuardDAG'].from_payload(payload)}
        old=execution['DagToolsExecutionLoop'](execution['DagToolsExecutor'](upstream))
        new=local['OfficialIPIGuardDagExecutor'](adapted)
        ro=old.query('task',runtime(0),messages=messages(),extra_args=oe)
        rn=new.query('task',runtime(1),messages=blocks(messages()),extra_args=ne)
        self.assertEqual(a.requests,b.requests)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual([name for name,args in calls[1]],['get_webpage','write'])
        self.assertEqual([wire['_message_to_openai'](m) for m in ro[3]],
                         local['_simple_chat_messages'](rn[3],'qwen3.5-flash'))

    def test_null_new_calls_failure_timing(self):
        for unknown in (True, False):
            payload = {'tool_calls': [{'id':1,'function_name':'write',
                       'args':{'text':'<unknown>: string' if unknown else 'value'},'depends_on':[]}]}
            response = '{"args":{},"new_tool_calls":null}'
            a,b = Client([response]),Client([response])
            calls=[[],[]]
            def runtime(i):
                def run(env,name,args):
                    calls[i].append((name,copy.deepcopy(args))); return 'success',None
                return SimpleNamespace(run_function=run,functions={})
            old=execution['DagToolsExecutionLoop'](execution['DagToolsExecutor'](up['OpenAITraverseLLM'](a)))
            new=local['OfficialIPIGuardDagExecutor'](local['OfficialIPIGuardTraverseLLM'](b,'qwen3.5-flash'))
            oe={'dag':up['OpenAIConstructLLM']().construct_dag(json.dumps(payload)), 'input_tokens':0,'output_tokens':0}
            ne={'ipiguard_official_dag':local['OfficialIPIGuardDAG'].from_payload(payload)}
            with self.assertRaises(TypeError): old.query('task',runtime(0),messages=messages(),extra_args=oe)
            with self.assertRaises(TypeError): new.query('task',runtime(1),messages=blocks(messages()),extra_args=ne)
            self.assertEqual(calls[0],calls[1])
            self.assertEqual(len(calls[1]),0 if unknown else 1)
            self.assertEqual(len(a.requests),1)
            self.assertEqual(len(b.requests),1)
            self.assertEqual(a.requests,b.requests)

    def test_fake_receipt_and_scheduled_write(self):
        new=local['OfficialIPIGuardDagExecutor'](None)
        call=Call('write',{},'x'); fake=new._fake_command_result(call)
        self.assertEqual(fake['error'],'null')
        self.assertEqual(local['_simple_chat_messages']([fake],'qwen3.5-flash')[0]['content'],'null')
        self.assertIn('I’ll',fake['content'][0]['content'])
        # Missing args do not silently suppress commands; malformed calls raise.
        _,ms,_=new._handle_new_tool_calls('task',SimpleNamespace(functions={}),None,[],{},[{'function_name':'write','args':{}}])
        self.assertEqual(ms[-1]['error'],'null')
        with self.assertRaises(KeyError): new._handle_new_tool_calls('task',None,None,[],{},[{}])

if __name__ == '__main__': unittest.main()
