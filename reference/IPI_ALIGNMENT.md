# IPIGuard 官方行为对齐

目标：复现固定官方提交 `4e686ed2f62c135cb12564a4466daa95f3ace878`，不增强防御方法。只修改新目录中的适配实现；此前 fixed/legacy 代码、轨迹和分数保持原样。

## 本次对齐

- 保留此前五项对齐：无全局模拟/真实调用去重、标准工具调用历史、前置规划进入执行历史、参数更新不重复附加工具结果、temperature=0 显式传递。
- 官方构造/更新/扩展/reflection 提示文本逐字保留，包括零宽字符、官方构造 prompt 中 args 示例缺左括号，以及更新提示中重复的一句话。这些虽可能影响效果，但不是本地可以修好的问题。
- 工具定义标签缩进、输入第一条 system/user 消息的选择和前置规划拼接与官方一致。
- 用专用序列化适配当前 AgentDojo text content blocks；发送给 API 的 content 为官方的字符串格式，保留 system role、tool name、call ID、error 优先行为。无需依赖共享序列化器的 model-specific developer/reasoning 分支。
- 模拟调用保留官方弯引号文字，并设置字符串 `error="null"`，因此模型实际看到 `null` 而不是模拟成功正文。
- JSON 严格 json.loads；去掉抽取 fenced JSON、缺字段猜测/别名、解析失败空 DAG、pre-plan 失败 `{}` 等本地恢复。
- NetworkX DAG 保留官方节点插入/重复覆盖、悬空依赖、所有边和拓扑迭代行为；不忽略未知依赖或修复环。
- 去掉 missing-args 静默跳过、字符串列表自动转换、未知工具提前返回、自行过滤 malformed new calls。
- 保留运行一次后最多三轮 reflection；reflection 更新全部返回参数，包括 `reason`；不增加 unknown 二次校验。
- API/解析/update/expansion/reflection/final 异常向上传播；移除去 response_format 重试和吞异常 fallback。保留原始 None/空字符串响应差异。
- 查询白名单与固定官方文件逐项相同，不增加 URL 授权检查。

## 验证

`python tests/test_ipiguard_official_parity.py`：9 项通过。测试直接抽取同目录 reference 下官方 AST，使用同一模型响应与工具桩进行差分：

1. 白名单集合一致。
2. pre-plan、construct 完整 API 请求、DAG 和传递历史一致。
3. update/reflection 请求、参数与 unknown 保留一致。
4. fenced JSON / 带前缀 JSON 抛异常，API 失败只调用一次。
5. 重复节点、缺失依赖、环保持官方行为。
6. 缺参数/未知工具不跳过，不转换字符串数组，执行一次及三轮 reflection。
7. 完整 DAG 执行的 expansion/final 请求、fake 后 real 执行及最终 wire history 一致。
8. fake null 与弯引号、缺参数模拟调用保留、malformed 新调用报错。
9. update/expansion 返回 `new_tool_calls:null` 时，立即按官方 list.extend 语义报 TypeError；update 分支工具执行零次、API 一次，expansion 分支工具执行一次、API 一次，不延后失败时机。

`python -m py_compile src/agentdojo/agent_pipeline/ipiguard_official_defense.py`：通过。

这些是离线控制流及请求差分验证，不是新一轮性能实验。真实依赖导入及集成验证由本轮主流程执行。

## 有意保留与边界

保留官方未知参数可执行、漏规划、查询白名单按函数名授权、模拟回执 error=null、缺少完成核验等局限。不增加阈值、重规划或安全校验。

必要适配仍有：当前 AgentDojo 的文本块与消息类型、现有公开类/方法接口、extra_args 中 `ipiguard_official_dag` 名称、自包含 DAG 包装，以及 token 计量/旁路日志。调用方历史在本地复制后返回，官方原地修改 user 消息；pipeline 接收到的内容相同，但外部保留旧消息对象的调用方会观察到不同的对象别名行为。

当前模型/provider、SDK、工具运行时和任务套件版本与官方论文环境并非相同；差分测试证明已覆盖路径的适配行为，不能宣称整套官方环境逐比特复现，也不能以此推断论文效用。SDK 自带重试策略属于客户端配置，应在实验 manifest 中固定。
