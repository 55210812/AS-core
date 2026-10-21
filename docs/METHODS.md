# 方法与输入边界

## 共用运行逻辑

`as_all.methods.build` 显式接收方法 ID、提供商、密钥、API 地址、案例输出目录和可选冻结任务键。保留来源代码的 18 轮工具循环及一次最终回答尝试。方法内部的格式修复流程、工具调用顺序和错误传播沿用来源实现。

Agent 开启 Thinking，Guard 与其他生成式防御关闭 Thinking，temperature=0。提供商客户端按调用侧显式标记 `agent` / `defense`；未送达连接错误与 429 最多两次传输重试，读超时或部分响应不会自动重发。原始响应、usage 与尝试记录保存在案例目录 `requests.jsonl`。

`usage.cost` 保留原账本统计函数：输入为供应商原始总输入，实际缓存及可核验非缓存另列；Qwen completion 已含 reasoning，Gemini candidate 与 thought 只合计一次。缺失 usage 和缓存字段各自计数。

## 基础方法

| 方法 | 主要实现 | 行为 |
|---|---|---|
| No Defense | `methods.compose`、`vendor/agentdojo/.../tool_execution.py` | 原模型提议与领域工具执行流程 |
| Tool Filtering | `reference/tool_filter_openai.py` | 更新后的严格工具名过滤；过滤请求使用独立防御客户端 |
| Prompt Repetition | `methods.compose` | 每轮工具执行后重复可信用户任务 |
| Input Delimiting | `methods.compose` | 以 `<<` / `>>` 标记工具输出，并添加原分隔提示 |
| Injection Detection | `vendor/agentdojo/.../pi_detector.py` | DeBERTa、message 模式、SAFE 标签、阈值 0.5 |
| IPIGuard | `vendor/agentdojo/.../ipiguard_aligned.py` | 官方行为对齐的 DAG 构造、遍历、模拟反馈和 reflection |
| Task Shield | `vendor/agentdojo/.../task_shield_defense.py` | Agent 开 Thinking、Judge 关 Thinking，保留原输入／输出过滤 |

检测器固定 `protectai/deberta-v3-base-prompt-injection-v2` revision `90c9989b1a342275dd0d1a95aad283c04e075671`；七个必要文件内容哈希见 `methods.DETECTOR_FILES`。已有缓存逐文件验证后复用，否则按固定 revision 下载。

IPIGuard 对齐来源是 `Greysahy/ipiguard@4e686ed2f62c135cb12564a4466daa95f3ace878`。严格 JSON 解析、原工具白名单、fake 后 real 执行、参数更新、DAG 错误和最多三轮 reflection 均保留；差分参考与测试位于 `reference/`。官方对齐序列化器只接入 IPIGuard 路径。

## Holistic

`src/as_all/holistic.py` 使用原始可信用户任务和原 `contribute_to_score` / `reason` 自评 Witness，不增加人工 Frame 或独立 Guard。零自评分不自动阻断工具。

Gemini 使用 A 版原生领域工具，将 Witness 放在各工具参数中，完整回传 native parts 与 `thoughtSignature`。Qwen 使用原代理工具和 system 领域工具注册表。

## AS-only 三组

`src/as_all/native_a.py` 复用 `reference/native_a/ablation_variants.py` 或 `reference/qwen_via/ablation_variants.py`，两提供商原 Prompt、Schema 和证据边界分别保留。

- Gemini：领域工具直接注册到 `functionDeclarations`，必填 `authorization_witness`；system 不重复附完整领域工具注册表。
- Qwen：使用 `submit_tool_call_with_authorization_witness` 代理，system 保留 `available_domain_tools`；沿用 via 强化版。
- 程序剥离 Witness 后，将原领域参数交给实际领域工具。
- Guard 输入为当前调用、原 Witness、root 引用 Frame 和明确引用的此前成功输出；不补证、不替换 root、不附完整历史。
- `as_only` 只做原结构检查，不发 Guard 请求。
- `as_only_guard` 仅 allow 执行；Block 保留原拒绝理由。
- `as_only_guard_frame_reminder` 仅在有效 Guard Block 后，追加全部当前冻结 Frame 与未完成任务提醒。协议拒绝不触发提醒。

Frame 位于 `data/manual_frames97.json`，每项为 `id` / `task` 简化结构。构建 API 用 `task_id` 选取当前任务，直接传入方法，不要求批量 manifest 或案例配置文件。

## Qwen 四种 Witness 形式

四组位于 `src/as_all/witness_forms.py`，均保留各自原 Prompt 和裁决 Schema，无 Frame 提醒。

| 方法 | Guard 输入 | 裁决字段 |
|---|---|---|
| `no_witness_guard` | 当前任务完整 Frame、当前调用、完整动作前可见历史、当前 Assistant 正文 | root、via、decision、reason |
| `reasoning_context_guard` | 当前任务完整 Frame、当前调用、提出该调用的原始 reasoning_content | decision、reason |
| `as_pg` | 无 Guard | 无 |
| `as_pg_guard` | AS / PG 引用 Frame 并集、当前调用、原 Witness、明确引用的此前成功输出 | action_support_valid、parameter_grounding_valid、decision、reason |

无 Witness 组从历史中去掉当前提出调用的整条 Assistant 回复；当前工具调用只在 proposed_tool_call 中。同轮多个提议的审核输入在执行任何工具前快照，避免将同轮结果当作先前证据。

Reasoning 组使用当前响应原始推理文字，不合成说明、不拼接历史推理；缺失时传空字符串。

AS + PG 包含 action_support 的 root / via 和联合 parameter_grounding 的 roots / via。非空参数缺 PG 时进入原协议流程；真实零参数调用将 PG 归一化为 null。有 Guard 时动作授权和参数授权均有效才执行。

这些方法的信息范围、授权验证对象和裁决字段有区别，比较时需保留对应方法身份。
