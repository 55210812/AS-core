# 源码组成与完整性

AS-core 包含方法实现、模型接口、工具执行器、基准环境和冻结输入。

## 源码组成

- 十五种方法的组合、原 Prompt、Schema、工具执行、拒绝反馈和证据切片。
- Gemini 原生 A 版与 Qwen 原代理接口，两套独立传输协议、Thinking 和 usage 账本。
- AgentDojo 0.1.35 / benchmark v1.2.2 的四域环境、任务、工具、攻击构造器与评分器。
- semantic-action-frame-guard 核心、官方行为对齐参考及测试。
- 97 份任务与简化 Frame，Gemini / Qwen 两类各 949 组合冻结攻击数据。
- 必要 Python 依赖声明、上游源码说明及许可证。

`vendor/`、`reference/` 和 `data/` 分别保存依赖源码、方法参考与冻结输入；`source_manifest.json` 记录逐文件 SHA。上游自带的源码入口位于其依赖目录，顶层使用库 API。

## 接口与运行职责

- `methods.build` 显式接收方法选择、提供商、密钥、API 地址和案例输入。
- 单案例构建 API 设置一次最终回答尝试。
- `native_a.build` 与 `witness_forms.build` 支持直接传入已选冻结 Frame，保留低层原 case.json 读取兼容。
- 停止与进程管理由调用方负责。
- `usage.py` 中的 `cost` / `_add_usage` 函数负责账本统计。

方法通过 Python API 调用；批量调度、监控与结果汇总由调用方负责。

## 来源验证

`source_manifest.json` 记录源码文件与算法符号的 SHA。`tests/test_core_export.py` 校验完整性、方法组合及账本算法一致性，以及独立 API 的构建行为。

实际接入路径的测试使用 `vendor/agentdojo`；`reference/tests` 的共享模型与工具差分测试需单独加载 `reference/src`。两者保留来源仓库中不同的接入职责，测试命令见 README。

AgentDojo 使用 MIT；IPIGuard 使用 Apache-2.0。来源及许可证位置见 `THIRD_PARTY_NOTICES.md` 和 `reference/reference/ipiguard-provenance.json`。
