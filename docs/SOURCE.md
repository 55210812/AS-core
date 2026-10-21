# 来源与导出边界

基线为公开仓库 `55210812/AS-all` v0.4.0，提交 `f30245d4c4e8b73c03a45b637a4242bd199b3d11`。AS-core 独立维护，原仓库保持不变。

## 保留

- 十五种方法的组合、原 Prompt、Schema、工具执行、拒绝反馈和证据切片。
- Gemini 原生 A 版与 Qwen 原代理接口，两套独立传输协议、Thinking 和 usage 账本。
- AgentDojo 0.1.35 / benchmark v1.2.2 的四域环境、任务、工具、攻击构造器与评分器。
- semantic-action-frame-guard 核心、官方行为对齐参考及测试。
- 97 份任务与简化 Frame，Gemini / Qwen 两类各 949 组合冻结攻击数据。
- 必要 Python 依赖声明、上游源码说明及许可证。

`vendor/`、`reference/` 和 `data/` 从基线逐文件原样保留；`source_manifest.json` 记录 SHA。上游自带的源码入口也保留在其依赖目录，顶层使用库 API。

## 导出适配

- `methods.build` 从实验 profile / 环境凭据读取改为显式 Python 参数；保留方法选择和提供商专用分支。
- 将原 worker 设置的一次最终回答尝试绑定到单案例构建 API。
- `native_a.build` 与 `witness_forms.build` 支持直接传入已选冻结 Frame，保留低层原 case.json 读取兼容。
- 全历史 / Reasoning 组件的停止回调解除对批量父目录 STOP.json 的依赖；停止与进程管理交给调用方。
- 账本 `cost` / `_add_usage` 函数原样提取到 `usage.py`。
- 保留 `as_all` 导入名，发行包更名为 `as-core`。

顶层安装启动器、命令行批量调度、实验 profile、后台会话与 watchdog、反馈上传、结果汇总生成器和相关发布配置不随本库导出；方法算法和原始请求日志逻辑保留。

## 来源验证

`source_manifest.json` 绑定基线提交、原样保留文件 SHA、算法符号 SHA 与适配文件 SHA。`tests/test_core_export.py` 校验完整性、原方法组合及账本算法一致性，以及独立 API 的构建行为。

实际接入路径的测试使用 `vendor/agentdojo`；`reference/tests` 的共享模型与工具差分测试需单独加载 `reference/src`。两者保留来源仓库中不同的接入职责，测试命令见 README。

AgentDojo 使用 MIT；IPIGuard 使用 Apache-2.0。来源及许可证位置见 `THIRD_PARTY_NOTICES.md` 和 `reference/reference/ipiguard-provenance.json`。
