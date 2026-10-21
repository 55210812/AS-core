# Injection Detection 官方基线对齐

此目录的 `src/agentdojo/agent_pipeline/pi_detector.py` 已与历史 legacy 文件逐字一致，因此本次无需再次修改检测器生产源码。新目录未包含此前实验修复版的 error 通道检测、清除 error 和增强日志逻辑。

## 官方依据与保留行为

自包含参考：`reference/ipiguard/pi_detector.py`，提交及来源见 `reference/ipiguard-provenance.json`。

- 模型 `protectai/deberta-v3-base-prompt-injection-v2`，`safe_label="SAFE"`，安全概率 `< 0.5` 判为注入；等于 0.5 放行。
- 只检测 content，不读取或清除 error。保留上游漏洞用于忠实 baseline，不称其为安全修复版。
- 保留 message/full_conversation 两种模式及连续末尾工具消息从后向前检测的顺序；原消息列表顺序不变。
- 判为注入后替换 content 或按配置抛出异常，不增加恢复、重试、去重或写操作保护。
- 原样调用分类 pipeline，不添加截断、最大长度或阈值调整。
- `agent_pipeline.py` 的默认接线仍显式指定 SAFE、0.5、message；离线 AST 检查已覆盖。

## 必需的版本适配边界

固定官方版本使用字符串 content；当前框架使用 content blocks。保留历史适配：检测时提取文本，屏蔽时创建文本 block。标准 ToolsExecutor（含错误分支）始终产生一个 text block，离线差分验证覆盖这一实际 benchmark 输入范围。

不声称对任意多模态/多 block 输入完全等价：历史适配的 transform 逐个替换 text block，并保留非 text block；固定官方没有这类输入类型。本次未扩大方法或输入范围。

## 验证

执行 `tests/test_detector_official_parity.py`：7 项离线差分测试通过。直接执行固定官方和本地检测器源码，仅替换框架依赖与模型构造；content 转换 helpers 从当前源码抽取执行。覆盖 error-only 漏检保持一致、content 屏蔽但 error 保留、工具检测顺序、两种 mode、abort/early return、SAFE/INJECTION 映射、0.5 边界、长文本原样传给 pipeline，以及正式 pipeline 配置。

运行命令：

```sh
/opt/anaconda3/bin/python -m unittest discover -s outputs/baseline-official-aligned-20261014/tests -p test_detector_official_parity.py -v
/opt/anaconda3/bin/python -m py_compile outputs/baseline-official-aligned-20261014/src/agentdojo/agent_pipeline/pi_detector.py outputs/baseline-official-aligned-20261014/tests/test_detector_official_parity.py
```

未联网、未加载分类模型、未安装依赖、未重跑付费实验。该验证证明代码路径与配置对齐，不能证明模型效果或不同 transformers 版本的数值一致性。系统 `python3` 为较旧 Python，不能执行现有源码的 Python 3.10+ union 类型；使用已有 Anaconda Python 后通过。

SHA256：

- 官方：`f97ea0c8caccac47daee6135b8a1a6b7ac910481266ce78d499e8d6ebc1cce96`
- 本地：`9008a2c063ed5cf826f4de23468d4f78369b127782d398a4e5933561e746ecb8`

历史快照与历史实验结果均未修改；原 error 修复版及 0.92 变体继续仅作为独立辅助实验留档。
