# AS-core

Gemini 与 Qwen 的 Agent 防御方法源码库，基于 [AS-all v0.4.0](https://github.com/55210812/AS-all/tree/f30245d4c4e8b73c03a45b637a4242bd199b3d11) 整理。

保留方法实现、Prompt、Witness Schema、模型协议、工具执行器、AgentDojo 模拟环境与评分器，以及冻结任务、Frame 和攻击输入。仓库通过 Python API 构建单个方法；批量实验的一键启动、安装器、调度、监控和 Git 反馈上传由调用方管理。

## 方法

| 方法 | ID | 提供商 |
|---|---|---|
| No Defense | `no_defense` | Gemini / Qwen |
| Tool Filtering | `tool_filter` | Gemini / Qwen |
| Prompt Repetition | `repeat_user_prompt` | Gemini / Qwen |
| Input Delimiting | `spotlighting_with_delimiting` | Gemini / Qwen |
| Injection Detection | `transformers_pi_detector` | Gemini / Qwen |
| IPIGuard | `ipiguard_official` | Gemini / Qwen |
| Task Shield | `task_shield_guard` | Gemini / Qwen |
| Holistic | `holistic` | Gemini / Qwen |
| AS-only | `as_only` | Gemini / Qwen |
| AS-only + Guard | `as_only_guard` | Gemini / Qwen |
| AS-only + Guard + Frame 提醒 | `as_only_guard_frame_reminder` | Gemini / Qwen |
| 无 Witness + 全历史 Guard | `no_witness_guard` | Qwen |
| Reasoning Context + Guard | `reasoning_context_guard` | Qwen |
| AS + PG | `as_pg` | Qwen |
| AS + PG + Guard | `as_pg_guard` | Qwen |

Gemini 固定 `gemini-2.5-flash`，走原生 `generateContent` 接口；Qwen 固定 `qwen3.5-flash`，走 OpenAI 兼容接口。Gemini 的 AS-only / Holistic 保留原生领域工具 A 版，Qwen 保留原代理工具及 system 领域工具注册表。

## 安装依赖

使用 Python 3.12，在仓库根目录安装本地依赖和当前项目：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./vendor/agentdojo -e ./vendor/core -e '.[dev]'
```

保留源码目录并使用 editable 安装：方法会读取仓库内的 `reference/` 与 `data/`。`as_all` Python 导入名保持与来源源码一致，发行包名为 `as-core`。

Injection Detection 使用固定 revision 的 DeBERTa 检测器，优先 CUDA、否则 CPU。首次构建该方法会按需下载权重，模型来源与固定版本见 [方法说明](docs/METHODS.md)。

## Python API

以下示例构建一个 Qwen AS-only + Guard，并在 AgentDojo Slack 模拟环境中执行一条 Clean 任务。显式执行任务时会产生模型 API 费用。

```python
import os
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suite
from as_all.methods import build
from as_all.usage import cost

task_id = "slack/user_task_1"
case_dir = Path("results/example")
pipeline = build(
    "as_only_guard",
    case_dir,
    provider="qwen",
    key=os.environ["QWEN_API_KEY"],
    base_url=os.environ["QWEN_BASE_URL"],
    task_id=task_id,
)
suite = get_suite("v1.2.2", "slack")
task = suite.get_user_task_by_id("user_task_1")
utility, asr = suite.run_task_with_pipeline(
    pipeline, task, injection_task=None, injections={}
)
print(utility, asr)
print(cost(case_dir / "requests.jsonl"))
```

使用 Gemini 时将 `provider` 设为 `gemini`，传入 Gemini 密钥与原生 API 地址，例如 `https://generativelanguage.googleapis.com/v1beta`。构建方法本身不发模型请求；实际调用 `query` 或任务执行函数才发出请求。

`task_id` 是 AS-only 三组和 Qwen 四种 Witness 形式所需的冻结 Frame 选择键；其他方法可省略。任务文本须与所选键对应，可从 `data/task_texts97.json` 读取。每个案例使用独立 `case_dir`。

**每个进程只构建并执行一个方法案例。** 部分原方法通过模块级 hook 接线，方法和案例之间应使用独立进程，避免状态与原生签名混用。仓库不创建进程池或后台会话。

## 源码导航

- `src/as_all/methods.py`：十五方法索引、单案例构建 API 与方法组合。
- `src/as_all/native_a.py`、`a_schema.py`：Gemini 原生领域工具与 Witness 包装。
- `src/as_all/holistic.py`、`witness_forms.py`：Holistic 与四种原版 Witness 形式接线。
- `src/as_all/transport.py`、`qwen_transport.py`：提供商协议、Thinking、重试和请求账本。
- `src/as_all/usage.py`：按 Agent / 防御侧汇总实际 usage、缓存和请求耗时。
- `vendor/agentdojo/`：框架、工具、任务环境、攻击构造器与评分器。
- `vendor/core/`：Frame、Witness、Guard Schema、Prompt 和运行时核心。
- `reference/`：原方法源码、冻结 Prompt、官方行为对齐参考及差分测试。
- `data/`：97 任务、简化 Frame、两提供商各自的两类 949 组合攻击输入。

详细输入边界见 [方法说明](docs/METHODS.md)，来源和导出边界见 [来源说明](docs/SOURCE.md)，逐文件 SHA 见 `source_manifest.json`。

## 离线测试

```bash
python -m pytest tests -q
PYTHONPATH=reference/src:src:vendor/core/src python -m pytest reference/tests -q
```

测试使用 Mock 模型和模拟工具；检测器测试使用分类器桩。第二条命令单独加载冻结官方对齐参考树，验证参考行为；第一条命令验证本库实际接入路径。

## 许可证

项目代码使用 MIT；AgentDojo、IPIGuard 等第三方代码按各自许可证保留。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
