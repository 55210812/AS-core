# 共享模型与工具路径对齐

本目录统一以`Greysahy/ipiguard@4e686ed2f62c135cb12564a4466daa95f3ace878`所包含的IPIGuard及Injection Detection基线为参照。不能把不同AgentDojo版本的局部行为混用后声称逐项一致。

## 本次修正

1. 删除工具调用参数解析失败后额外生成一次的本地逻辑。官方API请求有自己的重试策略，但返回格式错误不触发第二次生成。本地原逻辑的回归测试修复前为1 failed/1 passed，失败证据是预期1次请求而实际2次；修复后通过。
2. 共享工具执行器不再自动把字符串数组转换成list。固定fork直接将原参数送进runtime；较新的AgentDojo提交有这项转换，但不混入本次固定fork复现。
3. 恢复固定fork的双重JSON解码：第一次解码结果为字符串时再解码一次。保留框架本身的FunctionCall参数类型校验，不增加恢复策略。
4. 共享OpenAI消息序列化恢复固定fork的字符串正文、system角色、tool调用关联和error优先行为；仅将当前AgentDojo的单text-block内容转成字符串。
5. 温度按固定fork直接传入，包括0与None。AgentDojo其他版本曾用`temperature or NOT_GIVEN`省略0，这不是本次参照版本的行为。
6. API重试排除条件恢复固定fork的BadRequestError。SDK重试次数、超时及模型提供商应在每次实验manifest另行固定。

正式pipeline保持原来的顺序与名称：`ipiguard_official`、`transformers_pi_detector`。本轮未引入新的防御策略。

## 验证范围

- `test_shared_model_path.py`：错误格式无额外生成、正常调用计量、官方API有效参数与wire消息一致（温度0/非零/None）、单层/双层JSON、工具参数不做字符串数组自动修复。
- `test_pipeline_integration.py`：在真实已安装框架/SDK/Pydantic环境下构造两条正式pipeline，使用脚本化模型响应、内存玩具工具和分类器桩，验证fake后计划工具仍真正执行、fake null的实际wire可见性、检测器默认参数和过滤流程。

离线测试不下载分类模型、不发出真实模型请求。验证对象是当前Qwen OpenAI兼容接口下的文本工具调用路径；其他模型专用的流式或reasoning接口没有在本轮建立官方等价结论。

当前任务数据/评分仍为已有AgentDojo benchmark v1.2.2环境，SDK、Pydantic与模型提供商不同于作者原始环境。此目录复现固定参考的方法与已覆盖执行路径，不宣称原论文环境或性能数字完全复现。
