# Judge Model 选型结论

本文件用于确定当前统计学 RAG 项目的第一版评估模型（judge model）。

适用范围：
- 当前 `eval/eval_set_template.jsonl` 的 36 条单轮评测集
- 当前 `eval/EVAL_METRICS.md` 中已经确定的 8 个指标
- 重点服务于生成层指标与部分拒答判定

## 一、judge model 在这个项目里负责什么

在当前指标体系里：

### 不需要 judge 的指标

- `Hit@3`
- `MRR@5`
- `Evidence Recall@5`
- `Refusal F1`（如果系统能结构化输出 `decision=answer/refuse`）
- `Over-Refusal Rate`（如果系统能结构化输出 `decision=answer/refuse`）

这些指标可以由脚本直接计算。

### 需要 judge 的指标

- `Answer Accuracy`
- `Key Point Coverage`
- `Faithfulness`
- `Refusal F1`（如果当前回答系统没有显式拒答标签）

因此，judge model 的核心要求不是“聊天强”，而是：
- 中文统计问答理解稳定
- 能严格按要求输出结构化 JSON
- 能基于给定证据做打分，而不是自由发挥
- 成本可控，便于反复跑评测

## 二、正式推荐结论

## 正式主推荐：`zai-org/GLM-4.6`

第一版正式建议你把 `zai-org/GLM-4.6` 作为主 judge model。

### 选择理由

#### 1. 它更适合中文评测场景

你当前 RAG 的知识库、题集、标准答案和关键点全部是中文统计学内容。

硅基流动官方在模型中心对智谱 GLM 系列的描述是：
- 在中文理解与生成、跨语言对话与知识问答中能力突出
- 在结构化输出、工具调用与代码生成等任务上具备良好泛化能力

对于 judge 任务来说，这两点非常关键：
- 中文理解强，意味着对“定义差一点”“术语不严谨”“关键点漏了一条”这类细粒度判断更稳
- 结构化输出强，意味着更容易稳定返回 JSON 评分结果

#### 2. 它比纯聊天模型更像“评分器”

你的 judge 任务不是生成漂亮答案，而是做如下工作：
- 对照 `reference_answer` 判断对错
- 对照 `key_points` 判断覆盖率
- 对照检索证据判断 `Faithfulness`

这个任务更看重：
- 指令遵循
- 稳定格式输出
- 中立打分

从当前平台信息看，`GLM-4.6` 同时支持：
- OpenAI 兼容 chat completions
- `response_format`
- `enable_thinking`

这意味着它很适合被你包装成“固定 JSON 输出的评分器”。

#### 3. 它虽然单价不低，但在你这个项目里总成本仍然很低

硅基流动当前价格显示：
- `zai-org/GLM-4.6`：输入 `¥3.5 / M tokens`，输出 `¥14 / M tokens`
- `deepseek-ai/DeepSeek-V3.2`：输入 `¥2 / M tokens`，输出 `¥3 / M tokens`
- `Qwen/Qwen3-32B`：输入 `¥1 / M tokens`，输出 `¥4 / M tokens`

单看单价，`GLM-4.6` 比 `Qwen/Qwen3-32B` 贵，也比 `DeepSeek-V3.2` 贵。

但结合你的实际规模：
- 一次完整评测只有 36 条
- judge 输出建议限制为短 JSON
- 所以单轮评测的总成本其实非常低

### 粗略成本估算

下面是我基于当前项目结构做的工程估算，不是官方报价结论，而是便于你决策的近似值。

假设每条样本送给 judge 的内容包括：
- 问题
- 标准答案
- key points
- 模型答案
- 检索上下文或证据摘要

则单条 judge 请求大致可能在：
- 输入：`2000 ~ 3500 tokens`
- 输出：`120 ~ 250 tokens`

按 36 条评测集估算，一次完整跑评测大约会消耗：
- 输入：`72K ~ 126K tokens`
- 输出：`4.3K ~ 9K tokens`

据此估算：
- `GLM-4.6`：约 `¥0.31 ~ ¥0.57 / 次完整评测`
- `DeepSeek-V3.2`：约 `¥0.16 ~ ¥0.28 / 次完整评测`
- `Qwen/Qwen3-32B`：约 `¥0.09 ~ ¥0.16 / 次完整评测`

结论：
- 这三者都不贵
- 在你这个项目里，judge 的第一优先级应该是“评得稳”，不是“再省几毛钱”

所以第一版直接用 `GLM-4.6` 是合理的。

## 三、备选模型

## 备选 1：`deepseek-ai/DeepSeek-V3.2`

这是我给你的第一备选，也是最适合做“复核 judge”的模型。

### 优点

- 硅基流动官方将其描述为兼具高计算效率与卓越推理和 Agent 性能
- 价格很好：输入 `¥2 / M tokens`，输出 `¥3 / M tokens`
- 上下文长度 `160K`，足够装下评测输入
- 支持 `enable_thinking`

### 适合它的角色

更适合做：
- 第二评委
- 有争议样本复核器
- Faithfulness 专项复核模型

### 我不把它放在主推荐位的原因

这部分是我的工程判断，不是硅基流动官方明说：
- `DeepSeek-V3.2` 的推理能力非常强，但 judge 任务并不只是“会推理”
- 你的第一版更需要“中文评分稳定 + JSON 结构稳定 + 少跑偏”
- 在这种面向中文知识库评测的项目里，我更愿意先把 `GLM-4.6` 作为主 judge，再把 `DeepSeek-V3.2` 放在复核位

### 推荐用法

如果你后面要做“双 judge 复核”，最佳组合就是：
- 主 judge：`zai-org/GLM-4.6`
- 复核 judge：`deepseek-ai/DeepSeek-V3.2`

## 备选 2：`Qwen/Qwen3-32B`

这是我给你的预算友好型备选。

### 优点

- 价格低：输入 `¥1 / M tokens`，输出 `¥4 / M tokens`
- 官方 API 文档明确支持 `enable_thinking`
- 在硅基流动生态案例中，`Qwen/Qwen3-32B` 被反复用作通用文本模型

### 适合它的角色

- 你想先快速搭出第一版 judge 流程
- 你后面准备大量重复跑评测
- 你希望 judge、generation、embedding 都尽量保持在通义系附近

### 不放到主推荐位的原因

这部分同样是我的工程判断：
- 它更适合做“高性价比通用模型”
- 但对于你这种中文统计知识库的细粒度评分任务，我仍然更看重 `GLM-4.6` 在中文与结构化输出上的适配感

## 四、不建议作为第一版主 judge 的模型

## 1. `deepseek-ai/DeepSeek-R1`

不建议作为第一版主 judge。

原因：
- 价格明显更高：输入 `¥4 / M tokens`，输出 `¥16 / M tokens`
- judge 任务不需要那么重的长链推理
- 这类强推理模型在评分任务里有时会“想太多”，反而增加格式漂移和解释冗长的问题

更适合：
- 个别高难样本人工复查
- 不是常规批量 judge

## 2. `Qwen/Qwen3.5-35B-A3B`

不建议作为第一版主 judge。

原因：
- 官方介绍里它非常强，但更偏“高能力通用模型”
- 输出价格较高：`¥12.8 / M tokens`
- 在你的这个评测集规模下不是不能用，而是没有显著必要

更适合：
- 做第二意见模型
- 或你后面想对 judge 能力再上一个台阶时使用

## 五、正式落地建议

## 第一版最简方案

正式推荐配置：
- `JUDGE_BASE_URL=https://api.siliconflow.cn/v1`
- `JUDGE_MODEL=zai-org/GLM-4.6`
- `JUDGE_TEMPERATURE=0`
- `JUDGE_ENABLE_THINKING=false`
- `JUDGE_MAX_TOKENS=600`
- `JUDGE_RESPONSE_FORMAT=json_object`

### 为什么这样配

- `temperature=0`
  让评分结果尽量稳定，减少同一答案多次评估结果不一致

- `enable_thinking=false`
  第一版更强调稳定、低成本和结构一致性

- `max_tokens=600`
  足够输出完整 JSON 和简短解释

- `response_format=json_object`
  便于脚本直接解析，后续你做结果汇总会轻松很多

## 第二版增强方案

如果你后面想把 judge 做得更稳，可以上双 judge：

- 主 judge：`zai-org/GLM-4.6`
- 复核 judge：`deepseek-ai/DeepSeek-V3.2`

### 触发复核的条件可以设为

- `Answer Accuracy` 小于等于 0.5
- `Faithfulness=0`
- `Key Point Coverage` 介于 `0.33 ~ 0.67`
- 两次评分波动超过阈值

这样可以只对少量有争议样本做二次评估，而不是所有样本都双跑。

## 六、关于“是否要和回答模型同家族”

这部分是我的工程推断，不是官方结论。

### 建议

如果你的回答模型本身就是通义千问系，最好不要让 `Qwen` 同时做你唯一的 judge。

原因：
- 同家族模型在表达偏好、术语风格、答案习惯上更接近
- 可能会放大“自家风格正确”的偏差

因此更稳妥的做法是：
- 若回答模型是 `Qwen`，judge 优先用 `GLM-4.6` 或 `DeepSeek-V3.2`
- 若回答模型是 `DeepSeek`，judge 优先用 `GLM-4.6`

## 七、最终结论

当前项目进入“选评估模型”阶段后，我给你的正式结论是：

### 第一版正式 judge model

- `zai-org/GLM-4.6`

### 第一备选

- `deepseek-ai/DeepSeek-V3.2`

### 预算友好备选

- `Qwen/Qwen3-32B`

### 推荐落地顺序

1. 先用 `GLM-4.6` 跑通单 judge 评测链路
2. 再根据误差样本引入 `DeepSeek-V3.2` 做复核
3. 后面再决定是否需要双 judge 或人工抽检

## 八、参考来源

以下信息基于硅基流动官方页面整理：
- 模型中心：https://www.siliconflow.cn/models
- 价格页面：https://siliconflow.cn/pricing
- Chat Completions API：https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
- JSON 模式文档：https://docs.siliconflow.cn/cn/userguide/guides/json-mode
