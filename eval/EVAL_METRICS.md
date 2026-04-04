# 第一版 RAG 评估指标确定稿

本文件用于正式确定当前统计学 RAG 项目的第一版评估指标。

适用范围：
- 当前 `eval/eval_set_template.jsonl` 的 36 条单轮评测集
- 仅评估单轮 RAG 问答
- 暂不覆盖 memory、多轮追问、流式输出体验等后续能力

当前评测集结构：
- 总样本数：36
- 可回答样本：32
- 拒答样本：4
- 题型分布：definition 8、method_selection 8、assumption 6、method_comparison 6、formula_interpretation 4、refusal 4

## 一、正式采用的指标

第一版正式采用 8 个指标，分成三层：检索层、生成层、拒答层。

### A. 检索层指标

这些指标只在 `should_refuse=false` 的 32 条可回答样本上计算。

#### 1. Hit@3

定义：
- 对每个问题，检查检索到的前 3 个 chunk 中，是否至少有 1 个 chunk 命中 `gold_evidence`
- 命中条件建议定义为：
  - `source` 相同
  - chunk 的页码区间与任一 `gold_evidence.page_start ~ page_end` 有重叠

公式：
- `Hit@3 = 命中样本数 / 32`

作用：
- 这是第一版最核心的检索指标
- 它直接回答一个问题：真正关键证据能不能在很靠前的位置被捞出来

为什么选它：
- 你的系统最终给大模型的上下文不会太长，前 3 条是否命中比“检索了很多但排得很后”更重要

#### 2. MRR@5

定义：
- 对每个问题，找到前 5 个检索结果中第一个命中 `gold_evidence` 的位置 `rank`
- 如果第一个命中出现在第 `rank` 位，则该题得分为 `1/rank`
- 如果前 5 没命中，则得分为 0

公式：
- `MRR@5 = 所有样本 reciprocal rank 的平均值`

作用：
- 衡量“正确证据排得有多靠前”
- 比 Hit@3 更细，能区分“第 1 条命中”和“第 5 条才命中”

为什么选它：
- 当前评测集的 `gold_evidence` 已经足够支持这个指标
- 它很适合观察 rerank 是否真的带来排序提升

#### 3. Evidence Recall@5

定义：
- 一个问题可能对应 1 到 2 段 `gold_evidence`
- 检查前 5 个检索结果一共覆盖了多少条标准证据

公式：
- 对单题：`matched_gold_evidence_count / total_gold_evidence_count`
- 对全体：取平均值

作用：
- 衡量检索是否只抓到一小块相关内容，还是能把关键证据尽量找全

为什么选它：
- 你的题集中有不少问题依赖 2 段证据
- 只看 Hit@k 容易忽略“命中但不完整”的情况

### B. 生成层指标

这些指标只在 `should_refuse=false` 的 32 条可回答样本上计算。

#### 4. Answer Accuracy

定义：
- 判断最终答案是否正确回答了问题，是否与 `reference_answer` 一致，是否存在实质性错误
- 这是第一版最核心的回答质量指标

评分方式：
- 建议由 judge model 按 0 到 4 分打分，再归一化到 0 到 1

评分标准建议：
- `4`：答案正确、完整、无明显错误
- `3`：答案基本正确，有少量次要遗漏
- `2`：部分正确，但遗漏关键结论或表述不够准确
- `1`：只有少量相关内容，整体回答偏离问题
- `0`：答案错误、幻觉明显，或没有回答到问题核心

归一化：
- `Answer Accuracy = score / 4`

为什么选它：
- 它是用户最终最关心的结果指标
- 很适合课堂展示，因为直观

#### 5. Key Point Coverage

定义：
- 看模型答案实际覆盖了多少个 `key_points`
- 每个 `key_point` 由 judge model 判定为“覆盖”或“未覆盖”

公式：
- 单题：`covered_key_points / total_key_points`
- 全体：取平均值

作用：
- 它比整体准确率更细，可以知道答案到底漏了什么

为什么选它：
- 当前评测集专门准备了 `key_points` 字段，这个指标和数据结构高度匹配
- 非常适合做误差分析

#### 6. Faithfulness

定义：
- 判断答案中的核心结论是否都能被检索到的上下文支持
- 如果答案出现检索证据没有支持的关键断言，视为不忠实

评分方式：
- 第一版建议使用严格二值：`0/1`

评分标准建议：
- `1`：答案中的关键结论均可由检索上下文支持，且没有明显外推或编造
- `0`：存在重要结论无法从上下文中得到支持，或与上下文冲突

公式：
- `Faithfulness = faithful_samples / 32`

为什么选它：
- RAG 和普通大模型问答最大的差别就在“是否基于证据回答”
- 这个指标必须保留，而且优先级很高

### C. 拒答层指标

这些指标在 36 条样本上计算，但重点观察 4 条 `should_refuse=true` 的拒答样本。

#### 7. Refusal F1

定义：
- 把系统是否拒答当成一个二分类任务
- 金标准来自 `should_refuse`

建议前提：
- 最好让被测系统输出结构化字段，例如：
  - `decision = answer`
  - `decision = refuse`
- 如果暂时没有结构化输出，再让 judge model 判定“该回答是否构成拒答”

公式：
- `Precision = 真拒答 / 系统判为拒答`
- `Recall = 真拒答 / 金标准拒答数`
- `F1 = 2PR / (P+R)`

为什么选它：
- 只有 4 条拒答样本，单看准确率不稳定
- F1 更能平衡“该拒不拒”和“过度拒答”

#### 8. Over-Refusal Rate

定义：
- 在 32 条本来可以回答的样本里，系统却选择拒答的比例

公式：
- `Over-Refusal Rate = answerable_samples_predicted_as_refusal / 32`

为什么选它：
- 对课堂展示来说，系统太保守也会显得不好用
- 它能专门盯住“明明能答却不答”的问题

## 二、第一版不正式采用的指标

下面这些指标不是没价值，而是当前这版数据和系统形态下不适合作为正式主指标。

### 1. Context Precision

暂不纳入主指标。

原因：
- 当前 `gold_evidence` 标的是“关键证据页”，不是“所有相关页”的完整标注
- 如果把未标到但实际上相关的 chunk 也算错，会对系统不公平

结论：
- 第一版先不用它做正式打分
- 后续若补全“所有相关 chunk”标注，再纳入

### 2. Response Relevancy

暂不纳入主指标。

原因：
- 它和 `Answer Accuracy` 有较强重叠
- 当前第一版更需要的是“答得对不对”和“有没有依据”

### 3. Citation Accuracy

暂不纳入主指标。

原因：
- 前提是系统输出格式里已经有规范化引用，例如 `source + page`
- 如果前端展示还没有固定引用格式，现在就上这个指标意义不大

结论：
- 等你后面把引用输出做稳定后，再加入

## 三、正式评估报告必须展示的内容

第一版评估时，正式报告中必须展示以下结果：

### 1. 总体主表

- Hit@3
- MRR@5
- Evidence Recall@5
- Answer Accuracy
- Key Point Coverage
- Faithfulness
- Refusal F1
- Over-Refusal Rate

### 2. 分组结果

至少按下面两个维度做分组统计：
- 按 `question_type` 分组
- 按 `difficulty` 分组

原因：
- 你的题型很均衡，如果只看一个总分，很容易掩盖问题
- 比如系统可能 definition 很强，但 method_comparison 很弱

## 四、推荐的主指标与辅助指标层级

为了课堂展示和后续迭代方便，建议把指标分层看：

### 主指标

- Hit@3
- Answer Accuracy
- Faithfulness
- Refusal F1

这 4 个指标最值得优先讲，因为它们分别对应：
- 能不能找到证据
- 能不能答对
- 是不是基于证据在答
- 越界问题能不能稳住

### 辅助指标

- MRR@5
- Evidence Recall@5
- Key Point Coverage
- Over-Refusal Rate

这些指标主要用于定位问题、做调参分析。

## 五、建议的演示阈值

如果你后面需要给这个项目设一个“课堂展示达标线”，我建议用下面的目标：

### 基本达标线

- `Hit@3 >= 0.75`
- `Answer Accuracy >= 0.75`
- `Faithfulness >= 0.85`
- `Refusal F1 >= 0.80`

### 展示效果较好

- `Hit@3 >= 0.85`
- `Answer Accuracy >= 0.82`
- `Faithfulness >= 0.90`
- `Refusal F1 >= 0.90`

说明：
- 当前知识库规模不算特别大，而且题集是围绕该知识库定制的
- 所以课堂展示阶段，适当把标准定得比通用开放域 RAG 更高是合理的

## 六、最终结论

基于当前 36 条第一版评测集，第一版正式指标体系确定为：

- 检索层：`Hit@3`、`MRR@5`、`Evidence Recall@5`
- 生成层：`Answer Accuracy`、`Key Point Coverage`、`Faithfulness`
- 拒答层：`Refusal F1`、`Over-Refusal Rate`

其中 4 个主指标为：
- `Hit@3`
- `Answer Accuracy`
- `Faithfulness`
- `Refusal F1`

后续工作顺序建议保持不变：
1. 评测集已完成
2. 指标已确定
3. 下一步开始选 judge model
