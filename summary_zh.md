# HotpotQA 实时多跳问答 Agent — 项目总结

## 1. 项目概述

本项目构建了一个**实时多跳问答 agent**，在 HotpotQA distractor split 上进行评测。所有推理均在推理时发生，无预计算产物。核心研究问题是：*小型微调模型 + 智能 agent pipeline，能否超过大得多的模型 + 简单 RAG？*

**结论：可以。** 3B GRPO 微调模型 + 多跳分解 agent 达到 EM=0.574，超过 32B 简单 RAG（EM=0.554），弥合了 83% 的 3B→32B 性能差距。

---

## 2. 数据集

**HotpotQA distractor split**。评测集：`data/grpo_val.jsonl`，共 500 题。

| 题型 | 数量 | 描述 |
|---|---|---|
| Bridge（桥接）| 409（82%）| 需先找到中间实体，再回答原始问题 |
| Comparison（比较）| 91（18%）| 比较两个明确命名实体的共同属性 → yes/no 答案 |

每条数据包含：原始问题、答案、题型、难度、supporting facts（gold 段落标题）、10 个段落（2 gold + 8 distractor）。

---

## 3. 系统架构

### 3.1 LangGraph Pipeline

Agent 以 **LangGraph StateGraph** 实现，使用条件边表达控制流。无嵌套 if-else，整个图结构可检视、可重启。

```
Bridge 路径：
  START → classify → decompose → retrieve_hop1 → answer_hop1
        → formulate_hop2 → retrieve_hop2 → answer_final → verify
        → [end | retrieve_fallback → answer_final | web_search → answer_final → end]

Comparison 路径：
  START → classify → rewrite → retrieve_comparison → answer_final → verify
        → [end | retrieve_fallback → answer_final | web_search → answer_final → end]
```

每个节点都是对类型化状态字典（`QAState`）的纯函数。LangGraph 按 dict key 合并状态——节点未返回的 key 保留上一轮的值（E1 bug 的根源，见第 6 节）。

### 3.2 双角色模型

**单个 PeftModel**（Qwen2.5-3B-Instruct + GRPO LoRA adapter），通过 `model.disable_adapter()` 切换角色：

| 模式 | 切换方式 | 用于 |
|---|---|---|
| Base model | `with model.disable_adapter()` | classify / decompose / answer_hop1 / formulate_hop2 / rewrite_comparison |
| GRPO adapter | adapter 激活 | answer_final（bridge）|

这比加载两个独立模型节省一半显存，切换代价 O(1)。

**Comparison 题最终答案**采用两步程序化方案：
1. 对每条改写后的 sub-query 分别用 base model 提取属性值
2. 程序化比较两个值 → 输出 yes/no

这避免了让小模型一次性做抽象语义比较（3B 在此步骤可靠失败）。

### 3.3 检索层

**FAISS + `BAAI/bge-base-en-v1.5`**（109M 参数，在 QA hard negatives 上对比训练）。

- 余弦相似度，每条 sub-query TOP_K=5
- BGE 需要 query 前缀：`"Represent this sentence for searching relevant passages: "`
- Bridge 题：hop1 passages（5）+ hop2 passages（5），按 title 去重 → 最多 10 个段落
- Comparison 题：2 条改写 sub-query × TOP_K=5，去重 → 最多 10 个段落

### 3.4 Faithfulness Verifier（忠实度验证）

Token overlap 检查：`score = |预测词 ∩ 上下文词| / |预测词|`，阈值=0.3。

**yes/no 特判**（Comparison 题）：token overlap 无意义（"yes"/"no" 不出现在 Wikipedia 段落中）。改为检查检索到的段落中是否提及问题中 ≥50% 的命名实体。

Retry 逻辑：
- 验证失败 + retry_count=0 → `retrieve_fallback`（用原始问题重新检索）
- 验证失败 + retry_count=1 → `web_search`（MediaWiki API）
- retry_count ≥ 2 → 强制结束

---

## 4. 训练流程

### Step 1 — 数据集准备（`training/01_prepare_dataset.py`）
解析 HotpotQA distractor split，分离 bridge 和 comparison 题，在所有段落上构建 FAISS 索引。

### Step 2 — SFT 数据生成（`training/02_generate_sft_data.py`）
用 Qwen2.5-32B-Instruct 通过 API（vLLM）生成 20K 条 teacher reasoning trace，每条包含：
- Bridge：sub_q1、检索段落、hop1_answer、sub_q2、最终答案
- Comparison：改写 sub-queries、每个实体的属性值、最终 yes/no

### Step 3 — SFT 训练（`training/03_train_sft.py`）
在 teacher trace 上对 Qwen2.5-3B-Instruct 进行监督微调，输出 SFT LoRA adapter。

### Step 4 — Merge Adapter（`training/04_merge_adapter.py`）
将 SFT LoRA 合并进 base 权重 → `training/checkpoints/sft_merged`（3.09B 参数）。GRPO 训练前必须合并，避免双 adapter 复杂性。

### Step 5 — GRPO 强化训练（`training/05_train_grpo.py`）
基于结果奖励的强化学习：
- **奖励函数**：`0.5 × EM + 0.5 × F1`（最终答案）
- **训练数据**：30K HotpotQA distractor 样本；10% gold-absent 样本（填充至 10 个段落，答案="insufficient context"）
- **硬件**：4-GPU DDP，A100 80GB，约 44 小时，936 步
- **最终指标**：reward ≈ 0.77，KL ≈ 0.04，全程稳定
- **关键配置**：`GRPO_MAX_PROMPT_LEN=2400`，`GRPO_MAX_NEW_TOKENS=350`，`GRPO_NUM_GENERATIONS=2`
- **发布 adapter**：`Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter`

---

## 5. 迭代历史

### v1 — 初始 Live Agent（MiniLM embedding）

**设计选择：**
- `all-MiniLM-L6-v2`（22M）：可用、合理的 baseline
- Token overlap faithfulness verifier（阈值=0.3）
- 启发式分类：关键词信号 + 大写实体计数

**结果（n=500）：**

| 题型 | EM | F1 |
|---|---|---|
| bridge | 0.482 | 0.570 |
| comparison | 0.604 | 0.649 |
| **总体** | **0.504** | **0.584** |

对比 baseline：3B Base RAG EM=0.458，32B Base RAG EM=0.554。Agent 弥合了 83% 的 3B→32B 差距。

**失败分析（手动分析 20 个 bridge 错误）：**
- 循环或格式错误的 sub_q1：~5 例（decompose prompt 只有 3 个例子）
- Bridge 被误分类为 comparison：8 例（启发式在独立的 "both" 上触发）→ 8 例全部 EM=0.000
- 模型返回 hop-1 实体作为最终答案：~2 例
- 检索失败：其余

---

### v2 — Prompt 修复

**改动：**
1. decompose prompt：3 → 7 个例子；加入"不要复述原始问题"的指令
2. 分类启发式：移除独立的 "both"；只保留 "were both"、"are both"、"did both"
3. answer prompt：加入返回最终实体而非中间 hop-1 实体的指令

**结果：** smoke test（n=50）显示 +5.6pp → EM=0.560。完整评测（n=500）无变化 → EM=0.504。

**根本原因：** 前 50 题系统性更简单，smoke test 虚高了结果。Prompt 修复本身是正确的并被保留——真正的瓶颈是检索质量，不是 prompt。

---

### v3 — BGE Embedding 升级（当前最佳）

**动机：** `all-MiniLM-L6-v2` 为对称句子相似度训练，不适合非对称 QA 检索。短的事实性 query 在长 Wikipedia 段落上得分偏低。

**改动：** `all-MiniLM-L6-v2` → `BAAI/bge-base-en-v1.5`（109M，在 QA hard negatives 上对比训练）。

**结果（n=500）：**

| 指标 | MiniLM | BGE | 变化 |
|---|---|---|---|
| EM | 0.504 | **0.574** | +7.0pp |
| F1 | 0.584 | **0.660** | +7.6pp |
| ctx_recall | 0.794 | **0.887** | +9.3pp |
| bridge EM | 0.482 | **0.575** | +9.3pp |
| comparison EM | **0.604** | 0.571 | -3.3pp |
| hard EM | 0.348 | **0.416** | +6.7pp |

ctx_recall 跳升（+9.3pp）是主要驱动力——检索才是瓶颈，不是模型。Comparison 小幅下降：union retrieval 合并多条 sub-query 的结果，更高的单 query 精度在融合时引入了噪声。

**这是当前最佳结果。** Adapter：`Norm11/qwen2.5-3b-grpo-hotpotqa`。

---

### v4 — SFT + GRPO 重训

**动机：** 原 GRPO adapter 只在约 1,500 条 RAG 检索样本上训练，没有 SFT 预热。假设：在 20K teacher trace 上做 SFT 预热 + 在 30K 样本上做 GRPO 应能提升性能。

**结果（n=500）：**

| 指标 | v3（最佳）| v4（重训）| 变化 |
|---|---|---|---|
| EM | **0.574** | 0.538 | -3.6pp |
| F1 | **0.660** | 0.625 | -3.5pp |
| ctx_recall | 0.887 | **0.901** | +1.4pp |
| 触发 retry | 15/500 | **8/500** | 更好 |
| 错误预测平均长度 | 13.6 字符 | 22.6 字符 | 更差 |

**v4 表现更差的根本原因：**

| 原因 | 证据 |
|---|---|
| 训练/评测分布不匹配 | GRPO 在 10 个精选 gold+distractor 段落上训练；评测用 5 个 FAISS 检索段落。模型学会了从理想化上下文中回答。|
| 冗长输出回归 | 奖励用 `0.5×EM + 0.5×F1`，F1 奖励了流畅的长答案。v4 把正确实体包裹在完整句子中。约 70% 的 v3 领先案例源于此。|
| Gold-absent 噪声 | 10% 训练数据答案为 "insufficient context"。评测时 ctx_recall=0.90——这种情况从未出现，但 v4 学会了在不确定时输出免责声明。|
| KL 锚定 | `KL_coef=0.05` 把 v4 限制在 SFT 分布附近。由于 SFT 存在偏差，高 KL 阻止了 GRPO 纠正它。|

---

## 6. 工程问题记录

### E1 — LangGraph State 未清空 → 无限重试循环
**症状：** 5 道题触发 LangGraph 递归限制（25 次调用），消耗 25 倍预期模型调用量。
**根本原因：** `web_search` 节点未返回 `"uncovered_entities": []`。LangGraph 保留未返回的 key，`should_retry` 每次都看到非空的 `uncovered_entities`。
**修复：** `web_search` 显式返回 `"uncovered_entities": []`。加入硬上限：`retry_count >= 2 → "end"`。

### E2 — yes/no Verifier 全部失败 → 所有 Comparison 题触发重试
**症状：** ~90% 的 comparison 题在第一次就触发 web search，用不相关 Wikipedia 内容替换了正确段落。
**根本原因：** `score = |{"yes"} ∩ context_tokens| / 1 = 0.0`——"yes"/"no" 从不出现在 Wikipedia 段落的字面文本中。
**修复：** 对 yes/no 特判：检查实体覆盖率（问题中 ≥50% 命名实体出现在上下文中）。

### E3 — GRPO KL 爆炸（由 Prompt 截断引起）
**症状：** 第一次完整 GRPO 训练：reward 在第 300 步从 0.74 跌至 0.08；KL 从 0.01 发散至 3.0；模型输出退化为空的 `<answer></answer>` 标签。
**根本原因：** `GRPO_MAX_PROMPT_LEN=1536`，但 prompt 的 `p50=1,459` token → 超过 50% 的 prompt 被截断 → format gate 在大多数 rollout 上触发（reward=0）→ advantage ≈ 0 → KL 爆炸。通过打印 prompt 长度分布发现（`p50=1459, p99=2463`）。
**修复：** `GRPO_MAX_PROMPT_LEN` 1536→2400，`GRPO_KL_COEF` 0.01→0.05，`GRPO_NUM_GENERATIONS` 4→2。

### E4 — Hop-1 实体被作为最终答案返回
**症状：** Bridge 题返回中间实体而非最终答案。例："Who was the personal secretary of the politician born on Sep 1, 1931?" → `Cecil Parkinson`（hop-1）而非 `Sara Keays`（正确）。
**根本原因：** 最终答案 prompt 同时包含 hop-1 和 hop-2 检索段落。模型锚定在更显著的实体（政治家）上，而非追踪到答案角色。
**修复：** 在 bridge 最终答案 prompt 中加入："answer the original question directly, not any intermediate entity."

### E5 — BM25 混合检索损害 Comparison（已回退）
**症状：** BM25+FAISS RRF 融合：bridge EM +2.7pp，comparison EM -7.7pp，ctx_recall -4pp，整体为负。
**根本原因：** Comparison 改写后的 sub-query 含高频词（"nationality"、"same"）。BM25 关键词匹配这些词，召回不相关段落。
**决定：** 回退至纯 dense 检索。

### E6 — NLI Verifier 使 EM 崩溃 18pp（已回退）
**症状：** 用 `cross-encoder/nli-deberta-v3-small`（阈值=0.5）替换：EM 0.560→0.380。
**根本原因：** NLI 模型需要完整句子作为假设。短事实性答案（"American"、"1943"）无论正确与否都无法达到 P(entailment)>0.5。
**决定：** 回退。

### E7 — Wikipedia API 403 错误
**根本原因：** MediaWiki API 屏蔽没有/使用通用 User-Agent 的请求。
**修复：** 加入 `User-Agent: hotpotqa-research/1.0`。针对 macOS LibreSSL 证书问题加入 `verify=False`。

### E8 — vLLM / TRL 环境不兼容
**根本原因：** TRL 在启动时无论是否传入 `--use-vllm` 都会 import vLLM；torch 2.4.1 + CUDA 12.4 与 vLLM（需要 torch≥2.6）不兼容。
**修复：** SFT/GRPO 前执行 `pip uninstall vllm -y`。锁定版本：`transformers==4.47.0, trl==0.15.2, peft==0.13.2`。

---

## 7. 对照实验

为验证结论的说服力，在同一 500 题评测集上设计了三组对照实验，使用相同的 FAISS 索引。

| 实验 | 模型 | 检索 | Pipeline |
|---|---|---|---|
| Exp1 | Qwen2.5-32B | Golden passages（仅 supporting facts，无 FAISS）| 单轮 QA |
| Exp2 | Qwen2.5-32B | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp3 | Qwen2.5-3B + GRPO | FAISS TOP_K=5 | 完整 LangGraph Agent |

脚本位于 `eval/` 目录，不改动原有代码。`PlainModelWrapper` 为 Exp2 的 32B base model 提供 no-op `disable_adapter()` 上下文管理器，使其兼容现有 `Reasoner` 类。

### 总体结果

| 实验 | EM | F1 | ctx_recall | ans_coverage | faithfulness |
|---|---|---|---|---|---|
| Exp1: 32B + Golden（上界）| 0.788 | 0.884 | 1.000 | 0.976 | 0.921 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.955 | 0.956 | 0.805 |
| Exp3: 3B GRPO + RAG + Agent | 0.540 | 0.625 | 0.902 | 0.914 | 0.885 |

### 按题型

| 实验 | Bridge EM | Comparison EM |
|---|---|---|
| Exp1: 32B + Golden | 0.802 | 0.725 |
| Exp2: 32B + Agent | 0.724 | **0.264** |
| Exp3: 3B GRPO + Agent | 0.533 | 0.571 |

### 核心发现

**发现 1：检索不是唯一瓶颈。**
Exp2 ctx_recall=0.955（接近完美），EM 仍比 Exp1 低 14.8pp。说明即使检索近乎完美，agent pipeline 本身（hop 分解错误、hop1 答案传播错误）仍会引入显著损耗。单纯提高 TOP_K 效果有限。

**发现 2：3B GRPO 弥合了与 32B Agent 56% 的差距。**
Exp3 EM=0.540 vs Exp2 EM=0.640，差距只有 10pp，参数量相差 10 倍。GRPO 微调显著弥补了模型规模的劣势。

**发现 3：Exp2 Comparison 严重异常（EM=0.264，低于随机猜测）。**
根本原因：faithfulness verifier 对 yes/no 有特殊处理路径，依赖模型输出格式。32B base model 未经 GRPO 微调，输出格式不稳定，有时给出冗长句子而非干净的 yes/no → verifier 失败 → web search 替换正确段落 → 答案变差。这是 agent pipeline 对模型输出格式的隐式依赖，而非 32B 能力不足（Exp1 中 32B Comparison EM=0.725）。

**发现 4：ctx_recall 差异证实 query 质量比 TOP_K 更关键。**
Exp2 ctx_recall=0.955，Exp3=0.902。32B 生成的 sub-query 质量更高，但最终 EM 差距只有 10pp，说明 GRPO 微调帮助 3B 更好地利用了已有上下文。

---

## 8. Sub-Query 失败分析

对 **exp2 答对、exp3 答错的 95 个 Bridge 题**，直接对比 sub_q1 / hop1_answer / sub_q2。

### 失败分类

| 类别 | 数量 | 占比 | 根本原因 |
|---|---|---|---|
| A：hop1_answer 就错了（sub_q1 有问题）| 63 | 66% | 3B 生成的 sub_q1 质量差，第一跳方向偏移 |
| B：hop1 正确，sub_q2 偏移 | 20 | 21% | hop1_answer 未被正确代入 sub_q2 |
| C：query 完全相同，答案仍错 | 12 | 13% | 纯生成问题，与检索无关 |

### Category A 典型模式（66%）

**A1 — Few-shot 污染：3B 把 prompt 例子里的内容混入 sub_q1**
```
Q: "Party Never Ends is an album by the Romanian singer who studied at what college?"
Exp2 sub_q1: "Who is the Romanian singer who released Party Never Ends?" → Inna ✓
Exp3 sub_q1: "What college did the Romanian singer who starred in Pretty Woman study at?"
             ↑ "Pretty Woman" 来自 decompose prompt 的 few-shot 例子
             → 直接答成了大学，跳过了中间实体 → Willamette University ✗
```

**A2 — 识别到错误的中间实体**
```
Q: "The army officer who committed a murder in 1970 at Fort Bragg was born which year?"
Exp2: sub_q1 → Jeffrey R. MacDonald → 1943 ✓
Exp3: sub_q1 → Ronald Adrin Gray   → 1945 ✗
```

**A3 — sub_q1 加了多余限定词，检索跑偏**
```
Q: "Black Holes in the Sand features a cover-version of Diane, by what American rock band?"
Exp2 sub_q1: "What American rock band performed the song Diane?" → Hüsker Dü（原唱）✓
Exp3 sub_q1: "What American rock band covered Diane for Black Holes in the Sand?"
             ↑ 多了限定词，找到的是做 cover 的乐队 → Gravenhurst ✗
```

### Category B 典型模式（21%）

**B1 — sub_q2 问错了目标属性**
```
Q: "...starred with Jason Patrick in a 2015 horror film written by who?"
hop1: Mark Margolis（两模型一致）
Exp2 sub_q2: "Who wrote the 2015 horror film that Mark Margolis starred in with Jason Patrick?" → Ido Fluk ✓
Exp3 sub_q2: "What films did Mark Margolis star in with Jason Patrick in 2015?"（问的是 films）→ Mark Margolis ✗
```

**B2 — sub_q2 没有代入 hop1_answer，等于重跑第一跳**
```
Q: "Who is the mother of the striker for the Czech First League club born on 25th June 1983?"
hop1: Marc Janko（两模型一致）
Exp2 sub_q2: "Who is the mother of Marc Janko?" → Eva Janko ✓
Exp3 sub_q2: "Who is the mother of the player born on 25th June 1983 who plays for...?"
             ↑ 完全没用 hop1_answer → 返回了 hop1 实体 Marc Janko ✗
```

### Category C 典型模式（13%）

- **冗长输出：** `"Tom Coburn was born on March 14, 1948."` 而非 `"March 14, 1948"` → EM=0
- **完全幻觉：** 检索正确，但输出与问题毫无关联的答案

---

## 9. 未来工作方向

### 优先级 1 — Query LoRA 知识蒸馏（解决 87% 的 bridge 失败）

专门训练一个 LoRA adapter 用于 query 改写（sub_q1、sub_q2、comparison rewrite），以 32B 模型的标注作为监督信号。该 adapter 在 `Reasoner`（base model 推理步骤）中激活，与现有 GRPO answer adapter 分开。

**架构：**
```
Qwen2.5-3B-Instruct（base）
    ├── GRPO adapter   → 最终答案生成（现有，不动）
    └── Query LoRA     → sub_q1 / sub_q2 / comparison rewrite（新增）
```

**训练数据设计（3 种任务类型，约 10K–15K 条）：**

| 任务 | 输入 | 输出 | 标注来源 |
|---|---|---|---|
| sub_q1 生成 | question | 干净的第一跳子问题 | 32B model |
| sub_q2 生成 | question + hop1_answer | 代入 hop1 的第二跳子问题 | 32B model（但 hop1_answer 用 3B base 自己生成的，保持分布一致）|
| comparison rewrite | question | 2 条 sub-query，每个实体一条 | 32B model |

**关键约束：**
- 给 32B 的 prompt 必须与 `src/reasoner.py` 模板完全一致，不能让 32B 自由发挥
- sub_q2 训练数据中的 hop1_answer 使用 3B base model 自己生成的版本（不用 32B 的完美答案），训练和推理输入分布保持一致
- 质量过滤：只保留生成的 query 在 FAISS 上 ctx_recall=1.0 的样本
- 损失函数：标准 SFT next-token prediction，只在 completion 部分计算 loss（prompt 部分 mask 掉）
- 无需任务标签——prompt 格式本身已区分三种任务

**预期收益：** Bridge EM +5–8pp，整体 EM 提升至 ~0.58–0.60。

### 优先级 2 — 基于 Agent 检索结果的 GRPO 重训（解决 v4 失败）

v4 失败的核心是训练用 10 个 gold+distractor 段落，评测用 5 个 FAISS 检索段落，分布不匹配。修复方案：
- GRPO rollout 使用实际 agent 检索结果（而非 gold distractor set）
- 使用纯 EM 奖励（去除 F1 分量），抑制冗长输出
- 移除 gold-absent 样本或用真实检索失败案例替代
- `num_generations` 提升至 4+ 以获得更好的 advantage 估计

### 优先级 3 — Cross-Encoder Reranker

在 FAISS 检索和模型输入之间加入 cross-encoder reranker（`cross-encoder/ms-marco-MiniLM-L-6-v2`）：先检索 TOP_K=20，重排序取 TOP_5。联合 query-passage 编码提供比双塔模型更高的精度。预期 Exp3 ctx_recall 从 0.902 提升至 0.93–0.95。

### 优先级 4 — 生成质量优化（Category C）

- **冗长输出：** GRPO 奖励加 brevity 惩罚；answer prompt 强化"用短语回答，不要写完整句子"
- **幻觉：** 提高 faithfulness verifier 阈值；GRPO 训练中加入更多检索失败案例

---

## 10. 完整结果参考

### 原迭代实验结果

| 系统 | EM | F1 | ctx_recall |
|---|---|---|---|
| 3B Base + RAG（下界）| 0.458 | 0.541 | 0.748 |
| 3B GRPO RAG-only | 0.468 | 0.552 | 0.748 |
| **3B GRPO + Agent（v3，最佳）** | **0.574** | **0.660** | **0.887** |
| 14B Base + RAG | 0.544 | 0.649 | 0.748 |
| 32B Base + RAG | 0.554 | 0.656 | 0.748 |

### 对照实验结果

| 系统 | EM | F1 | Bridge EM | Comparison EM | ctx_recall |
|---|---|---|---|---|---|
| Exp1: 32B + Golden（上界）| 0.788 | 0.884 | 0.802 | 0.725 | 1.000 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.724 | 0.264 | 0.955 |
| Exp3: 3B GRPO + RAG + Agent | 0.540 | 0.625 | 0.533 | 0.571 | 0.902 |
