# Eval Analysis — Two-Phase Experiment History

实验分为两个阶段，对应两个不同的 embedding 方案。两个阶段使用完全相同的模型、pipeline 和评测集（500 题），唯一变量是检索器。

---

## Phase 1 — MiniLM Embedding (`all-MiniLM-L6-v2`)

### 背景

项目最初使用 `all-MiniLM-L6-v2` 作为 embedding 模型。HotpotQA distractor 设置下，直接用原始问题检索的 ctx_recall 约为 0.748，存在明显的检索瓶颈——gold passage 经常不在 top-5 里。

### 实验结果

| 系统 | EM | F1 | ctx_recall |
|---|---|---|---|
| 3B Base + RAG（下界） | 0.458 | 0.541 | 0.748 |
| 3B GRPO RAG-only | 0.468 | 0.552 | 0.748 |
| **3B GRPO + Agent（最优）** | **0.574** | **0.660** | **0.887** |
| 14B Base + RAG | 0.544 | 0.649 | 0.748 |
| 32B Base + RAG | 0.554 | 0.656 | 0.748 |

按题型（3B GRPO + Agent）：

| 题型 | EM | F1 | n |
|---|---|---|---|
| Bridge | 0.570 | 0.657 | 409 |
| Comparison | 0.593 | 0.643 | 91 |

### 核心发现

**多跳分解是有效的（+10.6pp）。** MiniLM 下，naive RAG EM=0.458，而多跳 Agent EM=0.574，差距 11.6pp。这是因为：

- 直接检索 ctx_recall=0.748，gold passage 经常缺失
- 多跳分解将原始问题拆成 sub_q1 → hop1 → sub_q2，每个子查询更精准，把 ctx_recall 提升到 0.887
- 检索质量的提升直接带动了最终答案质量

**3B + Agent 打败 32B naive RAG（+2.0pp）。** GRPO 训练（+1.0pp）+ Agent 分解（+10.6pp）两个贡献叠加，让 3B 模型超越了 14B 和 32B 的简单 RAG。

---

## Phase 2 — BGE Embedding (`BAAI/bge-base-en-v1.5`)

### 背景

将 embedding 模型升级为 `BAAI/bge-base-en-v1.5` 后，ctx_recall 从 0.748 提升至 0.907（+15.9pp）。检索瓶颈基本消除，这根本性地改变了各组件的相对贡献。

在此阶段进行了 6 组受控实验，所有实验使用相同的 FAISS 索引和 500 题评测集。

### 实验设计

| 实验 | 模型 | 检索策略 | Pipeline |
|---|---|---|---|
| Exp0 | Qwen2.5-3B base（无 adapter） | 原始问题单跳检索 | 单轮 QA |
| Exp1 | Qwen2.5-32B-Instruct | Golden passages（oracle） | 单轮 QA（上界） |
| Exp2 | Qwen2.5-32B-Instruct | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp3 | Qwen2.5-3B + GRPO adapter | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp4 | Qwen2.5-3B + GRPO + Query LoRA | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp5 | Qwen2.5-3B + GRPO adapter | comparison→单跳，bridge→Agent | 自适应路由 |
| Exp6 | Qwen2.5-3B + GRPO adapter | 原始问题单跳检索 | 单轮 QA |

脚本位于 `eval/` 目录：`exp0_3b_naive_rag.py`、`exp1_upperbound.py`、`exp2_32b_agent.py`、`exp3_3b_grpo_agent.py`、`exp5_adaptive_routing.py`、`exp6_grpo_naive_rag.py`；Exp4 位于 `training/query_lora/04_eval_e2e.py`。

### 总体结果

| 实验 | EM | F1 | ctx_recall | ans_coverage | faithfulness |
|---|---|---|---|---|---|
| Exp0: 3B base + Naive RAG | 0.568 | 0.657 | 0.907 | 0.868 | 0.908 |
| Exp1: 32B + Golden（上界） | 0.788 | 0.884 | 1.000 | 0.976 | 0.921 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.955 | 0.956 | 0.805 |
| Exp3: 3B GRPO + Agent | 0.540 | 0.625 | 0.902 | 0.914 | 0.885 |
| Exp4: 3B GRPO + Query LoRA + Agent | 0.550 | 0.639 | 0.931 | 0.944 | — |
| Exp5: Adaptive routing | 0.544 | 0.640 | 0.901 | 0.912 | 0.900 |
| **Exp6: 3B GRPO + Naive RAG** | **0.586** | **0.665** | **0.907** | **0.868** | **0.904** |

### 按题型

| 实验 | Bridge EM | Comparison EM |
|---|---|---|
| Exp0: 3B base + Naive RAG | 0.560 | 0.604 |
| Exp1: 32B + Golden | 0.802 | 0.725 |
| Exp2: 32B + Agent | 0.724 | **0.264** |
| Exp3: 3B GRPO + Agent | 0.533 | 0.571 |
| Exp4: 3B GRPO + Query LoRA | 0.545 | 0.571 |
| Exp5: Adaptive routing | 0.533 | 0.593 |
| **Exp6: 3B GRPO + Naive RAG** | **0.584** | **0.593** |

### 按难度

| 实验 | Easy | Medium | Hard |
|---|---|---|---|
| Exp0 | 0.586 | 0.593 | 0.461 |
| Exp1 | 0.697 | 0.865 | 0.618 |
| Exp2 | 0.616 | 0.696 | 0.472 |
| Exp3 | 0.566 | 0.574 | 0.393 |
| Exp5 | 0.586 | 0.571 | 0.405 |
| **Exp6** | **0.636** | **0.615** | **0.427** |

---

## 核心发现

### 1. BGE 升级让 naive RAG 反超多跳 Agent

**Exp0（naive RAG）EM=0.568 > Exp3（GRPO + Agent）EM=0.540。**

MiniLM 时代，multi-hop agent 比 naive RAG 高 +11.6pp（0.574 vs 0.458）。切换到 BGE 后，排名逆转：naive RAG 反而更高。

根本原因：ctx_recall 从 0.748 提升到 0.907，**BGE 直接用原始问题就能可靠地找到 gold passage，多跳分解的检索增益消失了**，但分解带来的错误传播代价依然存在。

### 2. GRPO adapter 本身有效，但多跳 pipeline 有害

Exp6（GRPO + naive RAG）是 BGE 时代的最优组合（EM=0.586），分解贡献如下：

```
GRPO adapter 贡献：  Exp0 → Exp6  = +1.8pp  ✓ 正向
Multi-hop pipeline：  Exp6 → Exp3  = −4.6pp  ✗ 负向（BGE 时代）
```

GRPO 训练的价值得到验证（答案更简洁精准，EM 近失配减少），但多跳 pipeline 在强检索器面前得不偿失。

### 3. Query LoRA 改善了分解质量，但瓶颈不在分解

Query LoRA（Exp4）相比 Exp3 各指标均提升（EM +1pp，ctx_recall +2.9pp，sub_q2 entity rate +14pp），验证了 32B 蒸馏的方向有效。

但 Exp4（EM=0.550）仍低于 Exp0（EM=0.568）和 Exp6（EM=0.586）。**这说明 sub_q 质量不是决定性瓶颈**——即使分解质量大幅提升，多跳 pipeline 的误差传播代价依然无法被消除到让它超越单跳。

Query LoRA 在项目故事中的作用：正是"改善了分解质量、但增益只有 +1pp"这个结果，促使我们回头质疑多跳范式本身是否必要，从而发现了真正的最优解。

### 4. 自适应路由（Exp5）局部有效但整体不显著

Comparison 题走 naive RAG EM=0.593，确认了假设（comparison 题两个实体明确，BGE 直接就能检索）。但 Bridge 题被强制走 Agent 后 EM=0.533，比 Exp3 的 bridge EM（0.533 持平但没有改善），整体没有超过 Exp0。

根本原因：Exp3 中 Agent 内部会对部分 bridge 题误分类为 comparison，有时反而正确——Exp5 强制路由消除了这个"意外收益"。

### 5. Exp2 Comparison 题异常（EM=0.264）

32B base 模型 Comparison EM 仅 0.264，低于随机水平。根本原因：32B 未经 GRPO 微调，输出格式不稳定（输出冗长句子而非 yes/no），faithfulness verifier 判失败，触发 web search 覆盖了正确结果。这是 agent pipeline 对模型输出格式的隐式依赖。

---

## Exp0 vs Exp4 失败案例分析

对 **Exp0 答对而 Exp4 答错的 54 个案例**进行了人工分类（54 个样本，bridge 45 个，comparison 9 个）。

### 失败模式分布

| 失败模式 | 数量 | 占比 | 说明 |
|---|---|---|---|
| hop1 答案错误（有正确上下文） | 19 | 35% | sub_q2 基于错误实体，ctx_recall=1.00 但链跑偏 |
| EM 近失配 | 12 | 22% | 答案语义正确，多余词/缩写导致 EM=0 |
| hop1 错误 + 上下文缺失 | 11 | 20% | 中间实体错，检索也失败 |
| 路由错误（bridge→comparison） | 4 | 7% | bridge 题被误判走 comparison 路径 |
| 未生成 sub_q | 4 | 7% | 模型输出为空 |
| 召回正确但仍然幻觉 | 2 | 4% | ctx_recall=1.00，最终答案仍然错 |

**关键数据：** 54 个错误案例中 70%（38/54）ctx_recall=1.00——正确段落已经检索到，失败是推理错误，不是检索错误。

### 为什么 hop1 会答错（35%的主因）

sub_q1 平均词数 **11.1 词**，原始问题平均 **21.2 词**——分解时丢失了约一半的约束信息：

```
原始问题: "Which Academy Award-winning actress starred with Alec Baldwin
          in both the Broadway revival and TV movie of A Streetcar Named Desire?"

sub_q1:  "Who starred in Streetcar Named Desire with Alec Baldwin?"
                ↑ "Academy Award-winning" 消失——正是区分候选答案的关键约束

hop1:    Timothy Carhart（配角）  ← 错误实体
gold:    Jessica Lange
```

分解将多约束问题简化为单约束查询，反而使模型失去了原本可以利用的判别信息。

---

## Exp4 Bridge 失败案例深度分析

对 **Exp2（32B）答对、Exp3（3B GRPO）答错的 95 个 bridge 题**进行 sub_q 级别分析。

### 失败分类

| 类别 | 数量 | 占比 | 根本原因 |
|---|---|---|---|
| A：hop1_answer 就错了 | 63 | 66% | sub_q1 质量差，第一跳方向偏移 |
| B：hop1 对，sub_q2 偏 | 20 | 21% | hop1_answer 没有被正确代入 sub_q2 |
| C：query 相同，答案错 | 12 | 13% | 纯生成问题，与检索无关 |

### Category A（66%）— sub_q1 写错了

**模式 A1：few-shot 例子内容混入 sub_q1**
```
Q: Party Never Ends is an album by the Romanian singer who studied at what college?
Gold: Ovidius University

Exp3 sub_q1: What college did the Romanian singer who starred in Pretty Woman study at?
             ↑ "Pretty Woman" 来自 decompose prompt 的 few-shot 例子，被 3B 模型混入
Exp3 hop1  : Willamette University（直接答成了大学，跳过中间实体）
```

**模式 A2：识别到错误的中间实体**
```
Q: The army officer who committed a murder in 1970 at Fort Bragg was born which year?
Gold: 1943

Exp2 hop1: Jeffrey R. MacDonald → pred: 1943 ✓
Exp3 hop1: Ronald Adrin Gray   → pred: 1945 ✗
```

**模式 A3：sub_q1 加了多余限定词，检索跑偏**
```
Q: Black Holes in the Sand features a cover-version of Diane, by what American rock band?
Gold: Hüsker Dü

Exp2 sub_q1: What American rock band performed the song Diane?  → Hüsker Dü ✓
Exp3 sub_q1: What American rock band covered Diane for the song Black Holes in the Sand?
             ↑ 多余限定词，检索到做 cover 的乐队
Exp3 hop1  : Gravenhurst ✗
```

### Category B（21%）— hop1 对，sub_q2 没用好

**模式 B1：sub_q2 没有代入 hop1_answer，重跑第一跳**
```
Q: Who is the mother of the striker for the Czech First League club born on 25th June 1983?
Gold: Eva Janko

hop1_answer: Marc Janko（两模型一致）

Exp2 sub_q2: Who is the mother of Marc Janko? → Eva Janko ✓
Exp3 sub_q2: Who is the mother of the player born on 25th June 1983 who plays for...?
             ↑ 完全没用 hop1_answer，第二跳等于重跑第一跳
Exp3 pred  : Marc Janko ✗
```

**模式 B2：sub_q2 问错了目标属性**
```
Q: The actor that played Alberto in Scarface also starred with Jason Patrick
   in a 2015 horror film written by who?
Gold: Ido Fluk

hop1_answer: Mark Margolis（两模型一致）
Exp2 sub_q2: Who wrote the 2015 horror film that Mark Margolis starred in? → Ido Fluk ✓
Exp3 sub_q2: What films did Mark Margolis star in with Jason Patrick in 2015? ← 问 films 而非 writer
Exp3 pred  : Mark Margolis ✗
```

### Category C（13%）— 纯生成失败

```
Q: When was the father of Sarah Coburn born?
Gold: March 14, 1948
Exp3 pred: "Tom Coburn was born on March 14, 1948."  ← 完整句子，EM=0
```

---

## Exp4 Query LoRA 实验

### 实验设计

在 Exp3 基础上新增 Query LoRA adapter，专门训练 sub_q 生成步骤（decompose、formulate_hop2、rewrite_comparison）。

**训练数据**（`training/query_lora/01_generate_data.py`）：
- 从 HotpotQA train split 采样，32B teacher（vLLM）生成 sub_q 标注
- 质量过滤：sub_q ctx_recall ≥ 0.5，sub_q2 必须包含 hop1_answer 实体
- 最终数据：train 22,548 / val 1,186（sub_q1:10832, sub_q2:8894, comparison_rewrite:2822）
- 数据集：`Norm11/qwen2.5-3b-querylora-hotpotqa-dataset`

**训练配置**：LoRA rank=16, alpha=32，SFTTrainer，3 epochs，最终 eval loss=0.2317

**推理**：sub_q 生成时 `set_adapter("query")`，answer_final 时恢复 `set_adapter("grpo")`

### 结果

| 指标 | Exp3 Baseline | Exp4 Query LoRA | Delta |
|---|---|---|---|
| Exact Match | 0.540 | **0.550** | +1.0pp |
| Token F1 | 0.625 | **0.639** | +1.4pp |
| Context Recall | 0.902 | **0.931** | +2.9pp |
| sub_q2 entity rate | 0.775 | **0.915** | **+14.0pp** |

sub_q2 entity rate 提升 14pp 是最直接的信号，说明 Query LoRA 显著减少了 B 类错误（sub_q2 没有代入 hop1_answer）。

### 结论与定位

Query LoRA 改善了它被设计来改善的东西（sub_q 质量），工程上是成功的。但 +1pp 的 EM 提升（而非预期的 +3~5pp）是重要信号：**瓶颈不在 sub_q 质量**。这一结果直接触发了对多跳范式本身的质疑，最终发现了 Exp6 的最优解。

---

## 总结

| 阶段 | 最优系统 | EM | 核心发现 |
|---|---|---|---|
| MiniLM | 3B GRPO + Agent | 0.574 | 多跳分解 +10.6pp，检索是瓶颈 |
| BGE | **3B GRPO + Naive RAG (Exp6)** | **0.586** | GRPO +1.8pp，多跳 −4.6pp，答案质量是关键 |

**最终结论：** 多跳 agent 解决的是检索问题，不是推理问题。BGE 解决了检索问题之后，最优系统变成了"强检索器 + GRPO 答案模型 + 单跳"——比复杂的多跳 pipeline 更简单也更好。
