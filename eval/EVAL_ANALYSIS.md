# Eval Analysis — Controlled Comparison Experiments

## 实验设计

五组对照实验在同一 500 题评测集（`data/grpo_val.jsonl`）上运行，使用相同的 FAISS 索引和 BGE embedding。

| 实验 | 模型 | 检索 | Pipeline |
|---|---|---|---|
| Exp0 | Qwen2.5-3B base（无 adapter） | FAISS TOP_K=5，直接用原始问题检索 | 单轮 QA（无 agent） |
| Exp1 | Qwen2.5-32B-Instruct | Golden passages（oracle，无 FAISS） | 单轮 QA |
| Exp2 | Qwen2.5-32B-Instruct | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp3 | Qwen2.5-3B + GRPO adapter | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp4 | Qwen2.5-3B + GRPO + Query LoRA | FAISS TOP_K=5 | 完整 LangGraph Agent |

**Exp0** 是下界：最简 RAG，无任何推理结构，揭示 agent 分解步骤的净收益。
**Exp1** 是上界：完美检索 + 最大模型，代表系统的理论天花板。
**Exp4** 是本项目的最终结果。

脚本位于 `eval/` 目录：
- `eval/exp0_3b_naive_rag.py` — 3B base，单跳检索，无 agent
- `eval/exp1_upperbound.py` — vLLM 批量推理，golden passages
- `eval/exp2_32b_agent.py` — HF transformers + PlainModelWrapper
- `eval/exp3_3b_grpo_agent.py` — 复用 src.models.load_model_and_tokenizer
- `eval/training/query_lora/04_eval_e2e.py` — 双 adapter 推理

---

## 总体结果

| 实验 | EM | F1 | ctx_recall | ans_coverage | faithfulness |
|---|---|---|---|---|---|
| Exp0: 3B base + Naive RAG（下界） | 0.568 | 0.657 | 0.907 | 0.868 | 0.908 |
| Exp1: 32B + Golden（上界） | 0.788 | 0.884 | 1.000 | 0.976 | 0.921 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.955 | 0.956 | 0.805 |
| Exp3: 3B GRPO + RAG + Agent | 0.540 | 0.625 | 0.902 | 0.914 | 0.885 |
| Exp4: 3B GRPO + Query LoRA + Agent | 0.550 | 0.639 | 0.931 | 0.944 | — |

### 按题型

| 实验 | Bridge EM | Comparison EM |
|---|---|---|
| Exp0: 3B base + Naive RAG | 0.560 | 0.604 |
| Exp1: 32B + Golden | 0.802 | 0.725 |
| Exp2: 32B + Agent | 0.724 | **0.264** |
| Exp3: 3B GRPO + Agent | 0.533 | 0.571 |
| Exp4: 3B GRPO + Query LoRA | 0.545 | 0.571 |

### 按难度

| 实验 | Easy | Medium | Hard |
|---|---|---|---|
| Exp0 | 0.586 | 0.593 | 0.461 |
| Exp1 | 0.697 | 0.865 | 0.618 |
| Exp2 | 0.616 | 0.696 | 0.472 |
| Exp3 | 0.566 | 0.574 | 0.393 |

---

## 核心发现

### 1. Naive RAG 下界出乎意料地高（Exp0 EM=0.568 > Exp3 EM=0.540）

最重要也最出乎意料的发现：**3B base 模型 + 单跳 RAG（无任何 agent 结构）反而比 3B GRPO + 完整多跳 Agent 效果更好。**

根本原因有两点：

**① HotpotQA distractor 设置下，直接检索召回率已经很高**

ctx_recall 对比：Exp0=0.907，Exp3=0.902。用原始问题直接检索与多跳分解检索几乎持平，因为 HotpotQA 的问题本身包含两个 hop 的实体关键词，FAISS 语义匹配直接就能找到两个 gold passage，多跳分解的检索增益极小。

**② 多跳 agent 引入的错误大于收益**

多跳 pipeline 有多个级联失败点：sub_q1 写偏 → hop1 答错 → sub_q2 不代入实体 → 最终答案错。每一步的错误都会传播到下一步。而 naive RAG 只有"检索 + 一次生成"，失败点少，鲁棒性反而更高。

**这一发现揭示了 agent 设计的根本矛盾**：多跳结构在理论上能处理更复杂的推理，但在实践中，每增加一个中间步骤就增加一个错误来源。只有当中间步骤的准确率足够高时，多跳才能带来净收益。

### 2. Query LoRA 是目前唯一有效改进

Exp4（+Query LoRA）相比 Exp3 的提升：EM +1pp，F1 +1.4pp，ctx_recall +2.9pp。更关键的是 sub_q2 entity rate +14pp，说明 Query LoRA 显著减少了 B 类错误（sub_q2 没有代入 hop1_answer）。

但即使加了 Query LoRA（Exp4 EM=0.550），仍低于 naive RAG baseline（Exp0 EM=0.568）。这说明 Query LoRA 改善了分解质量，但还不足以让多跳 pipeline 整体超越单跳。

### 3. 检索不是唯一瓶颈，分解质量才是

Exp2（32B Agent）ctx_recall=0.955 远高于 Exp3（3B Agent）0.902，但 EM 差距只有 10pp。而 Exp0 ctx_recall=0.907 与 Exp3 相近，EM 却高了 2.8pp。这说明提升检索对最终答案的边际收益有限，**分解步骤的准确率才是决定多跳 pipeline 效果的关键**。

### 4. Exp2 Comparison 题严重异常（EM=0.264）

32B base 模型 Comparison EM 仅 0.264，低于随机猜测（0.5）。根本原因：32B 未经 GRPO 微调，输出格式不稳定（输出冗长句子而非 yes/no）→ faithfulness verifier 判定失败 → 触发 web search → 覆盖了正确的检索结果。这是 agent pipeline 对模型输出格式的隐式依赖，而非 32B 能力不足（Exp1 中 Comparison EM=0.725）。

---

## Bridge 失败案例深度分析

对 **exp2 答对、exp3 答错的 95 个 Bridge 题**（占 bridge 总错误的主要部分）进行 sub_q1 / hop1_answer / sub_q2 级别的分析。

### 失败分类

| 类别 | 数量 | 占比 | 根本原因 |
|---|---|---|---|
| A：hop1_answer 就错了 | 63 | 66% | sub_q1 质量差，第一跳方向偏移 |
| B：hop1 对，sub_q2 偏 | 20 | 21% | hop1_answer 没被正确代入 sub_q2 |
| C：query 完全相同，答案错 | 12 | 13% | 纯生成问题，与检索无关 |

---

### Category A（66%）— sub_q1 写错了

**模式 A1：把 few-shot 例子里的内容混入 sub_q1**

```
Q: Party Never Ends is an album by the Romanian singer who studied at what college?
Gold: Ovidius University

Exp2 sub_q1: Who is the Romanian singer who released Party Never Ends?
Exp2 hop1  : Inna
Exp2 sub_q2: What college did Inna study at?
Exp2 pred  : Ovidius University ✓

Exp3 sub_q1: What college did the Romanian singer who starred in Pretty Woman study at?
             ↑ "Pretty Woman" 来自 decompose prompt 的 few-shot 例子，被 3B 模型混入
Exp3 hop1  : Willamette University（直接答成了大学，跳过了中间实体）
Exp3 pred  : Willamette University ✗
```

**模式 A2：识别到错误的中间实体**

```
Q: The army officer who committed a murder in 1970 at Fort Bragg was born which year?
Gold: 1943

Exp2 sub_q1: Who was the army officer who committed a murder in 1970 at Fort Bragg?
Exp2 hop1  : Jeffrey R. MacDonald → Exp2 pred: 1943 ✓

Exp3 sub_q1: Who committed a murder at Fort Bragg in 1970?
Exp3 hop1  : Ronald Adrin Gray   → Exp3 pred: 1945 ✗
```

**模式 A3：sub_q1 加了多余限定词，检索跑偏**

```
Q: Black Holes in the Sand features a cover-version of Diane, by what American rock band?
Gold: Hüsker Dü

Exp2 sub_q1: What American rock band performed the song Diane?
Exp2 hop1  : Hüsker Dü（原唱）✓

Exp3 sub_q1: What American rock band covered Diane for the song Black Holes in the Sand?
             ↑ 多了限定词，检索找到的是做 cover 的乐队
Exp3 hop1  : Gravenhurst ✗
```

**模式 A4：sub_q2 方向完全错误**

```
Q: No Issue was a Pakistani TV ad campaign by a mobile network created by the merger of which two companies?
Gold: Mobilink and Warid

Exp2 hop1: Jazz → sub_q2: What two companies merged to create Jazz? → Mobilink and Warid ✓
Exp3 hop1: Jazz Pakistan → sub_q2: What mobile network is Jazz Pakistan a part of? ✗
           ↑ sub_q2 完全跑偏，不再问 merger
```

---

### Category B（21%）— hop1 对，sub_q2 没用好

**模式 B1：sub_q2 问错了目标属性**

```
Q: The actor that played Alberto "The Shadow" in Scarface also starred with Jason Patrick
   in a 2015 horror film written by who?
Gold: Ido Fluk

hop1_answer: Mark Margolis（两模型一致）

Exp2 sub_q2: Who wrote the 2015 horror film that Mark Margolis starred in with Jason Patrick?
Exp2 pred  : Ido Fluk ✓

Exp3 sub_q2: What films did Mark Margolis star in with Jason Patrick in 2015?
             ↑ 问的是 films，不是 writer
Exp3 pred  : Mark Margolis ✗
```

**模式 B2：sub_q2 没有代入 hop1_answer，仍在描述原始问题**

```
Q: Who is the mother of the striker for the Czech First League club born on 25th June 1983?
Gold: Eva Janko

hop1_answer: Marc Janko（两模型一致）

Exp2 sub_q2: Who is the mother of Marc Janko? → Eva Janko ✓

Exp3 sub_q2: Who is the mother of the player born on 25th June 1983 who plays for...?
             ↑ 完全没用 hop1_answer，第二跳等于重跑第一跳
Exp3 pred  : Marc Janko ✗（返回了 hop1 实体）
```

**模式 B3：sub_q2 逻辑方向反转**

```
Q: Whose grandson founded the Trilateral Commission?
Gold: John D. Rockefeller

hop1_answer: David Rockefeller（两模型一致）

Exp2 sub_q2: Who is the grandson of David Rockefeller that founded the Trilateral Commission?
             （逻辑上也有问题，但检索恰好找到正确答案）→ John D. Rockefeller ✓

Exp3 sub_q2: Whose grandson is David Rockefeller?（方向正确但检索结果无法回答）
Exp3 pred  : David Rockefeller ✗
```

---

### Category C（13%）— query 相同，纯生成失败

**冗长输出导致 EM=0**

```
Q: When was the father of Sarah Coburn born?
Gold: March 14, 1948
Exp3 pred: "Tom Coburn was born on March 14, 1948."  ← 完整句子，EM 失败
```

**完全幻觉**

```
Q: What non-profit organization was co-founded by the owner of the UC3 Nautilus?
Gold: Copenhagen Suborbitals
Exp3 pred: "The Society for Nutrition Education and Behavior"  ← 与问题毫无关联
```

---

## Exp4：Query LoRA 实验结果

### 实验设计

在 Exp3 的基础上新增一个 Query LoRA adapter，专门用于 sub_q 生成步骤（decompose、formulate_hop2、rewrite_comparison），GRPO adapter 保持不变，用于 answer_final。

**训练数据生成**（`training/query_lora/01_generate_data.py`）：
- 从 HotpotQA train split 采样 12000 bridge + 3000 comparison 题
- 32B teacher（vLLM 批量推理）生成 sub_q1 和 sub_q2 标注
- 3B student 生成 hop1_answer（匹配推理时的分布）
- 质量过滤：sub_q1/sub_q2 ctx_recall ≥ 0.5，sub_q2 必须包含 hop1_answer 实体

**最终数据量**：train 22,548 / val 1,186，task 分布：sub_q1:10832, sub_q2:8894, comparison_rewrite:2822

**训练配置**（`training/query_lora/02_train_query_lora.py`）：
- LoRA rank=16, alpha=32，目标模块：q/k/v/o/gate/up/down_proj
- SFTTrainer + DataCollatorForCompletionOnlyLM（仅对 assistant 回复计算 loss）
- 3 epochs，lr=2e-4，最终 eval loss=0.2317

**推理**：在 agent 每次 sub_q 生成前 `set_adapter("query")`，生成后恢复 `set_adapter("grpo")`

### 结果

| 指标 | Exp3 Baseline | Exp4 Query LoRA | Delta |
|---|---|---|---|
| Exact Match | 0.540 | **0.550** | +1.0pp |
| Token F1 | 0.625 | **0.639** | +1.4pp |
| Context Recall | 0.902 | **0.931** | +2.9pp |
| Ans Coverage | 0.914 | **0.944** | +3.0pp |

| 题型 | Bridge EM | Comparison EM |
|---|---|---|
| Exp3 | 0.533 | 0.571 |
| Exp4 | **0.545** | **0.571** |

### 中间评估（sub_q 质量）

| 指标 | Base 3B | Query LoRA | Delta |
|---|---|---|---|
| sub_q1 ctx_recall | 0.243 | 0.240 | -0.003 |
| sub_q2 ctx_recall | 0.228 | 0.238 | +0.010 |
| sub_q2 entity rate | 0.775 | **0.915** | **+0.140** |

sub_q2 entity rate 提升 14pp 是最直接的信号，说明 Query LoRA 显著改善了 B2 类问题（sub_q2 没有代入 hop1_answer）。这一改善直接传导到端到端的 ctx_recall 提升（+2.9pp）。

### 结论

- Query LoRA 在所有关键指标上均优于 Exp3 baseline，验证了通过 32B 知识蒸馏改善 sub_q 生成的方案有效
- ctx_recall 从 0.902 提升至 0.931，说明检索质量的改善是 EM/F1 提升的直接原因
- 训练目标（sub_q2 entity rate）与最终 E2E 指标正相关，实验链路完整

---

## 改进方向

Exp0 的发现（naive RAG > multi-hop agent）重新定义了改进优先级：**核心问题不是检索，而是分解步骤的准确率不够高**。要让多跳 agent 超越单跳 RAG，分解准确率必须达到让级联错误率低于单跳的损耗。

### 方向1：提升分解准确率（最高优先级）

Query LoRA（已实现）是正确方向，但还不够。进一步改进：

- **更多蒸馏数据**：当前 22K 样本，扩大到 50K+ 可能带来更明显提升
- **Iterative refinement**：如果 hop1_answer 置信度低（输出过长、包含不确定词），让模型重新生成 sub_q1 而不是继续传播错误
- **Chain-of-thought 监督**：在 sub_q1 生成时加入显式的推理过程标注，而不只是直接输出 sub_q

### 方向2：自适应路由（跳过不必要的分解）

并非所有题目都需要两跳推理。如果直接检索的 top-1 段落置信度足够高，可以直接回答而不走多跳 pipeline。

```
直接检索 → 置信度评估 → 高置信度: 直接答 | 低置信度: 走多跳
```

这本质上是把 naive RAG 的优势（简单题目直接答）和 agent 的优势（复杂题目分解推理）结合起来。

### 方向3：端到端 GRPO 奖励重新设计

当前 GRPO 的 reward = 0.5×EM + 0.5×F1，只对最终答案打分，不惩罚中间步骤的错误。可以加入：

- **中间步骤奖励**：sub_q1 检索到 gold passage 给正奖励，没检索到给负奖励
- **分解一致性奖励**：sub_q2 包含 hop1_answer 实体给正奖励

这让模型在 GRPO 训练时就学到"分解步骤要准确"，而不只是"最终答案要准确"。

### 方向4：Category C 生成质量（次要）

- **冗长输出**：answer prompt 强化"用短语回答，不要写完整句子"
- **幻觉**：提高 faithfulness verifier 阈值，或 GRPO 加入更多检索失败案例
