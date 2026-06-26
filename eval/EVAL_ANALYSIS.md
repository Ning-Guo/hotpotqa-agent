# Eval Analysis — Controlled Comparison Experiments

## 实验设计

为了验证项目结论的说服力，设计了三组对照实验，在同一 500 题评测集（`data/grpo_val.jsonl`）上运行，使用相同的 FAISS 索引和 BGE embedding。

| 实验 | 模型 | 检索 | Pipeline |
|---|---|---|---|
| Exp1 | Qwen2.5-32B-Instruct | Golden passages（直接用 supporting facts，无 FAISS） | 单轮 QA |
| Exp2 | Qwen2.5-32B-Instruct | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp3 | Qwen2.5-3B + GRPO adapter | FAISS TOP_K=5 | 完整 LangGraph Agent |

**Exp1** 是联合上界：完美检索 + 最大模型，代表系统的理论天花板。
**Exp2** 是大模型 baseline：相同 agent 架构，不做微调。
**Exp3** 是本项目的主要结果。

脚本位于 `eval/` 目录，不改动原有代码：
- `eval/exp1_upperbound.py` — vLLM 批量推理，golden passages
- `eval/exp2_32b_agent.py` — HF transformers + PlainModelWrapper（no-op disable_adapter）
- `eval/exp3_3b_grpo_agent.py` — 复用 src.models.load_model_and_tokenizer

---

## 总体结果

| 实验 | EM | F1 | ctx_recall | ans_coverage | faithfulness |
|---|---|---|---|---|---|
| Exp1: 32B + Golden（上界） | 0.788 | 0.884 | 1.000 | 0.976 | 0.921 |
| Exp2: 32B + RAG + Agent | 0.640 | 0.709 | 0.955 | 0.956 | 0.805 |
| Exp3: 3B GRPO + RAG + Agent | 0.540 | 0.625 | 0.902 | 0.914 | 0.885 |

### 按题型

| 实验 | Bridge EM | Comparison EM |
|---|---|---|
| Exp1: 32B + Golden | 0.802 | 0.725 |
| Exp2: 32B + Agent | 0.724 | **0.264** |
| Exp3: 3B GRPO + Agent | 0.533 | 0.571 |

### 按难度

| 实验 | Easy | Medium | Hard |
|---|---|---|---|
| Exp1 | 0.697 | 0.865 | 0.618 |
| Exp2 | 0.616 | 0.696 | 0.472 |
| Exp3 | 0.566 | 0.574 | 0.393 |

---

## 核心发现

### 1. 检索不是唯一瓶颈

Exp2 的 ctx_recall=0.955（已接近完美），但 EM 仍比 Exp1 低 14.8pp。说明即使检索近乎完美，agent pipeline 本身（hop 分解错误、hop1 答案传播错误等）仍会引入显著损耗。单纯提高 TOP_K 效果有限，query 质量才是核心瓶颈。

### 2. 3B GRPO 与 32B Agent 的差距只有 10pp

Exp3 EM=0.540，Exp2 EM=0.640，差距 10pp。参数量相差 10 倍，GRPO 微调 + agent 设计基本弥补了模型规模的劣势。

### 3. Exp2 Comparison 题严重异常（EM=0.264）

32B base 模型在 Comparison 题上表现比 3B GRPO（0.571）差得多，甚至低于随机猜测（0.5）。

**根本原因**：Comparison 答案为 yes/no，pipeline 的 faithfulness verifier 对 yes/no 做了特殊处理（检查实体覆盖率）。32B base 模型没有经过 GRPO 微调，answer 格式不稳定，有时输出冗长句子而非干净的 yes/no，verifier 判定失败 → 触发 web search → 用不相关段落替换原本正确的检索结果，答案变差。这是 agent pipeline 对模型输出格式的隐式依赖，而非 32B 模型能力不足（Exp1 中 32B Comparison EM=0.725）。

### 4. ctx_recall 差异印证 agent 的检索价值

Exp2（32B）ctx_recall=0.955，Exp3（3B）ctx_recall=0.902。32B 生成的 sub-query 质量更高，召回 gold 段落的能力更强。但最终答案质量差距只有 10pp，说明 GRPO 微调帮助 3B 模型更好地利用了已有上下文，一定程度上弥补了检索质量的差距。

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

## 改进方向

### 针对 Category A + B（87%）— 32B 蒸馏 SFT

用 32B 模型对 `decompose_bridge`（生成 sub_q1）、`formulate_hop2`（生成 sub_q2）、`rewrite_comparison` 三步生成高质量标注数据，SFT 3B 模型的 query 改写能力。

**关键约束**：
- 给 32B 的 prompt 必须与 `src/reasoner.py` 里的模板完全一致，避免分布偏移
- 生成时使用 FAISS 检索结果（而非 golden passages）作为输入，与推理环境对齐
- 目标：让 3B 学会"每一跳只问一个干净的问题，sub_q2 必须把 hop1_answer 代入"

可新增一个独立的 Query LoRA adapter，通过 PEFT 多 adapter 切换使用，不动现有 GRPO adapter。

### 针对 Category C（13%）— 生成质量

- **冗长输出**：GRPO 奖励加 brevity 惩罚，或 answer prompt 强化"用短语回答，不要写完整句子"
- **幻觉**：提高 faithfulness verifier 阈值，或在 GRPO 训练时加入更多检索失败案例
