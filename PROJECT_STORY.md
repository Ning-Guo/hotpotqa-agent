# HotpotQA Agent — 项目迭代故事线

本文档按时间顺序记录项目的完整改进历程：每一步的动机、策略、实验结果和具体失败案例。

---

## 全局实验结果一览

| 阶段 | 系统配置 | EM | F1 | ctx_recall |
|---|---|---|---|---|
| Phase 1 | 3B base + MiniLM + naive RAG | 0.458 | 0.541 | 0.748 |
| Phase 2 | 3B GRPO + MiniLM + multi-hop agent | 0.574 | 0.660 | 0.887 |
| Phase 3A | 3B base + **BGE** + naive RAG（下界重测） | 0.568 | 0.657 | 0.907 |
| Phase 3B | 3B GRPO + BGE + multi-hop agent | 0.540 | 0.625 | 0.902 |
| Phase 3C | 32B base + BGE + multi-hop agent | 0.640 | 0.709 | 0.955 |
| Phase 3D | 32B + golden passages（上界） | 0.788 | 0.884 | 1.000 |
| Phase 4 | 3B GRPO + BGE + **Query LoRA** + agent | 0.550 | 0.639 | 0.931 |

---

## Phase 1：起点 — 3B Base + MiniLM + Naive RAG

### 配置

- 模型：`Qwen2.5-3B-Instruct`（无微调，无 adapter）
- Embedding：`all-MiniLM-L6-v2`（通用句子 embedding，非检索专用）
- Pipeline：单跳检索 → 直接生成答案
- TOP_K：3

### 结果

| 指标 | 数值 |
|---|---|
| Exact Match | 0.458 |
| Token F1 | 0.541 |
| Context Recall | 0.748 |

### 核心问题

ctx_recall 只有 **0.748**，说明将近 1/4 的题目根本找不到 gold passage。即使模型能力足够，没有正确的上下文也无法回答。

**典型失败案例（检索失败类型）：**

```
Q: What nationality is the director of the film which has a song called "Telephone Line"?
Gold: American

检索词（原始问题）→ 返回了关于电话线路的段落，而非 ELO 乐队和相关电影
→ 模型根据无关段落猜测：British ✗

根本原因：MiniLM 语义理解弱，"Telephone Line" 被理解为字面意思，
而不是 ELO 乐队的歌曲名。正确段落根本没有进入 top-3。
```

**判断**：检索质量是当前最大瓶颈。两条并行改进方向：
1. 架构层面：通过多跳分解，把复杂问题拆成 embedding 更容易匹配的简单问题
2. 检索层面：升级 embedding 模型本身

---

## Phase 2：多跳 Agent + GRPO 训练

### 改进策略

**策略A：LangGraph 多跳 Agent**

Bridge 题的原始问题往往包含两层语义，MiniLM 无法同时匹配两个 gold passage。拆分之后：
- sub_q1 只问中间实体 → embedding 更容易匹配第一个 gold passage
- sub_q2 把 hop1_answer 代入 → embedding 更容易匹配第二个 gold passage

```
原始问题：What nationality is the actress who starred in Pretty Woman?
→ sub_q1: Who starred in Pretty Woman?      → 检索 Pretty Woman 相关段落 ✓
→ sub_q2: What nationality is Julia Roberts? → 检索 Julia Roberts 相关段落 ✓
```

**策略B：GRPO 微调**

- 奖励函数：reward = 0.5 × EM + 0.5 × F1
- 训练数据：30K 样本，基于 HotpotQA train split
- 目标：让 3B 模型在已有上下文的情况下更准确地提取和生成答案

### 结果

| 指标 | Phase 1 | Phase 2 | Delta |
|---|---|---|---|
| Exact Match | 0.458 | **0.574** | +11.6pp |
| Token F1 | 0.541 | **0.660** | +11.9pp |
| Context Recall | 0.748 | **0.887** | +13.9pp |

| 题型 | EM |
|---|---|
| Bridge | 0.570 |
| Comparison | 0.593 |

**多跳 agent 有效**：通过把复杂问题拆成简单子问题，在 MiniLM embedding 下 ctx_recall 从 0.748 提升到 0.887，最终 EM +11.6pp。

### 仍存在的失败案例

即使架构升级了，3B base 模型（decompose 步骤未微调）的 sub_q 生成仍然不稳定：

**问题：sub_q1 混入 few-shot 例子内容**
```
Q: Party Never Ends is an album by the Romanian singer who studied at what college?
Gold: Ovidius University

sub_q1: What college did the Romanian singer who starred in Pretty Woman study at?
        ↑ "Pretty Woman" 来自 prompt 里的 few-shot 例子，3B 模型混入了
hop1: Willamette University（直接答成了大学，跳过中间实体）
pred: Willamette University ✗
```

**问题：sub_q2 没有代入 hop1_answer**
```
Q: Who is the mother of the striker for Czech First League club born on 25th June 1983?
Gold: Eva Janko
hop1: Marc Janko ✓

sub_q2: Who is the mother of the player born on 25th June 1983 who plays for...?
        ↑ 完全没用 hop1_answer，第二跳等于重跑第一跳
pred: Marc Janko ✗（返回了 hop1 实体）
```

---

## Phase 3：Embedding 升级 — BGE-base-en-v1.5

### 改进策略

将 embedding 从通用模型 `all-MiniLM-L6-v2` 升级为检索专用模型 `BAAI/bge-base-en-v1.5`，并加入 BGE 专属 query prefix：

```python
# retriever.py
self._query_prefix = (
    "Represent this sentence for searching relevant passages: "
    if embedding_model.startswith("BAAI/bge")
    else ""
)
```

BGE query prefix 是 BGE 模型的设计要求，能显著提升检索准确率。这是一个主动的、基于模型文档的工程决策。

### 关键发现：Embedding 升级改变了系统格局

升级后用相同的 naive RAG pipeline 重测（Phase 3A）：

| 配置 | EM | ctx_recall |
|---|---|---|
| Phase 1（MiniLM + naive RAG） | 0.458 | 0.748 |
| Phase 3A（BGE + naive RAG） | **0.568** | **0.907** |

ctx_recall 从 0.748 → 0.907，**naive RAG 的效果从 0.458 跳到了 0.568**，几乎追平了 Phase 2 的多跳 agent（0.574）。

### 对照实验（全部在 BGE 环境下）

为了准确评估各组件的贡献，做了完整的对照实验：

| 实验 | 模型 | Pipeline | EM | ctx_recall |
|---|---|---|---|---|
| Exp0（下界） | 3B base | naive RAG | 0.568 | 0.907 |
| Exp3 | 3B GRPO | multi-hop agent | 0.540 | 0.902 |
| Exp2 | 32B base | multi-hop agent | 0.640 | 0.955 |
| Exp1（上界） | 32B base | golden passages | 0.788 | 1.000 |

**最重要的发现：BGE 环境下，3B GRPO + multi-hop agent（0.540）反而低于 3B base + naive RAG（0.568）。**

### 为什么多跳 agent 在 BGE 环境下失效？

| 问题 | 解释 |
|---|---|
| 检索增益消失 | MiniLM 时代，多跳分解让 ctx_recall 从 0.748→0.887（+13.9pp）。BGE 时代，直接检索已达 0.907，多跳只有 0.902，几乎没有增益 |
| 分解错误仍然存在 | Phase 2 中发现的 sub_q1 偏移、sub_q2 不代入 hop1_answer 等问题依然存在 |
| 净收益为负 | 检索增益≈0，但分解错误带来的损耗依然存在 → 多跳整体拖累了结果 |

**深度失败分析**（Exp2 答对、Exp3 答错的 95 个 Bridge 题）：

| 类别 | 占比 | 根本原因 |
|---|---|---|
| A：sub_q1 写错，第一跳就偏了 | 66% | 3B 模型分解质量不足 |
| B：hop1 对，sub_q2 没代入 hop1_answer | 21% | sub_q2 生成的系统性缺陷 |
| C：检索正确，纯生成问题 | 13% | 输出冗长 / 幻觉 |

**Category A 具体案例：**
```
Q: The army officer who committed a murder in 1970 at Fort Bragg was born which year?
Gold: 1943

32B sub_q1: Who was the army officer who committed a murder in 1970 at Fort Bragg?
32B hop1  : Jeffrey R. MacDonald → pred: 1943 ✓

3B  sub_q1: Who committed a murder at Fort Bragg in 1970?（丢失关键限定"army officer"）
3B  hop1  : Ronald Adrin Gray   → pred: 1945 ✗
```

```
Q: Black Holes in the Sand features a cover-version of Diane, by what American rock band?
Gold: Hüsker Dü

32B sub_q1: What American rock band performed the song Diane?
32B hop1  : Hüsker Dü（原唱）✓

3B  sub_q1: What American rock band covered Diane for the song Black Holes in the Sand?
            ↑ 多了"covered"限定，检索找到的是做 cover 的乐队，而非原唱
3B  hop1  : Gravenhurst ✗
```

**Category B 具体案例：**
```
Q: Whose grandson founded the Trilateral Commission?
Gold: John D. Rockefeller
hop1_answer: David Rockefeller（两模型一致）

32B sub_q2: Who is the grandson of David Rockefeller that founded the Trilateral Commission?
32B pred  : John D. Rockefeller ✓

3B  sub_q2: Whose grandson is David Rockefeller?（逻辑方向反转）
3B  pred  : David Rockefeller ✗
```

**特殊异常：Exp2 Comparison EM 仅 0.264（低于随机猜测 0.5）**
```
原因：32B base 未经 GRPO 微调，yes/no 答案格式不稳定
→ faithfulness verifier 判定失败 → 触发 web search → 覆盖正确检索结果
→ Comparison 答案变差

注：Exp1 中 32B golden passages Comparison EM=0.725，说明能力没问题，
是 pipeline 对模型输出格式的隐式依赖导致了这个问题。
```

### 小结

BGE 升级解决了 Phase 1 的检索瓶颈，但同时也让多跳 agent 的优势消失。**新的瓶颈是分解准确率**：3B 模型生成的 sub_q1 和 sub_q2 质量不够高，抵消了多跳结构的理论收益。

---

## Phase 4：Query LoRA — 针对分解质量的知识蒸馏

### 改进策略

用 32B 模型作为 teacher，生成高质量的 sub_q1 / sub_q2 标注数据，SFT 训练一个专门用于 query 生成的 LoRA adapter（与 GRPO adapter 并行，通过 PEFT 多 adapter 切换）。

**数据生成流程：**
```
Bridge 题处理：
  32B teacher → sub_q1（批量 vLLM）
  FAISS 检索 sub_q1 对应 passages
  3B student → hop1_answer（匹配推理分布，不用 32B）
  32B teacher → sub_q2（输入 hop1_answer）
  质量过滤：ctx_recall ≥ 0.5 + sub_q2 必须包含 hop1_answer 实体

Comparison 题处理：
  32B teacher → 两个 sub-query
  质量过滤：union ctx_recall = 1.0
```

**关键设计：hop1_answer 必须用 3B 生成，而不是 32B。** 因为推理时用的是 3B，训练数据的 hop1_answer 分布必须与推理时一致，否则 sub_q2 的训练目标和实际输入分布不匹配。

**数据规模：** train 22,548 / val 1,186
**训练：** LoRA rank=16，3 epochs，eval loss=0.2317

**推理时的 adapter 切换：**
```python
# sub_q 生成步骤：激活 Query LoRA
model.set_adapter("query")
sub_q = model.generate(...)

# answer_final 步骤：恢复 GRPO adapter
model.set_adapter("grpo")
answer = model.generate(...)
```

### 结果

| 指标 | Exp3（基线） | Exp4（+Query LoRA） | Delta |
|---|---|---|---|
| Exact Match | 0.540 | **0.550** | +1.0pp |
| Token F1 | 0.625 | **0.639** | +1.4pp |
| Context Recall | 0.902 | **0.931** | +2.9pp |
| sub_q2 entity rate | 0.775 | **0.915** | +14.0pp |

sub_q2 entity rate +14pp 是最直接的验证：Query LoRA 显著改善了 B 类失败（sub_q2 没有代入 hop1_answer），这一改善传导到了 ctx_recall 的提升。

**但 Exp4（0.550）仍低于 naive RAG 下界（0.568）**，说明分解准确率有改善，但还不足以让多跳整体超越单跳。

---

## 整体分析与反思

### 完整故事线

```
Phase 1: MiniLM + naive RAG → EM=0.458
  瓶颈：检索质量差（ctx_recall=0.748），1/4 题目根本找不到 gold passage

Phase 2: 多跳 Agent + GRPO → EM=0.574（+11.6pp）
  解法：把复杂问题拆成简单子问题，让弱 embedding 也能匹配
  新瓶颈：3B 模型的分解质量不够高（sub_q1 偏移、sub_q2 不代入实体）

Phase 3: BGE Embedding 升级 → naive RAG 达到 EM=0.568
  结果：直接检索 ctx_recall 从 0.748→0.907，
        multi-hop 的检索优势消失，分解错误成为净损耗
  新瓶颈：在高质量 embedding 下，分解准确率是决定因素

Phase 4: Query LoRA 蒸馏 → EM=0.550，ctx_recall=0.931
  解法：32B teacher 生成高质量 sub_q 标注，SFT 3B 的分解能力
  结果：分解质量有改善，ctx_recall 超过 naive RAG（0.931 vs 0.907），
        但 EM 仍略低（0.550 vs 0.568），分解误差尚未完全消除
```

### 核心洞察

**1. 组件升级会改变系统瓶颈**

embedding 从 MiniLM 升级到 BGE，不仅提升了检索质量，也改变了整个系统的瓶颈位置。原来"多跳分解帮助弱 embedding"的逻辑，在强 embedding 下变成了"多跳分解引入额外错误"。这说明 ML 系统的各组件之间存在深度耦合，单一组件的提升可能使其他组件的价值发生变化。

**2. 架构复杂性需要匹配的组件能力**

多跳 agent 的理论优势（结构化推理）在实践中能否发挥，取决于每个中间步骤的准确率。如果分解步骤的错误率高于单跳的损耗率，复杂架构反而有害。只有当分解准确率足够高时，多跳才能带来净收益。

**3. 训练数据分布必须匹配推理分布**

Query LoRA 训练时 hop1_answer 用 3B 生成（而非 32B），是一个关键设计决策。如果用 32B 生成 hop1_answer，训练时的 sub_q2 输入分布和推理时不一致，模型会学到一个错误的映射关系。

**4. ctx_recall 与 EM 的解耦**

Exp4 的 ctx_recall（0.931）已超过 naive RAG（0.907），但 EM（0.550）仍低于 naive RAG（0.568）。这说明检索质量的提升不能自动转化为答案质量的提升——分解错误在检索之后仍然会影响答案生成。

### 后续改进方向

1. **自适应路由**：简单题直接用 naive RAG 回答，只有复杂题才走多跳。本质上是结合两者的优势。
2. **更多蒸馏数据**：Query LoRA 当前 22K 样本，扩大可能进一步提升分解准确率。
3. **中间步骤奖励**：在 GRPO 训练中加入 sub_q 检索成功的奖励信号，让模型在训练时就学到"分解要准确"。
4. **在 BGE 环境下重新做 GRPO 训练**：当前 GRPO 模型是在 MiniLM 检索分布下训练的，与 BGE 环境存在分布偏移。
