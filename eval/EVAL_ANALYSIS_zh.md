# Eval Analysis — Two-Phase Experiment History

实验分为两个阶段，对应两个不同的 embedding 方案。两个阶段使用完全相同的模型、pipeline 和评测集（500 题），唯一变量是检索器。所有结果均为本次重新完整跑出的干净数据。

---

## Phase 1 — MiniLM Embedding (`all-MiniLM-L6-v2`)

### 背景

项目最初使用 `all-MiniLM-L6-v2` 作为 embedding 模型。HotpotQA distractor 设置下，直接用原始问题检索的 ctx_recall 约为 0.748——将近 1/4 的题目 gold passage 根本不在 top-5 里，检索质量是主要瓶颈。

### 实验结果

| 系统 | EM | F1 | ctx_recall | Bridge EM | Comparison EM |
|---|---|---|---|---|---|
| 3B Base + Naive RAG（下界） | 0.438 | 0.539 | 0.748 | 0.406 | 0.582 |
| 3B GRPO + Naive RAG | 0.456 | 0.541 | 0.748 | 0.418 | 0.626 |
| **3B GRPO + Agent（最优）** | **0.480** | **0.567** | **0.804** | **0.455** | **0.593** |

### 关键发现

**多跳分解有效（+4.2pp EM，ctx_recall +5.6pp）。** 在 MiniLM 弱检索器下，将原始问题拆解为 sub_q1 → hop1 → sub_q2，每个子查询更精准，把 ctx_recall 从 0.748 提升到 0.804，最终 EM 从 0.438 → 0.480。

**GRPO adapter 贡献稳定（+1.8pp）。** 相同检索下，GRPO 训练让答案更简洁精准。

### Case：多跳分解解决了检索失败

MiniLM 下直接检索经常找不到 gold passage：

```
Q: What famed director, actor and humanitarian once starred alongside
   both Johnny Depp and Faye Dunaway?
Gold: Jerry Lewis
MiniLM naive RAG: ctx_recall=0.0 → Pred: Barbet Schroeder  ✗
（gold passage 完全未进入 top-5）
```

而多跳分解通过拆解子问题，让 embedding 只需要匹配更简单的语义单元，从而找到正确段落：

```
Q: In what war did the man who narrated The Great American West fight?
Gold: World War II

sub_q1: Who narrated The Great American West?
hop1:   Jason Robards  ✓
sub_q2: In what war did Jason Robards fight?
Pred:   World War II   ✓  (ctx_recall 从 0 → 1.0)
```

---

## Phase 2 — BGE Embedding (`BAAI/bge-base-en-v1.5`)

### 背景

将 embedding 升级为 `BAAI/bge-base-en-v1.5` 并加入 BGE 专属 query prefix 后，ctx_recall 从 0.748 提升至 0.907（+15.9pp）。检索瓶颈基本消除，这根本性地改变了各组件的相对贡献。

在此阶段进行了 5 组受控实验，所有实验使用相同的 FAISS 索引和 500 题评测集。

### 实验设计

| 实验 | 模型 | 检索策略 | Pipeline |
|---|---|---|---|
| Exp0 | Qwen2.5-3B base（无 adapter） | 原始问题单跳 | 单轮 QA |
| Exp1 | Qwen2.5-32B-Instruct | Golden passages（oracle） | 单轮 QA（上界） |
| Exp2 | Qwen2.5-32B-Instruct | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp3 | Qwen2.5-3B + GRPO adapter | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp4 | Qwen2.5-3B + GRPO + Query LoRA | FAISS TOP_K=5 | 完整 LangGraph Agent |
| Exp5 | Qwen2.5-3B + GRPO adapter | comparison→单跳，bridge→Agent | 自适应路由 |
| Exp6 | Qwen2.5-3B + GRPO adapter | 原始问题单跳 | 单轮 QA |

### 总体结果

| 实验 | EM | F1 | ctx_recall | ans_coverage | faithfulness |
|---|---|---|---|---|---|
| Exp0: 3B base + Naive RAG | 0.572 | 0.659 | 0.907 | 0.868 | 0.908 |
| Exp1: 32B + Golden（上界） | 0.786 | 0.884 | 1.000 | 0.976 | 0.921 |
| Exp2: 32B + RAG + Agent | 0.644 | 0.714 | 0.949 | 0.954 | 0.806 |
| Exp3: 3B GRPO + Agent | 0.538 | 0.625 | 0.901 | 0.914 | 0.889 |
| Exp4: 3B GRPO + Query LoRA + Agent | 0.550 | 0.639 | 0.931 | 0.942 | 0.893 |
| Exp5: Adaptive routing | 0.538 | 0.633 | 0.908 | 0.912 | 0.899 |
| **Exp6: 3B GRPO + Naive RAG** | **0.592** | **0.670** | **0.907** | **0.868** | **0.904** |

### 按题型

| 实验 | Bridge EM | Comparison EM |
|---|---|---|
| Exp0: 3B base + Naive RAG | 0.565 | 0.604 |
| Exp1: 32B + Golden | 0.802 | 0.714 |
| Exp2: 32B + Agent | 0.729 | **0.264** |
| Exp3: 3B GRPO + Agent | 0.531 | 0.571 |
| Exp4: 3B GRPO + Query LoRA | 0.545 | 0.571 |
| Exp5: Adaptive routing | 0.528 | 0.582 |
| **Exp6: 3B GRPO + Naive RAG** | **0.592** | **0.593** |

### 按难度

| 实验 | Easy | Medium | Hard |
|---|---|---|---|
| Exp0 | 0.626 | 0.590 | 0.449 |
| Exp1 | 0.687 | 0.869 | 0.607 |
| Exp2 | 0.616 | 0.702 | 0.472 |
| Exp3 | 0.566 | 0.571 | 0.393 |
| Exp5 | 0.576 | 0.574 | 0.371 |
| **Exp6** | **0.647** | **0.622** | **0.427** |

---

## 核心发现

### 1. BGE 升级消除检索瓶颈，让 Naive RAG 反超多跳 Agent

**Exp0（naive RAG）EM=0.572 > Exp3（GRPO + Agent）EM=0.538。**

MiniLM 时代，多跳 agent 比 naive RAG 高 +4.2pp（0.480 vs 0.438）。切换到 BGE 后排名逆转：naive RAG 反而更高（−3.4pp 差距）。

根本原因：ctx_recall 从 0.748 提升到 0.907，BGE 直接用原始问题就能可靠地找到 gold passage，**多跳分解的检索增益消失了，但分解带来的错误传播代价依然存在**。

### Case：BGE 让原本检索失败的题直接答对

```
Q: What aerial aircraft using in the Vietnam War also injured american civilians?
Gold: Boeing KC-135 Stratotanker

MiniLM naive RAG: ctx_recall=0.0 → Pred: 错误（gold passage 未命中）
BGE   naive RAG: ctx_recall=1.0 → Pred: Boeing KC-135 Stratotanker  ✓
```

升级 embedding 后，不需要任何架构改变，直接检索就能找到答案——这类题占了 BGE 相对 MiniLM +13.4pp 提升的大头。

### 2. GRPO adapter 贡献稳定正向（+2.0pp）

```
Exp0（3B base + naive RAG） : EM=0.572
Exp6（3B GRPO + naive RAG） : EM=0.592  +2.0pp
```

GRPO 的价值在于改善答案格式，减少 EM 近失配：

```
Q: Who is the bassist of the band that Mike McCready founded?
Gold: Jeff Ament

Base pred: "Jeffrey Allen 'Jeff' Ament"   EM=0  ← 语义对但格式冗长
GRPO pred: "Jeff Ament"                   EM=1  ← 简洁精准
```

```
Q: Draco and the Malfoys rock band take their personas from a novel written by this author?
Gold: J. K. Rowling

Base pred: "J.K. Rowling"    EM=0  ← 缺空格
GRPO pred: "J. K. Rowling"   EM=1  ← 精确匹配
```

GRPO 的奖励函数（0.5×EM + 0.5×F1）直接训练模型输出与 gold 格式一致的短答案，减少了 base 模型输出冗长句子的倾向。

### 3. BGE 时代多跳 pipeline 有害（−5.4pp）

```
GRPO adapter 贡献：  Exp0 → Exp6  = +2.0pp  ✓
Multi-hop pipeline： Exp6 → Exp3  = −5.4pp  ✗
```

### Case：BGE 已经找到答案，Agent 拆解反而跑偏

**情形 A：简单题被强制拆解**
```
Q: What timezone do the United States Minor Outlying Islands observe?
Gold: Samoa Time

BGE naive RAG:  Pred: Samoa Time  ✓  (一跳直接命中)

Agent sub_q1:   What are the United States Minor Outlying Islands located in?
Agent hop1:     Pacific Ocean
Agent sub_q2:   What timezone does the Pacific Ocean observe?
Agent pred:     Hawaii-Aleutian Time  ✗
（这道题根本不需要分解，BGE 单跳就能找到答案段落，agent 的分解完全是噪声）
```

**情形 B：hop1 答案正确，生成步骤幻觉**
```
Q: Draco and the Malfoys rock band take their personas from a novel written by this author?
Gold: J. K. Rowling

BGE naive RAG:  Pred: J. K. Rowling  ✓

Agent sub_q1:   Who wrote the novel that Draco and the Malfoys are based on?
Agent hop1:     J.K. Rowling  ✓（正确找到中间实体）
Agent sub_q2:   What is the author of the novel that Draco and Malfoys' personas come from?
Agent pred:     Philip Pullman  ✗  ctx_recall=1.0
（gold passage 已经检索到，最终 answer_final 步骤却幻觉出了错误答案）
```

**情形 C：hop1 答案错误，链式传播**
```
Q: Marble Hill, South Australia is a ward that extends from what location in the north?
Gold: South Para Reservoir

BGE naive RAG:  Pred: South Para Reservoir  ✓

Agent sub_q1:   What location in the north does Marble Hill extend from?
Agent hop1:     Mount Bold Reservoir  ✗（第一跳就答错）
Agent sub_q2:   What is the location in the north that Marble Hill extends from?
Agent pred:     Adelaide Hills Council  ✗
（hop1 错误导致 sub_q2 基于错误前提，错误链式传播）
```

### 4. Query LoRA 改善分解质量，但瓶颈不在分解

Exp4（+Query LoRA）相比 Exp3 各指标均提升：

| 指标 | Exp3 | Exp4 | Delta |
|---|---|---|---|
| Exact Match | 0.538 | **0.550** | +1.2pp |
| Token F1 | 0.625 | **0.639** | +1.4pp |
| Context Recall | 0.901 | **0.931** | +3.0pp |

### Case：Query LoRA 改善 sub_q2 代入质量

```
Q: Rajindar Nath Rehbar is the writer of the song performed by the Ghazal singer
   married to whom?
Gold: Chitra Singh

Exp3 (base decompose):  sub_q2 未正确代入 hop1_answer → Pred: 错误
Exp4 (Query LoRA):
  sub_q1: Who is the Ghazal singer that performed the song?
  hop1:   Jagjit Singh
  sub_q2: Who was Jagjit Singh married to in the 1970s and '80s?  ← 正确代入
  Pred:   Chitra Singh  ✓
```

但 Exp4（EM=0.550）仍低于 Exp6（EM=0.592）。即使 ctx_recall（0.931）已超过 naive RAG（0.907），EM 仍然落后——说明**瓶颈不是 sub_q 质量，而是 pipeline 本身的误差传播结构**。

### 5. Exp2 Comparison 题异常（EM=0.264）

32B base 模型 Comparison EM 仅 0.264，远低于 Exp0（0.604）。根本原因：32B 未经 GRPO 微调，输出格式不稳定，倾向输出完整句子而非 yes/no，faithfulness verifier 判失败，触发 web search 覆盖了正确结果。这是 pipeline 对模型输出格式的隐式依赖导致的系统性故障。

---

## 总结

| 阶段 | 最优系统 | EM | 核心发现 |
|---|---|---|---|
| MiniLM | 3B GRPO + Agent | 0.480 | 多跳分解 +4.2pp，检索是瓶颈 |
| BGE | **3B GRPO + Naive RAG (Exp6)** | **0.592** | GRPO +2.0pp，多跳 −5.4pp，答案质量是关键 |
| 上界 | 32B + Golden Passages | 0.786 | 还差 19.4pp，主要在难题（Hard: 0.427 vs 0.607） |

**最终结论：** 多跳 agent 解决的是检索问题，不是推理问题。BGE 解决了检索问题之后，最优系统变成了「强检索器 + GRPO 答案模型 + 单跳」——比复杂的多跳 pipeline 更简单也更好。

GRPO 贡献量化：
```
GRPO adapter：  +1.8pp（MiniLM）/ +2.0pp（BGE）  ← 始终正向，跨 embedding 稳定
Multi-hop：     +4.2pp（MiniLM）/ −5.4pp（BGE）   ← 随检索质量翻转
BGE 升级：      +13.4pp（naive RAG 基线）          ← 单一变量最大收益
```
