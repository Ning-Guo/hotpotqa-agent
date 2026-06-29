# Query LoRA — 实验方案

## 动机

当前系统（3B GRPO + Agent）在与 32B base model + Agent 的对比实验（Exp2 vs Exp3）中，
发现 95 个 bridge 题 Exp2 答对、Exp3 答错的案例，经 sub_q1/hop1_answer/sub_q2 分析：

| 类别 | 数量 | 占比 | 根本原因 |
|---|---|---|---|
| A：sub_q1 质量差，第一跳方向偏移 | 63 | 66% | 3B base model 生成的 sub_q1 不精准 |
| B：hop1 正确，sub_q2 未代入 hop1_answer | 20 | 21% | 3B base model 在 formulate_hop2 上失败 |
| C：query 相同，答案仍错 | 12 | 13% | 纯生成问题，与 query 改写无关 |

**结论**：87% 的失败根源在 query 改写能力（sub_q1 / sub_q2 / comparison rewrite），
而这三步目前使用完全未微调的 base model。新增一个专门的 Query LoRA adapter 来解决这个问题。

---

## 方案概述

用 32B 模型作为 teacher，生成高质量的 sub_q1 / sub_q2 / comparison rewrite 标注数据，
SFT 3B 模型的一个新 LoRA adapter（Query LoRA），与现有 GRPO adapter 独立共存。

```
Qwen2.5-3B-Instruct (base)
    ├── GRPO adapter   → 最终答案生成（现有，不动）
    └── Query LoRA     → sub_q1 / sub_q2 / comparison rewrite（新增）
```

推理时通过 PEFT 的 `set_adapter()` 在两个 adapter 间切换，不需要加载两个独立模型。

---

## 数据方案

### 数据来源
HotpotQA 训练集（distractor split，~90K 题），**严禁使用 eval 集（data/grpo_val.jsonl）**。
抽取约 20K 题进行标注（bridge:comparison ≈ 8:2）。

### 三种任务类型

#### 任务 1：sub_q1 生成（bridge）
- 输入给 32B：`_decompose_prompt(question)`（与 `src/reasoner.py` 完全一致）
- 输出：干净的第一跳子问题
- 不需要 passages
- 质量过滤：用生成的 sub_q1 做 FAISS 检索，ctx_recall=1.0 才保留

#### 任务 2：sub_q2 生成（bridge）
**关键：hop1_answer 必须用 3B base model 生成，不能用 32B 的。**
原因：推理时 hop1_answer 是 3B 生成的，用 3B 的输出训练 sub_q2，能让训练和推理分布一致。

流程：
```
Step A: 32B 生成 sub_q1（与任务 1 共享，不重复生成）
Step B: 用 sub_q1 做 FAISS 检索，拿到 passages
Step C: 3B base model 看 passages 回答 sub_q1 → hop1_answer
Step D: 32B 以 _hop2_prompt(question, 3B_hop1_answer) 生成 sub_q2
```

质量过滤：
- sub_q2 做 FAISS 检索，ctx_recall=1.0 才保留
- 过滤掉 sub_q2 里没有出现 hop1_answer 关键词的样本（防止 B2 模式）

#### 任务 3：comparison rewrite
- 输入给 32B：`_rewrite_prompt(question)`（与 `src/reasoner.py` 完全一致）
- 输出：2 条 sub-query，每个实体一条
- 不需要 passages
- 质量过滤：union 检索 ctx_recall=1.0 才保留

### 数据规模

| 任务 | 生成量 | 过滤后（估计 65%）|
|---|---|---|
| sub_q1 | 12,000 | ~7,800 |
| sub_q2 | 12,000 | ~7,800 |
| comparison rewrite | 3,000 | ~2,000 |
| **合计** | **27,000** | **~17,600** |

### SFT 样本格式

```json
{
  "messages": [
    {"role": "user", "content": "<prompt 完整内容>"},
    {"role": "assistant", "content": "<32B 生成的目标 sub_q>"}
  ]
}
```

三种任务混合打乱，不加任务标签。

---

## Fine-Tune 方案

### LoRA 配置
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```

### 训练配置
| 参数 | 值 |
|---|---|
| Base model | Qwen2.5-3B-Instruct（直接从 base 开始，不加载 GRPO adapter）|
| 损失函数 | Cross-entropy，只在 assistant completion 上计算（prompt mask 掉）|
| Epochs | 3 |
| Learning rate | 2e-4，cosine decay |
| Batch size | 16（per GPU），gradient accumulation=2 |
| Max length | 1024 tokens |
| Hardware | 1× A100 40GB（3B + rank-16 LoRA < 20GB）|
| 预计时长 | 约 2–3 小时 |

### 推理集成（adapter 切换逻辑）

| Pipeline 步骤 | 激活的 adapter |
|---|---|
| classify | base model（disable_adapter）|
| decompose（sub_q1）| **Query LoRA** |
| answer_hop1 | base model（disable_adapter）|
| formulate_hop2（sub_q2）| **Query LoRA** |
| rewrite_comparison | **Query LoRA** |
| answer_final（bridge）| **GRPO adapter** |

---

## 测试方案

### 第一层：sub_q 质量评估（`03_eval_subq_quality.py`）

在验证集上，分别用 base model 和 Query LoRA 生成 sub_q1 / sub_q2，对比：
- sub_q1 的 ctx_recall（用 FAISS 检索后与 gold titles 对比）
- sub_q2 的 ctx_recall
- sub_q2 中 hop1_answer 关键词出现率（检验 B2 模式是否修复）

### 第二层：端到端 eval（`04_eval_e2e.py`）

加载 3B + Query LoRA + GRPO adapter，运行完整 500 题评测，与 Exp3 baseline 对比。

### 预期结果

| 指标 | 当前 Exp3 | 目标（Query LoRA 后）|
|---|---|---|
| 整体 EM | 0.540 | ~0.58–0.60 |
| Bridge EM | 0.533 | ~0.58–0.62 |
| ctx_recall | 0.902 | ~0.93–0.94 |
| Category A+B 失败数 | 83/95 | 目标 < 50 |

---

## 脚本列表

```
training/query_lora/
├── PLAN.md                  本文件
├── config.py                路径、模型名、超参数
├── 01_generate_data.py      数据生成（32B + FAISS + 3B）
├── 02_train_query_lora.py   SFT 训练
├── 03_eval_subq_quality.py  sub_q 质量中间评估
└── 04_eval_e2e.py           端到端完整 agent 评测
```

## 运行顺序

```bash
# 0. 确认 FAISS index 已存在（或先跑 build_corpus.py）

# 1. 生成训练数据（需要 1× A100 80GB，约 3-4 小时）
python training/query_lora/01_generate_data.py \
  --n-bridge 12000 --n-comparison 3000 --load-index

# 2. 训练 Query LoRA（需要 1× A100 40GB，约 2-3 小时）
python training/query_lora/02_train_query_lora.py \
  --data training/query_lora/data/train.jsonl \
  --output training/query_lora/checkpoints/query_lora

# 3. 中间质量评估（约 10 分钟）
python training/query_lora/03_eval_subq_quality.py \
  --adapter training/query_lora/checkpoints/query_lora \
  --load-index

# 4. 端到端评测（约 20-30 分钟）
python training/query_lora/04_eval_e2e.py \
  --query-adapter training/query_lora/checkpoints/query_lora \
  --load-index
```
