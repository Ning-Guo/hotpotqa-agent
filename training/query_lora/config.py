# training/query_lora/config.py
"""
Configuration for Query LoRA experiment.
All paths are relative to the project root.
"""

import os

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
# HotpotQA train split (full ~90K questions)
HOTPOT_TRAIN_PATH = os.path.join(ROOT, "data", "hotpot_train_v1.1.json")

# Eval set — NEVER used for training
EVAL_PATH = os.path.join(ROOT, "data", "grpo_val.jsonl")

# FAISS index (shared with main pipeline)
INDEX_PATH  = os.path.join(ROOT, "data", "faiss.index")
CORPUS_PATH = os.path.join(ROOT, "data", "corpus.jsonl")

# Output directory for generated data
DATA_DIR = os.path.join(ROOT, "training", "query_lora", "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_DATA_PATH   = os.path.join(DATA_DIR, "val.jsonl")
RAW_DATA_PATH   = os.path.join(DATA_DIR, "raw_generated.jsonl")  # before quality filter

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Teacher model for annotation (sub_q1, sub_q2, comparison rewrite)
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# Student model (base, no adapter)
STUDENT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Existing GRPO adapter (used in eval, not in training)
GRPO_ADAPTER  = "Norm11/qwen2.5-3b-sft-grpo-hotpotqa_v3/grpo_adapter"

# Embedding model for FAISS retrieval
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
N_BRIDGE_SAMPLES     = 12_000   # bridge questions to annotate
N_COMPARISON_SAMPLES = 3_000    # comparison questions to annotate
VAL_FRACTION         = 0.05     # fraction held out for validation
TOP_K                = 5        # FAISS retrieval top-k (same as main pipeline)
TEACHER_MAX_TOKENS   = 80       # max tokens for 32B teacher generation
HOP1_MAX_TOKENS      = 48       # max tokens for 3B hop1_answer generation

# Quality filter
MIN_CTX_RECALL = 1.0   # only keep samples where query retrieves all gold passages

# ---------------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------------
LORA_R            = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_EPOCHS         = 3
LEARNING_RATE        = 2e-4
PER_DEVICE_BATCH     = 16
GRAD_ACCUMULATION    = 2
MAX_SEQ_LENGTH       = 1024
WARMUP_RATIO         = 0.05
LR_SCHEDULER         = "cosine"
SAVE_STEPS           = 100
LOGGING_STEPS        = 10

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(ROOT, "training", "query_lora", "checkpoints", "query_lora")
EVAL_RESULTS_DIR = os.path.join(ROOT, "results")
