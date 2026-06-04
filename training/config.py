# training/config.py — shared configuration for all training scripts

import os

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
BASE_MODEL    = "Qwen/Qwen2.5-3B-Instruct"   # starting point for SFT
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # or set TEACHER_API_MODEL below

# If using an OpenAI-compatible API for teacher inference instead of local model:
# Set TEACHER_API_BASE and TEACHER_API_KEY as environment variables, e.g.:
#   export TEACHER_API_BASE="https://api.together.xyz/v1"
#   export TEACHER_API_KEY="your-key"
#   export TEACHER_API_MODEL="Qwen/Qwen2.5-72B-Instruct-Turbo"
TEACHER_API_BASE  = os.environ.get("TEACHER_API_BASE",  None)
TEACHER_API_KEY   = os.environ.get("TEACHER_API_KEY",   None)
TEACHER_API_MODEL = os.environ.get("TEACHER_API_MODEL", None)

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------
SFT_SIZE             = 30_000   # source examples fed to teacher (yields ~22K after filtering)
GRPO_SIZE            = 60_000
GOLD_ABSENT_FRACTION = 0.10     # 10% of examples get gold passage removed
                                 # padded back to 10 passages using global distractor pool
RANDOM_SEED          = 42

# ---------------------------------------------------------------------------
# SFT training
# ---------------------------------------------------------------------------
SFT_LORA_R               = 16
SFT_LORA_ALPHA            = 32
SFT_LORA_DROPOUT          = 0.05
SFT_LORA_TARGET_MODULES   = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
SFT_EPOCHS        = 3
SFT_BATCH_SIZE    = 4       # per device
SFT_GRAD_ACCUM    = 8       # effective batch = 32
SFT_LR            = 2e-4
SFT_MAX_SEQ_LEN   = 2048
SFT_WARMUP_RATIO  = 0.05

# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------
GRPO_LORA_R             = 16
GRPO_LORA_ALPHA         = 32
GRPO_LORA_TARGET_MODULES = SFT_LORA_TARGET_MODULES
GRPO_LR                 = 5e-6
GRPO_BATCH_SIZE         = 4       # per device; must be divisible by GRPO_NUM_GENERATIONS
GRPO_GRAD_ACCUM         = 8
GRPO_NUM_GENERATIONS    = 4       # rollouts per prompt
GRPO_MAX_NEW_TOKENS     = 512
GRPO_MAX_PROMPT_LEN     = 1536
GRPO_KL_COEF            = 0.01    # KL penalty against SFT-merged model
GRPO_EPOCHS             = 1

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
REWARD_ACCURACY_WEIGHT    = 0.70
REWARD_GROUNDING_WEIGHT   = 0.25
REWARD_LENGTH_MAX_PENALTY = 0.05  # max penalty for long answers

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
CKPT_DIR          = os.path.join(BASE_DIR, "checkpoints")

SFT_SOURCE_PATH   = os.path.join(DATA_DIR, "sft_source.jsonl")
SFT_CLEAN_PATH    = os.path.join(DATA_DIR, "sft_clean.jsonl")
GRPO_TRAIN_PATH   = os.path.join(DATA_DIR, "grpo_train.jsonl")
INNER_VAL_PATH    = os.path.join(DATA_DIR, "inner_val.jsonl")

SFT_ADAPTER_PATH  = os.path.join(CKPT_DIR, "sft_adapter")
SFT_MERGED_PATH   = os.path.join(CKPT_DIR, "sft_merged")
GRPO_ADAPTER_PATH = os.path.join(CKPT_DIR, "grpo_adapter")
