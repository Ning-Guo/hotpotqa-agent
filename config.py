# config.py — central configuration for the live multi-hop QA agent
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
INDEX_PATH  = os.path.join(DATA_DIR, "faiss.index")   # saved FAISS index
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")  # paragraphs: {title, text}
EVAL_PATH   = os.path.join(DATA_DIR, "grpo_val.jsonl")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_NAME        = "Qwen/Qwen2.5-3B-Instruct"
# GRPO_ADAPTER_REPO = os.path.join(BASE_DIR, "model", "grpo_rag")
GRPO_ADAPTER_REPO = "Norm11/qwen2.5-3b-grpo-hotpotqa" 

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS       = 64   # final answers are short phrases
REASONING_MAX_TOKENS = 48   # classification / decomposition outputs are shorter

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"
TOP_K                  = 5    # passages per sub-query
FAITHFULNESS_THRESHOLD = 0.3  # verify: fraction of answer tokens in context
