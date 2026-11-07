"""
Qwen2.5-3B + QLoRA Configuration
Experiment 3: Label Disambiguation + Optimized LoRA (r=32, alpha=64, Cosine LR, Warmup=25)

This configuration defines paths, model hyperparameters, ChatML formatting,
and training settings for Experiment 3.
"""

import json
import os
from typing import Any, Dict


# ============================================================
# 1. PROJECT & DATASET PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Reyan\Downloads\Case_Study"

EXP_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "qwen2.5_3b",
    "experiment_3_optimized_lora",
)

DATA_DIR = os.path.join(EXP_DIR, "data")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_disambiguated.jsonl")
VAL_DATA_PATH = os.path.join(DATA_DIR, "validation_disambiguated.jsonl")

# Test split is strictly frozen from the benchmark
TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "PREPROCESSING_EXPERIMENT",
    "output",
    "splits",
    "test.jsonl",
)

LABEL_CORRECTION_LOG_PATH = os.path.join(DATA_DIR, "label_correction_log.json")


# ============================================================
# 2. EXPERIMENT OUTPUT PATHS
# ============================================================

OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "checkpoints", "checkpoint-1050")
EVAL_DIR = os.path.join(EXP_DIR, "evaluation")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")


# ============================================================
# 3. OUTPUT SCHEMA
# ============================================================

OUTPUT_FIELDS = [
    "intent",
    "target",
    "speed_category",
    "linear_speed",
    "angular_speed",
    "action_sequence",
    "step_count",
]


# ============================================================
# 4. MODEL & CHATML SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_SEQ_LENGTH = 512
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a robot command mapping assistant. "
    "Convert the given Bangla robot command into a structured JSON "
    "action plan with exactly these fields: intent, target, "
    "speed_category, linear_speed, angular_speed, action_sequence, "
    "step_count. Return only valid JSON."
)

IM_END_TOKEN = "<|im_end|>"


# ============================================================
# 5. QLoRA & 4-BIT QUANTIZATION SETTINGS (EXP 3 OPTIMIZED)
# ============================================================

# LoRA rank and alpha doubled from baseline (16/32 -> 32/64)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = "float16"


# ============================================================
# 6. TRAINING HYPERPARAMETERS (EXP 3 OPTIMIZED)
# ============================================================

NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 25

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
GRADIENT_CHECKPOINTING = True

LOGGING_STEPS = 10
EVAL_STEPS = 175
SAVE_STEPS = 175
SAVE_TOTAL_LIMIT = 2

METRIC_FOR_BEST_MODEL = "eval_loss"
GREATER_IS_BETTER = False

FP16 = False
BF16 = False

OPTIMIZER = "paged_adamw_32bit"


# ============================================================
# 7. EVALUATION SETTINGS
# ============================================================

NUMERIC_TOLERANCE = 1e-3
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
DO_SAMPLE = False


# ============================================================
# 8. DIRECTORY SETUP HELPER
# ============================================================

def ensure_directories() -> None:
    """Ensure all required directories exist without overwriting."""
    for directory in [
        EXP_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        ADAPTER_DIR,
        EVAL_DIR,
        REPORTS_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)


# ============================================================
# 9. ACTION SEQUENCE PARSER
# ============================================================

def parse_action_sequence(value: Any) -> list:
    """Standardizes action sequence into list of strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value]

    if value is None:
        raise ValueError("Action Sequence cannot be None.")

    text = str(value).strip()
    if not text:
        raise ValueError("Action Sequence cannot be empty.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except json.JSONDecodeError:
        pass

    if "->" in text:
        return [part.strip() for part in text.split("->") if part.strip()]

    return [text]


# ============================================================
# 10. CHATML FORMATTER WITH STRICT TARGET CASING
# ============================================================

def format_sample_to_messages(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a raw record into ChatML messages.
    Enforces strict lowercase for target entities, NONE for empty.
    """
    required_columns = [
        "Bangla Command",
        "Intent",
        "Target",
        "Speed Category",
        "Linear Speed (m/s)",
        "Angular Speed (rad/s)",
        "Action Sequence",
        "Step Count",
    ]

    missing_columns = [col for col in required_columns if col not in record]
    if missing_columns:
        raise KeyError(f"Dataset missing required columns: {missing_columns}")

    command = str(record["Bangla Command"]).strip()
    if not command:
        raise ValueError("Bangla Command cannot be empty.")

    raw_target = str(record["Target"]).strip()
    clean_target = "NONE" if raw_target.upper() == "NONE" else raw_target.lower()

    action_plan = {
        "intent": str(record["Intent"]).strip(),
        "target": clean_target,
        "speed_category": str(record["Speed Category"]).strip(),
        "linear_speed": float(record["Linear Speed (m/s)"]),
        "angular_speed": float(record["Angular Speed (rad/s)"]),
        "action_sequence": parse_action_sequence(record["Action Sequence"]),
        "step_count": int(record["Step Count"]),
    }

    assistant_json = json.dumps(
        action_plan,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
            {"role": "assistant", "content": assistant_json},
        ]
    }
