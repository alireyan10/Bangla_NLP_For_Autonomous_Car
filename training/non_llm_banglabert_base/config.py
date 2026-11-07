"""
Configuration for Non-LLM BanglaBERT-Base (110M Parameters) Pipeline.
Model: csebuetnlp/banglabert (hidden_size=768)
"""

import os
import unicodedata
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

EXP_DIR = SCRIPT_DIR
DATA_DIR = os.path.join(EXP_DIR, "data")
OUTPUTS_DIR = os.path.join(EXP_DIR, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
EVAL_DIR = os.path.join(EXP_DIR, "evaluation")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")

# Clean Disambiguated Data Splits
TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_3_optimized_lora", "data", "train_disambiguated.jsonl"
)
VAL_DATA_PATH = os.path.join(
    PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_3_optimized_lora", "data", "validation_disambiguated.jsonl"
)
TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT, "PREPROCESSING_EXPERIMENT", "output", "splits", "test.jsonl"
)

LABEL_MAPPINGS_PATH = os.path.join(DATA_DIR, "label_mappings.json")
BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_banglabert_base_model.pt")

MODEL_NAME = "csebuetnlp/banglabert"
MAX_SEQ_LENGTH = 128
RANDOM_SEED = 42

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2  # Effective batch size = 32
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 12
WARMUP_RATIO = 0.1
PATIENCE = 3
NUMERIC_TOLERANCE = 1e-3

LOSS_WEIGHTS = {
    "intent": 1.2,
    "target": 1.2,
    "speed_category": 0.8,
    "step_count": 0.8,
    "step1": 1.0,
    "step2": 0.8,
    "step3": 0.6,
    "step4": 0.4,
    "linear_speed": 1.0,
    "angular_speed": 1.0,
}


def normalize_bangla_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    for ch in ["\u200B", "\uFEFF", "\u00AD"]:
        text = text.replace(ch, "")
    text = re.sub(r"[\s\u00A0\u3000]+", " ", text).strip()
    return text


def ensure_directories() -> None:
    for d in [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, EVAL_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
