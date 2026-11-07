"""
Qwen2.5-3B + QLoRA Training Script: Experiment 3 (Optimized LoRA)

Enhancements:
1. Disambiguated Train & Validation Dataset (0 conflicting labels).
2. LoRA r=32, alpha=64 (doubled capacity for sequential planning).
3. Cosine Learning Rate Scheduler with 25 Warmup Steps.
4. Strict target casing (lowercase entities, NONE for empty).
5. Comprehensive logging of training loss, validation loss, peak VRAM, and elapsed time.
"""

import os
import sys
import time
import json
import inspect

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config_optimized as cfg


def check_environment_and_paths():
    """Verify GPU and dataset paths before starting training."""
    required_files = {
        "Disambiguated Training Data": cfg.TRAIN_DATA_PATH,
        "Disambiguated Validation Data": cfg.VAL_DATA_PATH,
    }

    for label, path in required_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file missing: {path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please verify GPU setup.")

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("=" * 80)
    print("ENVIRONMENT & HARDWARE CHECK")
    print(f"GPU: {device_name} ({total_vram:.2f} GB VRAM)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Train path: {cfg.TRAIN_DATA_PATH}")
    print(f"Val path:   {cfg.VAL_DATA_PATH}")
    print(f"Test split is strictly excluded from training: {cfg.TEST_DATA_PATH}")
    print("=" * 80)


def load_and_prepare_datasets():
    """Loads disambiguated train and validation splits and formats to ChatML."""
    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": cfg.TRAIN_DATA_PATH,
            "validation": cfg.VAL_DATA_PATH,
        },
    )

    print(
        f"Raw sample count: Train={len(raw_dataset['train'])}, "
        f"Validation={len(raw_dataset['validation'])}"
    )

    formatted_dataset = {}
    for split_name in ["train", "validation"]:
        source = raw_dataset[split_name]
        orig_cols = source.column_names
        formatted_dataset[split_name] = source.map(
            cfg.format_sample_to_messages,
            remove_columns=orig_cols,
            desc=f"{split_name}: ChatML formatting",
        )

    print("\nSample formatted message (Train[0]):")
    print(json.dumps(formatted_dataset["train"][0]["messages"], ensure_ascii=False, indent=2))
    return formatted_dataset


def build_training_config():
    """Constructs SFTConfig with Cosine LR scheduler, 25 warmup steps, and checkpointing."""
    supported_params = inspect.signature(SFTConfig.__init__).parameters

    training_args = {
        "output_dir": cfg.CHECKPOINT_DIR,
        "per_device_train_batch_size": cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": cfg.PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": cfg.GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": cfg.GRADIENT_CHECKPOINTING,
        "learning_rate": cfg.LEARNING_RATE,
        "lr_scheduler_type": cfg.LR_SCHEDULER_TYPE,
        "warmup_steps": cfg.WARMUP_STEPS,
        "num_train_epochs": cfg.NUM_TRAIN_EPOCHS,
        "logging_steps": cfg.LOGGING_STEPS,
        "eval_steps": cfg.EVAL_STEPS,
        "save_strategy": "steps",
        "save_steps": cfg.SAVE_STEPS,
        "save_total_limit": cfg.SAVE_TOTAL_LIMIT,
        "load_best_model_at_end": True,
        "metric_for_best_model": cfg.METRIC_FOR_BEST_MODEL,
        "greater_is_better": cfg.GREATER_IS_BETTER,
        "fp16": cfg.FP16,
        "bf16": cfg.BF16,
        "optim": cfg.OPTIMIZER,
        "report_to": "none",
        "seed": cfg.RANDOM_SEED,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    if "eval_strategy" in supported_params:
        training_args["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported_params:
        training_args["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("Neither eval_strategy nor evaluation_strategy in SFTConfig.")

    if "max_length" in supported_params:
        training_args["max_length"] = cfg.MAX_SEQ_LENGTH
    elif "max_seq_length" in supported_params:
        training_args["max_seq_length"] = cfg.MAX_SEQ_LENGTH
    else:
        raise RuntimeError("Neither max_length nor max_seq_length in SFTConfig.")

    return SFTConfig(**training_args)


def main():
    print("=" * 80)
    print("EXPERIMENT 3: QWEN2.5-3B + QLORA (OPTIMIZED LORA & DISAMBIGUATED DATA)")
    print("=" * 80)

    set_seed(cfg.RANDOM_SEED)
    check_environment_and_paths()
    cfg.ensure_directories()

    dataset = load_and_prepare_datasets()

    print(f"\nLoading Tokenizer: {cfg.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nConfiguring 4-bit NF4 Quantization...")
    compute_dtype = torch.float16 if cfg.BNB_4BIT_COMPUTE_DTYPE == "float16" else torch.bfloat16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.LOAD_IN_4BIT,
        bnb_4bit_quant_type=cfg.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    print(f"Loading Base Model: {cfg.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.GRADIENT_CHECKPOINTING,
    )
    model.config.use_cache = False

    print(f"\nConfiguring LoRA (r={cfg.LORA_R}, alpha={cfg.LORA_ALPHA}, dropout={cfg.LORA_DROPOUT})...")
    print(f"Target Modules: {cfg.LORA_TARGET_MODULES}")
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        bias=cfg.LORA_BIAS,
        task_type=cfg.LORA_TASK_TYPE,
    )

    training_config = build_training_config()

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nTrainer successfully initialized.")
    print(f"Total training steps: {len(dataset['train']) // cfg.GRADIENT_ACCUMULATION_STEPS * cfg.NUM_TRAIN_EPOCHS}")
    print(f"Warmup steps: {cfg.WARMUP_STEPS} (Scheduler: {cfg.LR_SCHEDULER_TYPE})")
    print(f"Output Adapter: {cfg.ADAPTER_DIR}")
    print("\nStarting Training...")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # Check if a valid checkpoint exists to resume
    resume_checkpoint = None
    if os.path.exists(cfg.CHECKPOINT_DIR):
        ckpts = [
            os.path.join(cfg.CHECKPOINT_DIR, d)
            for d in os.listdir(cfg.CHECKPOINT_DIR)
            if d.startswith("checkpoint-")
            and d.split("-")[-1].isdigit()
            and os.path.isdir(os.path.join(cfg.CHECKPOINT_DIR, d))
        ]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = ckpts[-1]
            print(f"Resuming training from verified valid checkpoint: {resume_checkpoint}")

    training_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed_seconds = time.time() - start_time

    print(f"\nTraining completed in {elapsed_seconds/60:.2f} minutes ({elapsed_seconds:.1f} s).")

    print("\nSaving final fine-tuned adapter...")
    trainer.save_model(cfg.ADAPTER_DIR)
    tokenizer.save_pretrained(cfg.ADAPTER_DIR)
    print(f"Adapter saved to: {cfg.ADAPTER_DIR}")

    # Save log history
    log_history = trainer.state.log_history
    history_csv_path = os.path.join(cfg.REPORTS_DIR, "training_history.csv")
    pd.DataFrame(log_history).to_csv(history_csv_path, index=False)

    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    best_val_loss = min(eval_losses) if eval_losses else None
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    summary = {
        "experiment": "experiment_3_optimized_lora",
        "description": "Label Disambiguation + LoRA Optimization (r=32, alpha=64, Cosine LR, Warmup=25)",
        "model": cfg.MODEL_NAME,
        "lora_r": cfg.LORA_R,
        "lora_alpha": cfg.LORA_ALPHA,
        "lr_scheduler": cfg.LR_SCHEDULER_TYPE,
        "warmup_steps": cfg.WARMUP_STEPS,
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "test_samples_loaded_during_training": 0,
        "epochs": cfg.NUM_TRAIN_EPOCHS,
        "learning_rate": cfg.LEARNING_RATE,
        "training_seconds": round(elapsed_seconds, 2),
        "training_loss": float(training_result.training_loss),
        "best_validation_loss": best_val_loss,
        "peak_gpu_memory_gb": peak_vram_gb,
        "gpu": torch.cuda.get_device_name(0),
        "adapter_dir": cfg.ADAPTER_DIR,
        "checkpoint_dir": cfg.CHECKPOINT_DIR,
    }

    summary_path = os.path.join(cfg.REPORTS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY REPORT")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
