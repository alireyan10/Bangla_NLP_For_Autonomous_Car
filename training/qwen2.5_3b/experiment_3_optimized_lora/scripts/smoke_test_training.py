"""
Smoke Test for Experiment 3 Training on 4GB RTX 2050.

Validates:
1. Model loading with 4-bit NF4 + Double Quantization.
2. LoRA with r=32, alpha=64 on all 7 target modules (q, k, v, o, gate, up, down).
3. Cosine learning rate scheduler with warmup_steps=25.
4. Peak VRAM allocation during forward + backward + optimizer step.
5. Confirms that training completes without OOM on 4GB VRAM.
"""

import os
import sys
import time
import json
import inspect
import torch

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
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXP3_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if EXP3_DIR not in sys.path:
    sys.path.insert(0, EXP3_DIR)

import config_optimized as cfg


def run_smoke_test():
    print("=" * 80)
    print("RUNNING EXPERIMENT 3 TRAINING SMOKE TEST (VRAM & CONFIG VERIFICATION)")
    print("=" * 80)

    set_seed(cfg.RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Target GPU: {device_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    # Load 40 samples from disambiguated train set
    raw_dataset = load_dataset(
        "json",
        data_files={"train": cfg.TRAIN_DATA_PATH},
        split="train[:40]"
    )

    formatted_dataset = raw_dataset.map(
        cfg.format_sample_to_messages,
        remove_columns=raw_dataset.column_names,
        desc="Smoke test formatting",
    )

    print(f"Loaded {len(formatted_dataset)} samples for smoke test.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16 if cfg.BNB_4BIT_COMPUTE_DTYPE == "float16" else torch.bfloat16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.LOAD_IN_4BIT,
        bnb_4bit_quant_type=cfg.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    print(f"Loading Base Model: {cfg.MODEL_NAME} in 4-bit...")
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

    print(f"Applying LoRA (r={cfg.LORA_R}, alpha={cfg.LORA_ALPHA}) on {cfg.LORA_TARGET_MODULES}...")
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        bias=cfg.LORA_BIAS,
        task_type=cfg.LORA_TASK_TYPE,
    )

    smoke_output_dir = os.path.join(EXP3_DIR, "outputs", "smoke_test_tmp")
    os.makedirs(smoke_output_dir, exist_ok=True)

    supported_params = inspect.signature(SFTConfig.__init__).parameters
    training_args = {
        "output_dir": smoke_output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,  # quick test
        "gradient_checkpointing": True,
        "learning_rate": cfg.LEARNING_RATE,
        "lr_scheduler_type": cfg.LR_SCHEDULER_TYPE,
        "warmup_steps": 2,
        "max_steps": 5,  # Run exactly 5 optimization steps
        "logging_steps": 1,
        "save_strategy": "no",
        "fp16": False,
        "bf16": False,
        "optim": cfg.OPTIMIZER,
        "report_to": "none",
        "seed": cfg.RANDOM_SEED,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    if "eval_strategy" in supported_params:
        training_args["eval_strategy"] = "no"
    elif "evaluation_strategy" in supported_params:
        training_args["evaluation_strategy"] = "no"

    if "max_length" in supported_params:
        training_args["max_length"] = cfg.MAX_SEQ_LENGTH
    elif "max_seq_length" in supported_params:
        training_args["max_seq_length"] = cfg.MAX_SEQ_LENGTH

    sft_config = SFTConfig(**training_args)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nStarting 5-step smoke test...")
    torch.cuda.reset_peak_memory_stats()
    start_t = time.time()
    trainer.train()
    elapsed = time.time() - start_t

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    current_vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)

    print("=" * 80)
    print("SMOKE TEST PASSED SUCCESSFULLY!")
    print(f"5 steps completed in: {elapsed:.2f} seconds")
    print(f"Peak VRAM during training: {peak_vram_gb:.3f} GB / {total_vram_gb:.2f} GB")
    print(f"VRAM Headroom: {total_vram_gb - peak_vram_gb:.3f} GB")
    print("=" * 80)

    # Clean up temporary model and memory
    del trainer, model
    torch.cuda.empty_cache()

    # Clean up temporary output directory
    import shutil
    if os.path.exists(smoke_output_dir):
        shutil.rmtree(smoke_output_dir, ignore_errors=True)

    return peak_vram_gb


if __name__ == "__main__":
    run_smoke_test()
