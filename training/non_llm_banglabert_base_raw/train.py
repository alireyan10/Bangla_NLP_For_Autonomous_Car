"""
Training Script for Non-LLM BanglaBERT-Base (110M) Multi-Head Model.
Trains on: training/qwen2.5_3b/experiment_3_optimized_lora/data/train_disambiguated.jsonl (2800 samples)
Validates on: training/qwen2.5_3b/experiment_3_optimized_lora/data/validation_disambiguated.jsonl (600 samples)
Saves best model based on Comprehensive 7-Field Exact Match on validation split.
"""

import os
import sys
import time
import json
import random
import csv
import io
from typing import Dict, Any

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from model import BanglaBERTMultiHead
from dataset import BanglaRobotDataset, bangla_robot_collate_fn
from json_formatter import assemble_robot_json
from PREPROCESSING_EXPERIMENT.evaluation.evaluator import evaluate_dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    label_mappings: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    model.eval()
    total_val_loss = 0.0
    eval_items = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            labels = {
                "intent": batch["intent"].to(device),
                "target": batch["target"].to(device),
                "speed_category": batch["speed_category"].to(device),
                "step_count": batch["step_count"].to(device),
                "step1": batch["step1"].to(device),
                "step2": batch["step2"].to(device),
                "step3": batch["step3"].to(device),
                "step4": batch["step4"].to(device),
                "linear_speed_class": batch["linear_speed_class"].to(device),
                "angular_speed_class": batch["angular_speed_class"].to(device),
                "linear_speed_val": batch["linear_speed_val"].to(device),
                "angular_speed_val": batch["angular_speed_val"].to(device),
            }

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    loss_weights=cfg.LOSS_WEIGHTS,
                )

            total_val_loss += outputs["loss"].item()

            pred_intent = torch.argmax(outputs["intent_logits"], dim=-1).cpu().numpy()
            pred_target = torch.argmax(outputs["target_logits"], dim=-1).cpu().numpy()
            pred_speed_cat = torch.argmax(outputs["speed_category_logits"], dim=-1).cpu().numpy()
            pred_step_count = torch.argmax(outputs["step_count_logits"], dim=-1).cpu().numpy()
            pred_s1 = torch.argmax(outputs["step1_logits"], dim=-1).cpu().numpy()
            pred_s2 = torch.argmax(outputs["step2_logits"], dim=-1).cpu().numpy()
            pred_s3 = torch.argmax(outputs["step3_logits"], dim=-1).cpu().numpy()
            pred_s4 = torch.argmax(outputs["step4_logits"], dim=-1).cpu().numpy()
            pred_lin = torch.argmax(outputs["linear_speed_logits"], dim=-1).cpu().numpy()
            pred_ang = torch.argmax(outputs["angular_speed_logits"], dim=-1).cpu().numpy()
            pred_lin_reg = outputs["linear_reg"].cpu().numpy()
            pred_ang_reg = outputs["angular_reg"].cpu().numpy()

            bs = input_ids.size(0)
            for i in range(bs):
                sample_preds = {
                    "intent_id": int(pred_intent[i]),
                    "target_id": int(pred_target[i]),
                    "speed_cat_id": int(pred_speed_cat[i]),
                    "step_count_id": int(pred_step_count[i]),
                    "step1_id": int(pred_s1[i]),
                    "step2_id": int(pred_s2[i]),
                    "step3_id": int(pred_s3[i]),
                    "step4_id": int(pred_s4[i]),
                    "linear_speed_class_id": int(pred_lin[i]),
                    "angular_speed_class_id": int(pred_ang[i]),
                    "linear_speed_reg": float(pred_lin_reg[i]),
                    "angular_speed_reg": float(pred_ang_reg[i]),
                }

                json_str, action_dict, status = assemble_robot_json(
                    sample_preds, label_mappings, use_setpoint_speeds=True
                )

                gt_item = {
                    "id": batch["id"][i],
                    "command": batch["command"][i],
                    "ground_truth": batch["ground_truth"][i],
                    "raw_prediction": json_str,
                }
                eval_items.append(gt_item)

    avg_loss = total_val_loss / max(1, len(val_loader))
    summary_metrics, _ = evaluate_dataset(eval_items, tolerance=cfg.NUMERIC_TOLERANCE)
    summary_metrics["val_loss"] = avg_loss
    return summary_metrics


def train_model():
    print("=" * 80)
    print("STARTING NON-LLM BASELINE TRAINING: BanglaBERT-Base (110M Parameters)")
    print("=" * 80)

    cfg.ensure_directories()
    set_seed(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Load label mappings
    with open(cfg.LABEL_MAPPINGS_PATH, "r", encoding="utf-8") as f:
        label_mappings = json.load(f)

    print(f"Loading tokenizer: {cfg.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    print(f"Loading training data from: {cfg.TRAIN_DATA_PATH}...")
    train_dataset = BanglaRobotDataset(
        cfg.TRAIN_DATA_PATH, tokenizer, label_mappings, max_seq_length=cfg.MAX_SEQ_LENGTH
    )
    print(f"Loaded {len(train_dataset)} training samples.")

    print(f"Loading validation data from: {cfg.VAL_DATA_PATH}...")
    val_dataset = BanglaRobotDataset(
        cfg.VAL_DATA_PATH, tokenizer, label_mappings, max_seq_length=cfg.MAX_SEQ_LENGTH
    )
    print(f"Loaded {len(val_dataset)} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=bangla_robot_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=bangla_robot_collate_fn,
    )

    print("\nInitializing BanglaBERT-Base multi-head model...")
    model = BanglaBERTMultiHead(cfg.MODEL_NAME, label_mappings)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    total_steps = (len(train_loader) // cfg.GRAD_ACCUM_STEPS) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history_csv = os.path.join(cfg.REPORTS_DIR, "training_history.csv")
    history_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_intent_acc",
        "val_target_acc",
        "val_speed_cat_acc",
        "val_action_seq_acc",
        "val_step_count_acc",
        "val_linear_speed_acc",
        "val_angular_speed_acc",
        "val_4field_em",
        "val_7field_em",
        "epoch_time_sec",
    ]

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()

    best_val_em = -1.0
    best_epoch = -1
    patience_counter = 0
    start_total_time = time.time()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS} [Train]")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            labels = {
                "intent": batch["intent"].to(device),
                "target": batch["target"].to(device),
                "speed_category": batch["speed_category"].to(device),
                "step_count": batch["step_count"].to(device),
                "step1": batch["step1"].to(device),
                "step2": batch["step2"].to(device),
                "step3": batch["step3"].to(device),
                "step4": batch["step4"].to(device),
                "linear_speed_class": batch["linear_speed_class"].to(device),
                "angular_speed_class": batch["angular_speed_class"].to(device),
                "linear_speed_val": batch["linear_speed_val"].to(device),
                "angular_speed_val": batch["angular_speed_val"].to(device),
            }

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                    loss_weights=cfg.LOSS_WEIGHTS,
                )
                loss = outputs["loss"] / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += outputs["loss"].item()
            pbar.set_postfix({"loss": f"{outputs['loss'].item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_metrics = validate_epoch(model, val_loader, label_mappings, device)
        epoch_time = time.time() - epoch_start

        val_7field_em = val_metrics["exact_match_7field"]
        val_4field_em = val_metrics["exact_match_4field"]

        print(
            f"\nEpoch {epoch:02d} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} "
            f"| 4-Field EM: {val_4field_em*100:.2f}% | 7-Field EM: {val_7field_em*100:.2f}% "
            f"| Time: {epoch_time:.1f}s"
        )
        print(
            f"  Accuracy -> Intent: {val_metrics['intent_accuracy']*100:.2f}% "
            f"| Target: {val_metrics['target_accuracy']*100:.2f}% "
            f"| SpeedCat: {val_metrics['speed_category_accuracy']*100:.2f}% "
            f"| Seq: {val_metrics['action_sequence_accuracy']*100:.2f}%"
        )

        # Log history
        history_row = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_metrics["val_loss"], 4),
            "val_intent_acc": round(val_metrics["intent_accuracy"], 4),
            "val_target_acc": round(val_metrics["target_accuracy"], 4),
            "val_speed_cat_acc": round(val_metrics["speed_category_accuracy"], 4),
            "val_action_seq_acc": round(val_metrics["action_sequence_accuracy"], 4),
            "val_step_count_acc": round(val_metrics["step_count_accuracy"], 4),
            "val_linear_speed_acc": round(val_metrics["linear_speed_accuracy"], 4),
            "val_angular_speed_acc": round(val_metrics["angular_speed_accuracy"], 4),
            "val_4field_em": round(val_4field_em, 4),
            "val_7field_em": round(val_7field_em, 4),
            "epoch_time_sec": round(epoch_time, 2),
        }
        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writerow(history_row)

        # Check early stopping / best checkpoint
        if val_7field_em > best_val_em:
            best_val_em = val_7field_em
            best_epoch = epoch
            patience_counter = 0

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_em": best_val_em,
                "val_metrics": val_metrics,
                "label_mappings": label_mappings,
                "config": {
                    "model_name": cfg.MODEL_NAME,
                    "max_seq_length": cfg.MAX_SEQ_LENGTH,
                    "seed": cfg.RANDOM_SEED,
                },
            }
            torch.save(checkpoint_data, cfg.BEST_MODEL_PATH)
            print(f"  >>> Best checkpoint updated! Saved to: {cfg.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{cfg.PATIENCE}")
            if patience_counter >= cfg.PATIENCE:
                print(f"\nEarly stopping triggered after epoch {epoch}. Best epoch was {best_epoch} with 7-Field EM {best_val_em*100:.2f}%.")
                break

    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation 7-Field Exact Match: {best_val_em*100:.2f}% (Epoch {best_epoch})")
    print(f"Best Model Checkpoint: {cfg.BEST_MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    train_model()
