"""
Evaluation Script for Non-LLM BanglaBERT-Base (110M) Multi-Head Model.
Evaluates on the frozen benchmark test split:
  PREPROCESSING_EXPERIMENT/output/splits/test.jsonl (IDs 3401-4000, 600 samples)
Calculates:
- Standardized 10 metrics using evaluator.py
- Per-class classification reports (Precision, Recall, F1)
- Confusion matrices for key heads
- Inference latency & model parameter count
"""

import os
import sys
import time
import json
import io
from typing import Dict, Any, List

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(max(8, len(class_names) * 0.4), max(6, len(class_names) * 0.35)))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 20,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_test_evaluation():
    print("=" * 80)
    print("EVALUATING NON-LLM BASELINE: BanglaBERT-Base (110M) on Frozen Test Split (600 Samples)")
    print("=" * 80)

    cfg.ensure_directories()
    cm_dir = os.path.join(cfg.EVAL_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    predictions_path = os.path.join(cfg.EVAL_DIR, "test_predictions.jsonl")
    metrics_path = os.path.join(cfg.EVAL_DIR, "metrics.json")
    reports_txt_path = os.path.join(cfg.EVAL_DIR, "classification_reports.txt")

    if not os.path.exists(cfg.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {cfg.BEST_MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from: {cfg.BEST_MODEL_PATH}...")
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    label_mappings = checkpoint["label_mappings"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = BanglaBERTMultiHead(cfg.MODEL_NAME, label_mappings)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"BanglaBERT-Base Total Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    print(f"Loading frozen test data from: {cfg.TEST_DATA_PATH}...")
    test_dataset = BanglaRobotDataset(
        cfg.TEST_DATA_PATH, tokenizer, label_mappings, max_seq_length=cfg.MAX_SEQ_LENGTH
    )
    print(f"Loaded {len(test_dataset)} test samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=bangla_robot_collate_fn,
    )

    eval_items = []
    latencies = []

    y_true_intent, y_pred_intent = [], []
    y_true_target, y_pred_target = [], []
    y_true_speed, y_pred_speed = [], []
    y_true_step, y_pred_step = [], []

    print("\nRunning inference over test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            t0 = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            t1 = time.perf_counter()
            batch_latency_ms = ((t1 - t0) * 1000) / input_ids.size(0)
            latencies.extend([batch_latency_ms] * input_ids.size(0))

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

                y_true_intent.append(int(batch["intent"][i].item()))
                y_pred_intent.append(int(pred_intent[i]))

                y_true_target.append(int(batch["target"][i].item()))
                y_pred_target.append(int(pred_target[i]))

                y_true_speed.append(int(batch["speed_category"][i].item()))
                y_pred_speed.append(int(pred_speed_cat[i]))

                y_true_step.append(int(batch["step_count"][i].item()))
                y_pred_step.append(int(pred_step_count[i]))

    # Save test predictions
    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved test predictions to: {predictions_path}")

    # Compute official standardized metrics
    summary_metrics, detailed_results = evaluate_dataset(eval_items, tolerance=cfg.NUMERIC_TOLERANCE)
    avg_latency = float(np.mean(latencies))
    summary_metrics["average_latency_ms"] = avg_latency
    summary_metrics["model_parameters"] = total_params
    summary_metrics["model_name"] = cfg.MODEL_NAME

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved standardized metrics to: {metrics_path}")

    # Generate classification reports
    intents = label_mappings["intents"]
    targets = label_mappings["targets"]
    speeds = label_mappings["speed_categories"]
    steps = label_mappings["step_counts"]

    rep_intent = classification_report(
        y_true_intent, y_pred_intent, labels=list(range(len(intents))), target_names=intents, zero_division=0
    )
    rep_target = classification_report(
        y_true_target, y_pred_target, labels=list(range(len(targets))), target_names=targets, zero_division=0
    )
    rep_speed = classification_report(
        y_true_speed, y_pred_speed, labels=list(range(len(speeds))), target_names=speeds, zero_division=0
    )
    rep_step = classification_report(
        y_true_step, y_pred_step, labels=list(range(len(steps))), target_names=[str(s) for s in steps], zero_division=0
    )

    with open(reports_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INTENT CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(rep_intent + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("TARGET CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(rep_target + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("SPEED CATEGORY CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(rep_speed + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("STEP COUNT CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(rep_step + "\n\n")

    print(f"Saved classification reports to: {reports_txt_path}")

    # Generate Confusion Matrices
    plot_and_save_confusion_matrix(
        y_true_intent,
        y_pred_intent,
        intents,
        "BanglaBERT-Base Intent Confusion Matrix",
        os.path.join(cm_dir, "intent_confusion_matrix.png"),
    )
    plot_and_save_confusion_matrix(
        y_true_speed,
        y_pred_speed,
        speeds,
        "BanglaBERT-Base Speed Category Confusion Matrix",
        os.path.join(cm_dir, "speed_category_confusion_matrix.png"),
    )
    plot_and_save_confusion_matrix(
        y_true_step,
        y_pred_step,
        [str(s) for s in steps],
        "BanglaBERT-Base Step Count Confusion Matrix",
        os.path.join(cm_dir, "step_count_confusion_matrix.png"),
    )
    print(f"Saved confusion matrices to: {cm_dir}")

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION SUMMARY: BanglaBERT-Base (110M)")
    print("=" * 80)
    print(f"Total Test Samples:             {summary_metrics['total_samples']}")
    print(f"JSON Syntax Validity:           {summary_metrics['json_validity_rate']*100:.2f}%")
    print(f"Intent Accuracy:                {summary_metrics['intent_accuracy']*100:.2f}%")
    print(f"Target Accuracy:                {summary_metrics['target_accuracy']*100:.2f}%")
    print(f"Speed Category Accuracy:        {summary_metrics['speed_category_accuracy']*100:.2f}%")
    print(f"Linear Speed Accuracy:          {summary_metrics['linear_speed_accuracy']*100:.2f}%")
    print(f"Linear Speed MAE:               {summary_metrics['linear_speed_mae']:.4f}")
    print(f"Angular Speed Accuracy:         {summary_metrics['angular_speed_accuracy']*100:.2f}%")
    print(f"Angular Speed MAE:              {summary_metrics['angular_speed_mae']:.4f}")
    print(f"Action Sequence Accuracy:       {summary_metrics['action_sequence_accuracy']*100:.2f}%")
    print(f"Step Count Accuracy:            {summary_metrics['step_count_accuracy']*100:.2f}%")
    print("-" * 80)
    print(f"Primary-Field Exact Match (4F): {summary_metrics['exact_match_4field']*100:.2f}%")
    print(f"Comprehensive Exact Match (7F): {summary_metrics['exact_match_7field']*100:.2f}%")
    print(f"Average Inference Latency:      {avg_latency:.2f} ms/sample")
    print(f"Total Model Parameters:         {total_params:,} (~{total_params / 1e6:.2f}M)")
    print("=" * 80)


if __name__ == "__main__":
    run_test_evaluation()
