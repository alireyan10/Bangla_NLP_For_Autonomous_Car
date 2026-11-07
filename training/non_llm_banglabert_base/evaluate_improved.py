"""
Improved Evaluation Script for Non-LLM BanglaBERT-Base (110M) Pipeline.
Technique: Hierarchical Clause-Aware Action Sequence Prediction + Ontology Alignment.
Evaluates on the exact same 600 frozen test samples (IDs 3401-4000).
"""

import os
import sys
import json
import time
import io
import re

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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


def split_bangla_clauses(text: str) -> list:
    """
    Splits compound multi-step Bangla commands on sequential connectors:
    - তারপর (then)
    - এবং (and)
    - Comma separator (,)
    """
    parts = re.split(r"\,?\s*(?:তারপর|এবং|\,)\s*", str(text))
    clauses = [p.strip() for p in parts if len(p.strip()) > 3]
    return clauses if clauses else [str(text).strip()]


def predict_single_clause(model, tokenizer, clause_text, label_mappings, device):
    """Predicts the atomic action of an individual clause."""
    norm_cmd = cfg.normalize_bangla_text(clause_text)
    encoding = tokenizer(
        norm_cmd,
        max_length=cfg.MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    intent_idx = torch.argmax(outputs["intent_logits"], dim=-1).item()
    s1_idx = torch.argmax(outputs["step1_logits"], dim=-1).item()

    intents = label_mappings["intents"]
    atomic_actions = label_mappings["atomic_actions"]

    pred_intent = intents[intent_idx]
    if pred_intent != "SEQUENTIAL":
        return pred_intent
    else:
        return atomic_actions[s1_idx]


def run_improved_evaluation():
    print("=" * 80)
    print("RUNNING IMPROVED NON-LLM EVALUATION: Hierarchical Clause-Aware BanglaBERT-Base (110M)")
    print("=" * 80)

    cfg.ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    label_mappings = checkpoint["label_mappings"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = BanglaBERTMultiHead(cfg.MODEL_NAME, label_mappings)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = BanglaRobotDataset(
        cfg.TEST_DATA_PATH, tokenizer, label_mappings, max_seq_length=cfg.MAX_SEQ_LENGTH
    )
    print(f"Loaded {len(test_dataset)} frozen test samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=bangla_robot_collate_fn,
    )

    eval_items = []
    latencies = []

    print("\nRunning Hierarchical Clause-Aware Inference...")
    for batch in tqdm(test_loader, desc="Clause-Aware Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        pred_intent = torch.argmax(outputs["intent_logits"], dim=-1).cpu().numpy()
        pred_target = torch.argmax(outputs["target_logits"], dim=-1).cpu().numpy()
        pred_speed_cat = torch.argmax(outputs["speed_category_logits"], dim=-1).cpu().numpy()
        pred_lin = torch.argmax(outputs["linear_speed_logits"], dim=-1).cpu().numpy()
        pred_ang = torch.argmax(outputs["angular_speed_logits"], dim=-1).cpu().numpy()

        bs = input_ids.size(0)
        for i in range(bs):
            cmd_text = batch["command"][i]
            intent_id = int(pred_intent[i])
            target_id = int(pred_target[i])
            speed_cat_id = int(pred_speed_cat[i])
            lin_id = int(pred_lin[i])
            ang_id = int(pred_ang[i])

            intent = label_mappings["intents"][intent_id]
            target = label_mappings["targets"][target_id]
            speed_cat = label_mappings["speed_categories"][speed_cat_id]
            lin_speed = float(label_mappings["linear_speeds"][lin_id])
            ang_speed = float(label_mappings["angular_speeds"][ang_id])

            clauses = split_bangla_clauses(cmd_text)

            if len(clauses) > 1:
                derived_intent = "SEQUENTIAL"
                clause_actions = [
                    predict_single_clause(model, tokenizer, c, label_mappings, device)
                    for c in clauses
                ]
                final_action_seq = [a for a in clause_actions if a != "NONE"]
                if not final_action_seq:
                    final_action_seq = [intent]
                final_step_count = len(final_action_seq)
            else:
                if intent == "SEQUENTIAL":
                    clause_actions = [
                        predict_single_clause(model, tokenizer, cmd_text, label_mappings, device)
                    ]
                    final_action_seq = [a for a in clause_actions if a != "NONE"]
                    if not final_action_seq:
                        final_action_seq = ["MOVE_FORWARD", "STOP"]
                    final_step_count = len(final_action_seq)
                else:
                    derived_intent = intent
                    final_action_seq = [intent]
                    final_step_count = 1

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000 / bs)

            action_plan = {
                "intent": derived_intent if len(clauses) > 1 else intent,
                "target": target,
                "speed_category": speed_cat,
                "linear_speed": lin_speed,
                "angular_speed": ang_speed,
                "action_sequence": final_action_seq,
                "step_count": final_step_count,
            }
            json_str = json.dumps(action_plan, ensure_ascii=False, indent=2)

            gt_item = {
                "id": batch["id"][i],
                "command": cmd_text,
                "ground_truth": batch["ground_truth"][i],
                "raw_prediction": json_str,
            }
            eval_items.append(gt_item)

    # Compute official standardized metrics
    summary_metrics, detailed_results = evaluate_dataset(eval_items, tolerance=cfg.NUMERIC_TOLERANCE)
    summary_metrics["average_latency_ms"] = float(np.mean(latencies))
    summary_metrics["model_parameters"] = sum(p.numel() for p in model.parameters())

    improved_metrics_path = os.path.join(cfg.EVAL_DIR, "improved_metrics.json")
    with open(improved_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved improved metrics to: {improved_metrics_path}")

    print("\n" + "=" * 80)
    print("HIERARCHICAL CLAUSE-AWARE EVALUATION SUMMARY: BanglaBERT-Base (110M)")
    print("=" * 80)
    print(f"Total Test Samples:             {summary_metrics['total_samples']}")
    print(f"JSON Syntax Validity:           {summary_metrics['json_validity_rate']*100:.2f}%")
    print(f"Intent Accuracy:                {summary_metrics['intent_accuracy']*100:.2f}%")
    print(f"Target Accuracy:                {summary_metrics['target_accuracy']*100:.2f}%")
    print(f"Speed Category Accuracy:        {summary_metrics['speed_category_accuracy']*100:.2f}%")
    print(f"Linear Speed Accuracy:          {summary_metrics['linear_speed_accuracy']*100:.2f}%")
    print(f"Angular Speed Accuracy:         {summary_metrics['angular_speed_accuracy']*100:.2f}%")
    print(f"Action Sequence Accuracy:       {summary_metrics['action_sequence_accuracy']*100:.2f}%")
    print(f"Step Count Accuracy:            {summary_metrics['step_count_accuracy']*100:.2f}%")
    print("-" * 80)
    print(f"Primary-Field Exact Match (4F): {summary_metrics['exact_match_4field']*100:.2f}%")
    print(f"Comprehensive Exact Match (7F): {summary_metrics['exact_match_7field']*100:.2f}%")
    print(f"Average Inference Latency:      {summary_metrics['average_latency_ms']:.2f} ms/sample")
    print("=" * 80)


if __name__ == "__main__":
    run_improved_evaluation()
