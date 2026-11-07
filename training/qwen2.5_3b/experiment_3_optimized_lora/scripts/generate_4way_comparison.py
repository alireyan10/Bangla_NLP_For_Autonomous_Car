"""
Comprehensive 4-Way Experimental Comparison Report Generator.

Compares:
1. Experiment 1: Raw Dataset Baseline (r=16, alpha=32)
2. Experiment 2: Light Preprocessed Baseline (r=16, alpha=32)
3. Experiment 2: Few-Shot In-Context Prompting
4. Experiment 3: Label Disambiguation + LoRA Optimization (r=32, alpha=64, Cosine LR, Warmup=25)

Generates:
  - training/qwen2.5_3b/experiment_3_optimized_lora/reports/experiment_3_comparison_report.md
"""

import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"C:\Users\Reyan\Downloads\Case_Study"
EXP3_DIR = os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_3_optimized_lora")
REPORTS_DIR = os.path.join(EXP3_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

AUDIT_RESULTS_PATH = os.path.join(PROJECT_ROOT, "PREPROCESSING_EXPERIMENT", "output", "audit_evaluation_results.json")
EXP2_BASELINE_METRICS_PATH = os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_2_light_preprocessed", "evaluation", "metrics.json")
EXP2_FEW_SHOT_METRICS_PATH = os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_2_light_preprocessed", "evaluation_few_shot", "metrics_few_shot.json")
EXP3_METRICS_PATH = os.path.join(EXP3_DIR, "evaluation", "metrics.json")
EXP3_SUMMARY_PATH = os.path.join(REPORTS_DIR, "training_summary.json")
EXP2_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_2_light_preprocessed", "reports", "training_summary.json")

OUTPUT_MD_PATH = os.path.join(REPORTS_DIR, "experiment_3_comparison_report.md")


def fmt_pct(val):
    if val is None:
        return "Pending"
    return f"{val * 100:.2f}%"


def fmt_num(val, dec=4):
    if val is None:
        return "Pending"
    return f"{val:.{dec}f}"


def fmt_int(val):
    if val is None:
        return "Pending"
    return str(val)


def main():
    print("=" * 80)
    print("GENERATING 4-WAY EXPERIMENTAL COMPARISON REPORT")
    print("=" * 80)

    # 1. Load Exp 1 metrics from audit
    with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    exp1_metrics = audit_data["experiment_1_raw"]["reevaluated_robust"]

    # 2. Load Exp 2 baseline metrics
    with open(EXP2_BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
        exp2_baseline = json.load(f)

    # 3. Load Exp 2 few-shot metrics
    with open(EXP2_FEW_SHOT_METRICS_PATH, "r", encoding="utf-8") as f:
        exp2_few_shot = json.load(f)

    # 4. Load Exp 3 metrics (if available)
    exp3_metrics = None
    if os.path.exists(EXP3_METRICS_PATH):
        with open(EXP3_METRICS_PATH, "r", encoding="utf-8") as f:
            exp3_metrics = json.load(f)

    # Load training summaries
    exp2_summary = {}
    if os.path.exists(EXP2_SUMMARY_PATH):
        with open(EXP2_SUMMARY_PATH, "r", encoding="utf-8") as f:
            exp2_summary = json.load(f)

    exp3_summary = {}
    if os.path.exists(EXP3_SUMMARY_PATH):
        with open(EXP3_SUMMARY_PATH, "r", encoding="utf-8") as f:
            exp3_summary = json.load(f)

    rows = [
        ("Total Test Samples", "600", "600", "600", "600"),
        ("JSON Validity Rate", fmt_pct(exp1_metrics["json_validity_rate"]), fmt_pct(exp2_baseline["json_validity_rate"]), fmt_pct(exp2_few_shot["json_validity_rate"]), fmt_pct(exp3_metrics["json_validity_rate"] if exp3_metrics else None)),
        ("Format Errors", fmt_int(exp1_metrics["format_error_count"]), fmt_int(exp2_baseline["format_error_count"]), fmt_int(exp2_few_shot["format_error_count"]), fmt_int(exp3_metrics["format_error_count"] if exp3_metrics else None)),
        ("Intent Accuracy", fmt_pct(exp1_metrics["intent_accuracy"]), fmt_pct(exp2_baseline["intent_accuracy"]), fmt_pct(exp2_few_shot["intent_accuracy"]), fmt_pct(exp3_metrics["intent_accuracy"] if exp3_metrics else None)),
        ("Target Accuracy", fmt_pct(exp1_metrics["target_accuracy"]), fmt_pct(exp2_baseline["target_accuracy"]), fmt_pct(exp2_few_shot["target_accuracy"]), fmt_pct(exp3_metrics["target_accuracy"] if exp3_metrics else None)),
        ("Speed Category Accuracy", fmt_pct(exp1_metrics["speed_category_accuracy"]), fmt_pct(exp2_baseline["speed_category_accuracy"]), fmt_pct(exp2_few_shot["speed_category_accuracy"]), fmt_pct(exp3_metrics["speed_category_accuracy"] if exp3_metrics else None)),
        ("Action Sequence Accuracy", fmt_pct(exp1_metrics["action_sequence_accuracy"]), fmt_pct(exp2_baseline["action_sequence_accuracy"]), fmt_pct(exp2_few_shot["action_sequence_accuracy"]), fmt_pct(exp3_metrics["action_sequence_accuracy"] if exp3_metrics else None)),
        ("Step Count Accuracy", fmt_pct(exp1_metrics["step_count_accuracy"]), fmt_pct(exp2_baseline["step_count_accuracy"]), fmt_pct(exp2_few_shot["step_count_accuracy"]), fmt_pct(exp3_metrics["step_count_accuracy"] if exp3_metrics else None)),
        ("Linear Speed Acc (<=1e-3)", fmt_pct(exp1_metrics["linear_speed_accuracy"]), fmt_pct(exp2_baseline["linear_speed_accuracy"]), fmt_pct(exp2_few_shot["linear_speed_accuracy"]), fmt_pct(exp3_metrics["linear_speed_accuracy"] if exp3_metrics else None)),
        ("Angular Speed Acc (<=1e-3)", fmt_pct(exp1_metrics["angular_speed_accuracy"]), fmt_pct(exp2_baseline["angular_speed_accuracy"]), fmt_pct(exp2_few_shot["angular_speed_accuracy"]), fmt_pct(exp3_metrics["angular_speed_accuracy"] if exp3_metrics else None)),
        ("Linear Speed MAE (m/s)", fmt_num(exp1_metrics["linear_speed_mae"]), fmt_num(exp2_baseline["linear_speed_mae"]), fmt_num(exp2_few_shot["linear_speed_mae"]), fmt_num(exp3_metrics["linear_speed_mae"] if exp3_metrics else None)),
        ("Angular Speed MAE (rad/s)", fmt_num(exp1_metrics["angular_speed_mae"]), fmt_num(exp2_baseline["angular_speed_mae"]), fmt_num(exp2_few_shot["angular_speed_mae"]), fmt_num(exp3_metrics["angular_speed_mae"] if exp3_metrics else None)),
        ("Primary 4-Field Exact Match", fmt_pct(exp1_metrics["exact_match_4field"]), fmt_pct(exp2_baseline["exact_match_4field"]), fmt_pct(exp2_few_shot["exact_match_4field"]), fmt_pct(exp3_metrics["exact_match_4field"] if exp3_metrics else None)),
        ("Comprehensive 7-Field EM", fmt_pct(exp1_metrics["exact_match_7field"]), fmt_pct(exp2_baseline["exact_match_7field"]), fmt_pct(exp2_few_shot["exact_match_7field"]), fmt_pct(exp3_metrics["exact_match_7field"] if exp3_metrics else None)),
    ]

    table_md = "| Metric | Exp 1 (Raw) | Exp 2 (Preprocessed) | Exp 2 (Few-Shot) | Exp 3 (Optimized LoRA) |\n"
    table_md += "| :--- | :---: | :---: | :---: | :---: |\n"
    for r in rows:
        table_md += f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n"

    # Training resources comparison
    res_rows = [
        ("LoRA Configuration", "r=16, alpha=32", "r=16, alpha=32", "r=16, alpha=32", "r=32, alpha=64"),
        ("LR Scheduler", "Linear / Const", "Linear / Const", "N/A (Inference)", "Cosine Annealing (Warmup=25)"),
        ("Training Loss", "N/A", fmt_num(exp2_summary.get("training_loss"), 4), "N/A", fmt_num(exp3_summary.get("training_loss"), 4)),
        ("Best Validation Loss", "N/A", fmt_num(exp2_summary.get("best_validation_loss"), 4), "N/A", fmt_num(exp3_summary.get("best_validation_loss"), 4)),
        ("Peak GPU VRAM (GB)", "~3.2 GB", f"{exp2_summary.get('peak_gpu_memory_gb', 'N/A')} GB", "N/A", f"{exp3_summary.get('peak_gpu_memory_gb', 'N/A')} GB"),
        ("Training Time", "~40 min", f"{exp2_summary.get('training_seconds', 0)/60:.1f} min", "N/A", f"{exp3_summary.get('training_seconds', 0)/60:.1f} min"),
    ]

    res_table_md = "| Hyperparameter / Resource | Exp 1 (Raw) | Exp 2 (Preprocessed) | Exp 2 (Few-Shot) | Exp 3 (Optimized LoRA) |\n"
    res_table_md += "| :--- | :---: | :---: | :---: | :---: |\n"
    for r in res_rows:
        res_table_md += f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n"

    report_content = f"""# 4-Way Experimental Benchmark Comparison Report

**Project:** Bangla Voice Command to Robot Action Mapping (Autonomous Navigation)  
**Base Model:** Qwen2.5-3B (4-bit NF4 Quantization)  
**Evaluation Standard:** 600 Strictly Frozen Test Samples (IDs 3401-4000), Tolerance $\Delta \le 10^{{-3}}$  

---

## 1. Executive Summary & Comparative Matrix

{table_md}

---

## 2. Training Hyperparameters & Computational Footprint

{res_table_md}

---

## 3. Methodological Integrity & Attribution Note

> [!NOTE]
> **Scientific Attribution (Label Disambiguation + LoRA Optimization):**
> Experiment 3 incorporates two simultaneous improvements over Experiment 2:
> 1. **Label Disambiguation:** 468 conflicting training/validation samples across 97 duplicate command phrases were resolved using consistent linguistic ontology rules (steady slow motion -> `MOVE_SLOW`, steady fast motion -> `MOVE_FAST`, in-place spin verbs -> `ROTATE`, steering turn verbs -> `TURN`).
> 2. **LoRA Capacity & Scheduler Tuning:** LoRA rank was doubled from $r=16$ to $r=32$ ($\alpha=64$), target modules expanded across all 7 linear projections, and learning rate scheduled with Cosine decay and 25 warmup steps.
>
> Therefore, this configuration is accurately reported as **"Label Disambiguation + LoRA Optimization ($r=32$)"** rather than purely LoRA hyperparameter tuning. The benchmark test set (IDs 3401-4000) was maintained strictly frozen and unedited to preserve empirical validity.
"""

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report saved to: {OUTPUT_MD_PATH}")
    print("\n" + table_md)


if __name__ == "__main__":
    main()
