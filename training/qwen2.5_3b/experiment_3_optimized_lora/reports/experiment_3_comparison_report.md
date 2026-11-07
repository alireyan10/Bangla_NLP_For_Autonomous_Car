# 4-Way Experimental Benchmark Comparison Report

**Project:** Bangla Voice Command to Robot Action Mapping (Autonomous Navigation)  
**Base Model:** Qwen2.5-3B (4-bit NF4 Quantization)  
**Evaluation Standard:** 600 Strictly Frozen Test Samples (IDs 3401-4000), Tolerance $\Delta \le 10^{-3}$  

---

## 1. Executive Summary & Comparative Matrix

| Metric | Exp 1 (Raw) | Exp 2 (Preprocessed) | Exp 2 (Few-Shot) | Exp 3 (Optimized LoRA) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Test Samples** | 600 | 600 | 600 | 600 |
| **JSON Validity Rate** | 100.00% | 100.00% | 99.67% | 100.00% |
| **Format Errors** | 0 | 0 | 2 | 0 |
| **Intent Accuracy** | 86.17% | 86.67% | 82.50% | 88.67% |
| **Target Accuracy** | 99.33% | 99.50% | 93.50% | 99.17% |
| **Speed Category Accuracy** | 97.33% | 97.33% | 95.83% | 97.67% |
| **Action Sequence Accuracy** | 62.17% | 62.83% | 61.50% | 63.83% |
| **Step Count Accuracy** | 100.00% | 100.00% | 98.00% | 100.00% |
| **Linear Speed Acc (<=1e-3)** | 94.33% | 91.83% | 89.00% | 94.67% |
| **Angular Speed Acc (<=1e-3)** | 98.50% | 98.17% | 86.50% | 98.83% |
| **Linear Speed MAE (m/s)** | 0.0040 | 0.0041 | 0.0064 | 0.0039 |
| **Angular Speed MAE (rad/s)** | 0.0021 | 0.0031 | 0.0117 | 0.0020 |
| **Primary 4-Field Exact Match** | 58.83% | 59.67% | 53.83% | 61.17% |
| **Comprehensive 7-Field EM** | 55.83% | 54.00% | 44.50% | 58.17% |


---

## 2. Training Hyperparameters & Computational Footprint

| Hyperparameter / Resource | Exp 1 (Raw) | Exp 2 (Preprocessed) | Exp 2 (Few-Shot) | Exp 3 (Optimized LoRA) |
| :--- | :---: | :---: | :---: | :---: |
| **LoRA Configuration** | r=16, alpha=32 | r=16, alpha=32 | r=16, alpha=32 | r=32, alpha=64 |
| **LR Scheduler** | Linear / Const | Linear / Const | N/A (Inference) | Cosine Annealing (Warmup=25) |
| **Training Loss** | N/A | 0.3988 | N/A | Pending |
| **Best Validation Loss** | N/A | 0.3701 | N/A | Pending |
| **Peak GPU VRAM (GB)** | ~3.2 GB | 3.386 GB | N/A | N/A GB |
| **Training Time** | ~40 min | 319.1 min | N/A | 0.0 min |


---

## 3. Methodological Integrity & Attribution Note

> [!NOTE]
> **Scientific Attribution (Label Disambiguation + LoRA Optimization):**
> Experiment 3 incorporates two simultaneous improvements over Experiment 2:
> 1. **Label Disambiguation:** 468 conflicting training/validation samples across 97 duplicate command phrases were resolved using consistent linguistic ontology rules (steady slow motion -> `MOVE_SLOW`, steady fast motion -> `MOVE_FAST`, in-place spin verbs -> `ROTATE`, steering turn verbs -> `TURN`).
> 2. **LoRA Capacity & Scheduler Tuning:** LoRA rank was doubled from $r=16$ to $r=32$ ($lpha=64$), target modules expanded across all 7 linear projections, and learning rate scheduled with Cosine decay and 25 warmup steps.
>
> Therefore, this configuration is accurately reported as **"Label Disambiguation + LoRA Optimization ($r=32$)"** rather than purely LoRA hyperparameter tuning. The benchmark test set (IDs 3401-4000) was maintained strictly frozen and unedited to preserve empirical validity.
