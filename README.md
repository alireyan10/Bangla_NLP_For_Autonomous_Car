# Bangla NLP For Autonomous Car 🚗💨
### Natural Language Voice/Text Command-to-JSON Action Parsing Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Autonomous Car Core](https://img.shields.io/badge/%F0%9F%9A%97%20Hardware%20Repo-LLM__Based__Autunomous__Car-red)](https://github.com/alireyan10/LLM_Based_Autunomous_Car)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🔗 **Connected Project**: This repository serves as the **Natural Language Understanding (NLU) & Command-Parsing Brain** for the **[LLM_Based_Autunomous_Car](https://github.com/alireyan10/LLM_Based_Autunomous_Car)** project. It translates natural language Bangla instructions given by passengers or controllers into precise, structured 7-slot JSON action packets for real-time vehicular execution.

---

## 📌 Project Overview: What Does This System Do?

When operating an autonomous car or mobile robot in Bengali-speaking environments, users give unstructured, colloquial spoken or typed commands such as:
- *"রোবট একটু আস্তে সোজা গিয়ে ডানদিকে ঘুরে ড্রয়িং রুমের দরজার সামনে থামো"* (Robot, go straight a bit slowly, turn right, and stop in front of the drawing room door)
- *"গাড়ি ২০ মিটার দ্রুত গতিতে সামনে যাও"* (Car, go forward 20 meters at high speed)
- *"একদম বামে ঘুরে সোজা টেবিলের কাছে দাঁড়াও"* (Turn sharp left and stand straight near the table)

The vehicle's low-level microcontroller (e.g., Arduino, ROS2 node, STM32, Raspberry Pi) cannot parse raw Bengali sentences directly. **This repository provides the core AI engine** that ingests the raw/normalized Bangla text and outputs a strict, machine-executable **7-Field JSON Action Packet**:

```json
{
  "action": "MOVE",
  "target": "DOOR",
  "direction": "FORWARD",
  "distance": 20.0,
  "distance_unit": "m",
  "angle": 90.0,
  "angle_unit": "deg",
  "speed": "SLOW",
  "modifier": "GENTLE"
}
```

---

## 🔬 Architectural Paradigms & Research Benchmark

We explored, engineered, and rigorously evaluated two fundamentally distinct deep learning paradigms on a **frozen 600-sample test set (4,000 total verified samples)**:

1. **Generative Autoregressive LLM (`Qwen2.5-3B` via 4-bit QLoRA)**:
   - High-capacity reasoning engine for resolving implicit, multi-clause linguistic nuances.
   - Fine-tuned using ChatML prompt supervision with LoRA (Rank 64, Alpha 128).
2. **Discriminative Multi-Task Encoder (`BanglaBERT-Base` Multi-Head)**:
   - Ultra-lightweight 110M parameter pretrained transformer equipped with 7 task-specific classification and regression heads (116M total parameters).
   - Designed for sub-3ms ultra-low edge latency directly onboard low-cost robotics hardware without a GPU.

### 🏆 Master Benchmark Comparison

| Architecture | Model Variant | Input Mode | 7-Field Exact Match (Strict) | Inference Latency (Batch 1) | Footprint / Hardware Target |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Generative LLM** | **Qwen2.5-3B (QLoRA Exp 3)** | Preprocessed (Normalized) | **58.17%** | ~10,980 ms | High-Accuracy Server / Jetson Orin |
| **Generative LLM** | Qwen2.5-3B (QLoRA Exp 1) | Raw Natural | 55.83% | ~11,200 ms | Base LLM Baseline |
| **Multi-Head Encoder** | **BanglaBERT-Base** | Preprocessed Multi-Head | 29.17% (30.17% Clause-Aware) | **2.19 ms** | **Ultra-Fast Edge (~5,000x faster)** |
| **Multi-Head Encoder** | BanglaBERT-Base (Raw) | Raw Multi-Head | 29.50% | **1.99 ms** | Real-Time Edge Control |

> ⚡ **Latency vs Reasoning Trade-off**:
> - If your autonomous car has internet access or an onboard edge GPU: **Qwen2.5-3B** yields the highest semantic accuracy (**58.17% EM**).
> - If your autonomous car requires instant, obstacle-avoiding millisecond responsiveness without an expensive GPU: **Multi-Head BanglaBERT-Base** responds in **~2.19 milliseconds** (~5,000x faster than the LLM).

---

## 📂 Repository Layout

```text
Bangla_NLP_For_Autonomous_Car/
│
├── NEW dataset/                               # Primary curated dataset (4,000 rows, 14 columns)
│   ├── data/
│   │   ├── dataset_v2.csv                     # Cleaned CSV format
│   │   ├── dataset_v2.json                    # Cleaned JSON format
│   │   └── dataset_v2.xlsx                    # Cleaned Excel format
│   └── README.md
│
├── PREPROCESSING_EXPERIMENT/                  # Bangla linguistic normalizer & split generator
│   ├── scripts/
│   │   └── preprocess_dataset.py              # Rule-based Bangla normalizer & entity tagger
│   ├── evaluation/                            # Multi-field robust evaluation harness
│   │   ├── evaluator.py
│   │   └── robust_json_parser.py
│   └── output/
│       ├── dataset_v2_preprocessed.csv
│       └── splits/                            # Frozen 70/15/15 splits (Train: 2800, Val: 600, Test: 600)
│           ├── train.jsonl
│           ├── validation.jsonl
│           └── test.jsonl
│
├── training/
│   ├── non_llm_banglabert_base/               # Champion Non-LLM Multi-Head Architecture
│   │   ├── config.py                          # Hyperparameters & slot configurations
│   │   ├── dataset.py                         # Multi-target PyTorch tensor dataset
│   │   ├── model.py                           # BanglaBERT-Base backbone + 7 multi-task heads
│   │   ├── train.py                           # Multi-loss trainer
│   │   ├── evaluate.py                        # Strict 7-field EM & per-slot accuracy evaluator
│   │   ├── evaluate_improved.py               # Clause-aware parser (Boosts EM to 30.17%)
│   │   ├── inference.py                       # Single command inference & latency benchmark
│   │   ├── json_formatter.py                  # Predictor to structured JSON formatter
│   │   ├── evaluation/                        # Test predictions & classification reports
│   │   └── reports/                           # Confusion matrices & benchmark charts
│   │
│   ├── non_llm_banglabert_base_raw/           # Raw baseline for non-LLM ablation
│   │   ├── ...                                # Identical structure trained on raw commands
│   │   └── evaluation/
│   │
│   └── qwen2.5_3b/                            # Generative LLM Experiments
│       ├── experiment_1_raw/                  # QLoRA Baseline (Raw commands, 55.83% EM)
│       │   ├── data/                          # ChatML formatted splits
│       │   ├── scripts/                       # Train & inference scripts
│       │   ├── evaluation/                    # Metrics summary (metrics_summary.json)
│       │   └── report/                        # Training curves & loss histories
│       └── experiment_3_optimized_lora/       # Champion LLM (Rank 64, Preprocessed, 58.17% EM)
│           ├── data/                          # Preprocessed ChatML splits
│           ├── scripts/                       # Optimized training & inference scripts
│           ├── evaluation/                    # Predictions & metrics summary
│           └── reports/                       # Validation logs
│
├── Voice/                                     # Voice-to-Action module (STT + Inference)
│   ├── 1.mp4, 2.mp4, 3.mp4                    # Raw recorded voice samples
│   ├── 1.wav, 2.wav, 3.wav                    # Converted 16kHz mono audio
│   └── voice_to_action_pipeline.py            # End-to-end voice-to-JSON action script
│
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git rules (excludes heavy checkpoints >100MB)
└── README.md                                  # Project overview and documentation
```

---

## 🛠️ Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/alireyan10/Bangla_NLP_For_Autonomous_Car.git
cd Bangla_NLP_For_Autonomous_Car
pip install -r requirements.txt
```

### 2. Run Edge Inference (BanglaBERT-Base)
```bash
cd training/non_llm_banglabert_base
python inference.py --text "রোবট সোজা গিয়ে টেবিলের কাছে থামো"
```
**Output:**
```json
{
  "action": "MOVE",
  "target": "TABLE",
  "direction": "FORWARD",
  "distance": "NONE",
  "distance_unit": "NONE",
  "angle": "NONE",
  "angle_unit": "NONE",
  "speed": "NORMAL",
  "modifier": "NONE",
  "latency_ms": 2.19
}
```

### 3. Run Preprocessing Pipeline
```bash
cd PREPROCESSING_EXPERIMENT/scripts
python preprocess_dataset.py
```

### 4. Run Voice-to-Action Pipeline (Audio -> Bangla Text -> Robot JSON)
```bash
cd Voice
python voice_to_action_pipeline.py --input "1.mp4"
```

---

## 🤝 Integration with Autonomous Car Hardware

To integrate this NLP pipeline with the [LLM_Based_Autunomous_Car](https://github.com/alireyan10/LLM_Based_Autunomous_Car) controller:
1. Stream voice audio from the car's microphone -> Speech-to-Text (STT) -> Bangla text string.
2. Pass Bangla string into `inference.py`.
3. Send the output JSON payload over Serial / MQTT / WebSocket to the vehicle's motor controller for autonomous navigation.

---

## 📄 License & Attribution
Developed by **[Ali Reyan](https://github.com/alireyan10)** as part of research on Embedded Autonomous Robotics and Natural Language Semantic Parsing in Bengali.
Licensed under the [MIT License](LICENSE).
