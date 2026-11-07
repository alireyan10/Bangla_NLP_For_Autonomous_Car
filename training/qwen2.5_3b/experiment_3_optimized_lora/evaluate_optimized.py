"""
Evaluation Script for Experiment 3: Qwen2.5-3B + Optimized LoRA (r=32).

Strictly evaluates on the frozen benchmark test split:
  PREPROCESSING_EXPERIMENT/output/splits/test.jsonl (IDs 3401-4000, 600 samples)

Outputs:
  - evaluation/predictions.jsonl
  - evaluation/metrics.json
"""

import os
import sys
import time
import json

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config_optimized as cfg
from PREPROCESSING_EXPERIMENT.evaluation.robust_json_parser import parse_robot_json
from PREPROCESSING_EXPERIMENT.evaluation.evaluator import evaluate_dataset


class JsonStoppingCriteria(StoppingCriteria):
    """
    Halts token generation the moment a valid, closed JSON object is detected.
    """
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "}" in text:
            parsed, is_valid, _ = parse_robot_json(text)
            if is_valid and isinstance(parsed, dict):
                return True
        return False


def load_test_split():
    """Load preprocessed test split records (IDs 3401-4000)."""
    if not os.path.exists(cfg.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test split file missing: {cfg.TEST_DATA_PATH}")

    items = []
    with open(cfg.TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} frozen test samples from: {cfg.TEST_DATA_PATH}")
    return items


def main():
    print("=" * 80)
    print("STARTING EVALUATION: EXPERIMENT 3 (OPTIMIZED LORA & DISAMBIGUATED)")
    print("=" * 80)

    cfg.ensure_directories()
    predictions_file = os.path.join(cfg.EVAL_DIR, "predictions.jsonl")
    metrics_file = os.path.join(cfg.EVAL_DIR, "metrics.json")

    if not os.path.exists(cfg.ADAPTER_DIR):
        print(f"❌ ERROR: Trained adapter not found at: {cfg.ADAPTER_DIR}")
        print("Please run train_qlora_optimized.py first.")
        sys.exit(1)

    existing_ids = set()
    if os.path.exists(predictions_file):
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record["id"])
                    except Exception:
                        pass
        print(f"Found existing predictions.jsonl with {len(existing_ids)} completed samples.")

    print(f"\nLoading tokenizer: {cfg.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    im_end_id = tokenizer.convert_tokens_to_ids(cfg.IM_END_TOKEN)
    eos_token_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)
    print(f"Active EOS token IDs: {eos_token_ids} (including <|im_end|>)")

    print("\nConfiguring 4-bit quantization for base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.LOAD_IN_4BIT,
        bnb_4bit_quant_type=cfg.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg.BNB_4BIT_USE_DOUBLE_QUANT
    )

    print(f"Loading base model: {cfg.MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"Attaching fine-tuned LoRA adapter from: {cfg.ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, cfg.ADAPTER_DIR)
    model.eval()

    test_samples = load_test_split()
    samples_to_evaluate = [s for s in test_samples if s["ID"] not in existing_ids]

    if not samples_to_evaluate:
        print("All test samples have already been evaluated.")
    else:
        print(f"Running inference on {len(samples_to_evaluate)} pending samples...")
        total_latency = 0.0

        with open(predictions_file, "a", encoding="utf-8") as out_f:
            for sample in tqdm(samples_to_evaluate, desc="Evaluating"):
                sample_id = sample["ID"]
                cmd = str(sample["Bangla Command"]).strip()

                seq_raw = sample.get("Action Sequence", "")
                if isinstance(seq_raw, list):
                    action_seq = seq_raw
                else:
                    seq_str = str(seq_raw).strip()
                    action_seq = [s.strip() for s in seq_str.split("->")] if "->" in seq_str else [seq_str]

                gt_dict = {
                    "intent": str(sample.get("Intent", "")).strip(),
                    "target": str(sample.get("Target", "")).strip(),
                    "speed_category": str(sample.get("Speed Category", "")).strip(),
                    "linear_speed": float(sample.get("Linear Speed (m/s)", 0.0)),
                    "angular_speed": float(sample.get("Angular Speed (rad/s)", 0.0)),
                    "action_sequence": action_seq,
                    "step_count": int(sample.get("Step Count", 1))
                }

                messages = [
                    {"role": "system", "content": cfg.SYSTEM_PROMPT},
                    {"role": "user", "content": cmd}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
                prompt_len = inputs.input_ids.shape[1]

                stopping_criteria = StoppingCriteriaList([
                    JsonStoppingCriteria(tokenizer, prompt_len)
                ])

                start_t = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=cfg.MAX_NEW_TOKENS,
                        temperature=cfg.TEMPERATURE,
                        do_sample=cfg.DO_SAMPLE,
                        eos_token_id=eos_token_ids,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria
                    )
                latency = time.time() - start_t
                total_latency += latency

                gen_tokens = outputs[0][prompt_len:]
                prediction_str = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                result_entry = {
                    "id": sample_id,
                    "command": cmd,
                    "ground_truth": gt_dict,
                    "prediction_raw": prediction_str,
                    "latency": latency
                }

                out_f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                out_f.flush()

        print(f"Inference complete. Results saved to: {predictions_file}")

    print("\nComputing standardized metrics across all 600 test predictions...")
    all_predictions = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_predictions.append(json.loads(line))

    eval_items = []
    for p in all_predictions:
        eval_items.append({
            "id": p["id"],
            "command": p["command"],
            "ground_truth": p["ground_truth"],
            "raw_prediction": p["prediction_raw"]
        })

    summary_metrics, detailed_records = evaluate_dataset(eval_items, tolerance=cfg.NUMERIC_TOLERANCE)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_file}")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY (EXPERIMENT 3 - OPTIMIZED LORA & DISAMBIGUATED)")
    print("=" * 80)
    print(f"Total Test Samples:             {summary_metrics['total_samples']}")
    print(f"JSON Validity Rate:             {summary_metrics['json_validity_rate'] * 100:.2f}% ({summary_metrics['valid_json_count']}/600)")
    print(f"Format Errors:                  {summary_metrics['format_error_count']}")
    print(f"Intent Accuracy:                {summary_metrics['intent_accuracy'] * 100:.2f}%")
    print(f"Target Accuracy:                {summary_metrics['target_accuracy'] * 100:.2f}%")
    print(f"Speed Category Accuracy:        {summary_metrics['speed_category_accuracy'] * 100:.2f}%")
    print(f"Action Sequence Accuracy:       {summary_metrics['action_sequence_accuracy'] * 100:.2f}%")
    print(f"Step Count Accuracy:            {summary_metrics['step_count_accuracy'] * 100:.2f}%")
    print(f"Linear Speed Accuracy (<=1e-3): {summary_metrics['linear_speed_accuracy'] * 100:.2f}% (MAE: {summary_metrics['linear_speed_mae']:.4f} m/s)")
    print(f"Angular Speed Accuracy (<=1e-3):{summary_metrics['angular_speed_accuracy'] * 100:.2f}% (MAE: {summary_metrics['angular_speed_mae']:.4f} rad/s)")
    print("-" * 80)
    print(f"PRIMARY 4-FIELD EXACT MATCH:    {summary_metrics['exact_match_4field'] * 100:.2f}% ({summary_metrics['exact_match_4field_count']}/600)")
    print(f"COMPREHENSIVE 7-FIELD EM:       {summary_metrics['exact_match_7field'] * 100:.2f}% ({summary_metrics['exact_match_7field_count']}/600)")
    print("=" * 80)


if __name__ == "__main__":
    main()
