"""
Interactive & Batch Inference Pipeline for Bangla Robot Commands using BanglaBERT-Base (110M).
Accepts raw Bangla commands, runs multi-head prediction, formats into JSON, and validates.
"""

import os
import sys
import json
import time
import io
from typing import Dict, Any, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from model import BanglaBERTMultiHead
from json_formatter import assemble_robot_json


class BanglaBERTBaseInferencePipeline:
    def __init__(self, checkpoint_path: str = None, device: str = None):
        if checkpoint_path is None:
            checkpoint_path = cfg.BEST_MODEL_PATH

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.label_mappings = checkpoint["label_mappings"]

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        self.model = BanglaBERTMultiHead(cfg.MODEL_NAME, self.label_mappings)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, command: str, use_setpoints: bool = True) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Processes a raw Bangla robot command and returns structured JSON and validation status.
        """
        start_t = time.time()
        norm_cmd = cfg.normalize_bangla_text(command)

        if not norm_cmd.strip():
            empty_plan = {
                "intent": "STOP",
                "target": "none",
                "speed_category": "STOP",
                "linear_speed": 0.0,
                "angular_speed": 0.0,
                "action_sequence": ["STOP"],
                "step_count": 1,
            }
            return json.dumps(empty_plan, indent=2), empty_plan, {
                "is_valid_json": True,
                "syntax_error": None,
                "has_unsupported_target": False,
                "physical_limits_ok": True,
                "semantic_warning": "Empty input command, defaulted to STOP",
            }

        encoding = self.tokenizer(
            norm_cmd,
            max_length=cfg.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        pred_dict = {
            "intent_id": torch.argmax(outputs["intent_logits"], dim=-1).item(),
            "target_id": torch.argmax(outputs["target_logits"], dim=-1).item(),
            "speed_cat_id": torch.argmax(outputs["speed_category_logits"], dim=-1).item(),
            "step_count_id": torch.argmax(outputs["step_count_logits"], dim=-1).item(),
            "step1_id": torch.argmax(outputs["step1_logits"], dim=-1).item(),
            "step2_id": torch.argmax(outputs["step2_logits"], dim=-1).item(),
            "step3_id": torch.argmax(outputs["step3_logits"], dim=-1).item(),
            "step4_id": torch.argmax(outputs["step4_logits"], dim=-1).item(),
            "linear_speed_class_id": torch.argmax(outputs["linear_speed_logits"], dim=-1).item(),
            "angular_speed_class_id": torch.argmax(outputs["angular_speed_logits"], dim=-1).item(),
            "linear_speed_reg": outputs["linear_reg"].item(),
            "angular_speed_reg": outputs["angular_reg"].item(),
        }

        json_str, action_plan, status = assemble_robot_json(
            pred_dict, self.label_mappings, use_setpoint_speeds=use_setpoints
        )
        status["inference_latency_sec"] = time.time() - start_t
        return json_str, action_plan, status


def run_demo_suite():
    print("=" * 80)
    print("DEMO INFERENCE SUITE: Testing BanglaBERT-Base (110M) on Diverse Robot Commands")
    print("=" * 80)

    pipeline = BanglaBERTBaseInferencePipeline()

    test_cases = [
        "রোবট ধীরে টেবিলের কাছে যাও",
        "সামনে দ্রুত গতিতে এগোও",
        "বামে আস্তে করে টার্ন নাও",
        "সামনে যাও, তারপর বামে ঘুরে থেমে যাও",
        "কালকে সকালে আকাশ কেমন হবে",
        "জরুরি ভিত্তিতে ব্রেক করো এবং থামো",
    ]

    for idx, cmd in enumerate(test_cases, 1):
        print(f"\n--- [Test Case {idx}] Input Command: '{cmd}' ---")
        json_output, parsed, status = pipeline.predict(cmd)
        print("Generated JSON Output:")
        print(json_output)
        print("Validation Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_cmd = " ".join(sys.argv[1:])
        pipeline = BanglaBERTBaseInferencePipeline()
        json_res, _, val_status = pipeline.predict(user_cmd)
        print(json_res)
    else:
        run_demo_suite()
