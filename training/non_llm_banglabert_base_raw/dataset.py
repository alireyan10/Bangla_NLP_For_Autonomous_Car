"""
Dataset loader for Bangla Robot Command Multi-Task Learning (BanglaBERT-Base).
"""

import json
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import config as cfg
from config import normalize_bangla_text


def parse_action_sequence_list(val: Any) -> List[str]:
    """Extract individual actions into a list of strings."""
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    text = str(val).strip()
    if "->" in text:
        return [part.strip() for part in text.split("->") if part.strip()]
    return [text] if text else []


def clean_target_label(t: Any) -> str:
    """Canonicalize target label: 'NONE' if none, else lowercase."""
    s = str(t).strip()
    return "NONE" if s.upper() == "NONE" else s.lower()


class BanglaRobotDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        label_mappings: dict,
        max_seq_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.label_mappings = label_mappings
        self.max_seq_length = max_seq_length
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.intent2id = label_mappings["intent2id"]
        self.target2id = label_mappings["target2id"]
        self.speed2id = label_mappings["speed2id"]
        self.action2id = label_mappings["action2id"]
        self.step_action2id = label_mappings["step_action2id"]
        self.linear2id = label_mappings["linear2id"]
        self.angular2id = label_mappings["angular2id"]
        self.step2id = label_mappings["step2id"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]

        cmd_raw = record.get("Bangla Command", "")
        norm_cmd = normalize_bangla_text(cmd_raw)

        encoding = self.tokenizer(
            norm_cmd,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)

        # Labels
        intent_str = str(record.get("Intent", "")).strip()
        intent_id = self.intent2id.get(intent_str, 0)

        target_str = clean_target_label(record.get("Target", "NONE"))
        target_id = self.target2id.get(target_str, 0)

        speed_str = str(record.get("Speed Category", "NORMAL")).strip()
        speed_id = self.speed2id.get(speed_str, 0)

        step_count_val = int(record.get("Step Count", 1))
        step_count_id = self.step2id.get(str(step_count_val), 0)

        # Action sequence decomposition
        actions = parse_action_sequence_list(record.get("Action Sequence", ""))
        a1 = actions[0] if len(actions) > 0 else "MOVE_FORWARD"
        a2 = actions[1] if len(actions) > 1 else "NONE"
        a3 = actions[2] if len(actions) > 2 else "NONE"
        a4 = actions[3] if len(actions) > 3 else "NONE"

        step1_id = self.action2id.get(a1, 0)
        step2_id = self.step_action2id.get(a2, 0)
        step3_id = self.step_action2id.get(a3, 0)
        step4_id = self.step_action2id.get(a4, 0)

        # Speeds
        lin_speed_val = float(record.get("Linear Speed (m/s)", 0.15))
        ang_speed_val = float(record.get("Angular Speed (rad/s)", 0.0))

        lin_id = self.linear2id.get(str(lin_speed_val), 0)
        ang_id = self.angular2id.get(str(ang_speed_val), 0)

        gt_dict = {
            "intent": intent_str,
            "target": target_str,
            "speed_category": speed_str,
            "linear_speed": lin_speed_val,
            "angular_speed": ang_speed_val,
            "action_sequence": actions,
            "step_count": step_count_val,
        }

        return {
            "id": record.get("ID", idx),
            "command": cmd_raw,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "intent": torch.tensor(intent_id, dtype=torch.long),
            "target": torch.tensor(target_id, dtype=torch.long),
            "speed_category": torch.tensor(speed_id, dtype=torch.long),
            "step_count": torch.tensor(step_count_id, dtype=torch.long),
            "step1": torch.tensor(step1_id, dtype=torch.long),
            "step2": torch.tensor(step2_id, dtype=torch.long),
            "step3": torch.tensor(step3_id, dtype=torch.long),
            "step4": torch.tensor(step4_id, dtype=torch.long),
            "linear_speed_class": torch.tensor(lin_id, dtype=torch.long),
            "angular_speed_class": torch.tensor(ang_id, dtype=torch.long),
            "linear_speed_val": torch.tensor(lin_speed_val, dtype=torch.float),
            "angular_speed_val": torch.tensor(ang_speed_val, dtype=torch.float),
            "ground_truth": gt_dict,
        }


def bangla_robot_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "command": [item["command"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "token_type_ids": torch.stack([item["token_type_ids"] for item in batch]),
        "intent": torch.stack([item["intent"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "speed_category": torch.stack([item["speed_category"] for item in batch]),
        "step_count": torch.stack([item["step_count"] for item in batch]),
        "step1": torch.stack([item["step1"] for item in batch]),
        "step2": torch.stack([item["step2"] for item in batch]),
        "step3": torch.stack([item["step3"] for item in batch]),
        "step4": torch.stack([item["step4"] for item in batch]),
        "linear_speed_class": torch.stack([item["linear_speed_class"] for item in batch]),
        "angular_speed_class": torch.stack([item["angular_speed_class"] for item in batch]),
        "linear_speed_val": torch.stack([item["linear_speed_val"] for item in batch]),
        "angular_speed_val": torch.stack([item["angular_speed_val"] for item in batch]),
        "ground_truth": [item["ground_truth"] for item in batch],
    }
