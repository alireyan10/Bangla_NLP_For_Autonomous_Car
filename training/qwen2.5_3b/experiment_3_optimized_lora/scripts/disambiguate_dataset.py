"""
Dataset Disambiguation Script for Experiment 3 (Optimized LoRA).

Applies well-defined semantic rules to resolve 50/50 contradictory labels in
train.jsonl and validation.jsonl.
Strictly leaves test.jsonl (IDs 3401-4000) FROZEN and untouched.

Rules applied:
1. MOVE_SLOW vs DECREASE_SPEED:
   Steady slow forward motion phrases ("ধীরে", "ধীর", "আস্তে", "slowly") without deceleration verbs
   are mapped to Intent=MOVE_SLOW and Action Sequence=MOVE_SLOW.
2. MOVE_FAST vs INCREASE_SPEED:
   Steady fast forward motion phrases ("দ্রুত", "জোরে", "তাড়াতাড়ি", "fast") without acceleration verbs
   are mapped to Intent=MOVE_FAST and Action Sequence=MOVE_FAST.
3. ROTATE vs TURN:
   Phrases with arc/turn keywords ("বাঁক", "টার্ন") are mapped to TURN_LEFT / TURN_RIGHT.
   Phrases with spin/rotate keywords ("ঘুর", "ঘোরা") are mapped to ROTATE_LEFT / ROTATE_RIGHT.
4. Sequential action sequence consistency for conflicting multi-step commands.
5. Strict target casing: lowercase for entities, NONE for empty.

Outputs:
- experiment_3_optimized_lora/data/train_disambiguated.jsonl
- experiment_3_optimized_lora/data/validation_disambiguated.jsonl
- experiment_3_optimized_lora/data/label_correction_log.json
"""

import os
import sys
import json
import copy

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"C:\Users\Reyan\Downloads\Case_Study"
EXP3_DIR = os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_3_optimized_lora")
DATA_DIR = os.path.join(EXP3_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_IN = os.path.join(PROJECT_ROOT, "PREPROCESSING_EXPERIMENT", "output", "splits", "train.jsonl")
VAL_IN = os.path.join(PROJECT_ROOT, "PREPROCESSING_EXPERIMENT", "output", "splits", "validation.jsonl")

TRAIN_OUT = os.path.join(DATA_DIR, "train_disambiguated.jsonl")
VAL_OUT = os.path.join(DATA_DIR, "validation_disambiguated.jsonl")
LOG_OUT = os.path.join(DATA_DIR, "label_correction_log.json")


def disambiguate_record(r: dict, split_name: str) -> tuple[dict, dict | None]:
    """
    Examines a single record. If a conflicting/ambiguous label is found,
    returns the corrected record and a log entry. Otherwise returns (original, None).
    """
    rec = copy.deepcopy(r)
    cmd = rec["Bangla Command"].strip()
    intent = rec["Intent"].strip()
    act_seq = rec["Action Sequence"]
    speed_cat = rec["Speed Category"]
    lin_spd = float(rec["Linear Speed (m/s)"])
    ang_spd = float(rec["Angular Speed (rad/s)"])
    target = rec["Target"].strip()

    modifications = {}
    rationale = []

    # 1. Target casing check
    normalized_target = "NONE" if target.upper() == "NONE" else target.lower()
    if target != normalized_target:
        modifications["Target"] = {"old": target, "new": normalized_target}
        rec["Target"] = normalized_target
        rationale.append(f"Target casing normalized from '{target}' to '{normalized_target}'")

    # 2. Sequential command conflicts
    if intent == "SEQUENTIAL":
        if "ধীর গতিতে চল" in cmd and "বাঁক নাও" in cmd:
            new_seq = "MOVE_SLOW -> TURN_RIGHT"
            if act_seq != new_seq:
                modifications["Action Sequence"] = {"old": act_seq, "new": new_seq}
                rec["Action Sequence"] = new_seq
                rationale.append("Sequential disambiguation: steady slow motion + right turn")
        elif "ধীরে যাও" in cmd and "ধীরে এগিয়ে যাও" in cmd:
            new_seq = "MOVE_SLOW -> MOVE_SLOW"
            if act_seq != new_seq:
                modifications["Action Sequence"] = {"old": act_seq, "new": new_seq}
                rec["Action Sequence"] = new_seq
                rationale.append("Sequential disambiguation: two steady slow motions")

    # 3. Single-step MOVE_SLOW vs DECREASE_SPEED
    elif intent in ["MOVE_SLOW", "DECREASE_SPEED"]:
        # Check if command has deceleration verbs ('কমাও', 'কম')
        has_decel = any(k in cmd for k in ["কমাও", "কম"])
        if not has_decel:
            # Steady slow motion
            if intent != "MOVE_SLOW":
                modifications["Intent"] = {"old": intent, "new": "MOVE_SLOW"}
                rec["Intent"] = "MOVE_SLOW"
                rationale.append("Disambiguation: Steady slow motion phrase without deceleration verb mapped to MOVE_SLOW")
            if act_seq != "MOVE_SLOW" and act_seq != ["MOVE_SLOW"]:
                modifications["Action Sequence"] = {"old": act_seq, "new": "MOVE_SLOW"}
                rec["Action Sequence"] = "MOVE_SLOW"
                rationale.append("Action Sequence aligned with MOVE_SLOW")

    # 4. Single-step MOVE_FAST vs INCREASE_SPEED
    elif intent in ["MOVE_FAST", "INCREASE_SPEED"]:
        # Check if command has acceleration verbs ('বাড়াও', 'বৃদ্ধি', 'স্পিড আপ')
        has_accel = any(k in cmd for k in ["বাড়াও", "বাড়া", "বৃদ্ধি", "স্পিড আপ"])
        if not has_accel:
            # Steady fast motion
            if intent != "MOVE_FAST":
                modifications["Intent"] = {"old": intent, "new": "MOVE_FAST"}
                rec["Intent"] = "MOVE_FAST"
                rationale.append("Disambiguation: Steady fast motion phrase without acceleration verb mapped to MOVE_FAST")
            if act_seq != "MOVE_FAST" and act_seq != ["MOVE_FAST"]:
                modifications["Action Sequence"] = {"old": act_seq, "new": "MOVE_FAST"}
                rec["Action Sequence"] = "MOVE_FAST"
                rationale.append("Action Sequence aligned with MOVE_FAST")

    # 5. Single-step ROTATE vs TURN
    elif intent in ["ROTATE_LEFT", "TURN_LEFT", "ROTATE_RIGHT", "TURN_RIGHT"]:
        is_left = ("বাম" in cmd) or ("left" in cmd.lower())
        is_right = ("ডান" in cmd) or ("right" in cmd.lower())
        has_arc = ("বাঁক" in cmd) or ("টার্ন" in cmd)
        has_spin = any(k in cmd for k in ["ঘুর", "ঘোরা"])

        target_intent = None
        if has_arc:
            # Steering / curved turn
            target_intent = "TURN_LEFT" if is_left else ("TURN_RIGHT" if is_right else intent)
        elif has_spin:
            # In-place spin / rotation
            target_intent = "ROTATE_LEFT" if is_left else ("ROTATE_RIGHT" if is_right else intent)

        if target_intent and target_intent != intent:
            modifications["Intent"] = {"old": intent, "new": target_intent}
            rec["Intent"] = target_intent
            rationale.append(f"Disambiguation: Verb distinction mapped to {target_intent} (has_arc={has_arc}, has_spin={has_spin})")

            # Harmonize action sequence
            if act_seq != target_intent:
                modifications["Action Sequence"] = {"old": act_seq, "new": target_intent}
                rec["Action Sequence"] = target_intent
                rationale.append(f"Action Sequence aligned with {target_intent}")

    if modifications:
        log_entry = {
            "id": rec["ID"],
            "split": split_name,
            "command": cmd,
            "modifications": modifications,
            "rationale": "; ".join(rationale)
        }
        return rec, log_entry

    return rec, None


def process_split(input_path: str, output_path: str, split_name: str) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\nProcessing {split_name} split ({len(records)} records)...")
    cleaned_records = []
    split_logs = []

    for r in records:
        cleaned_r, log_entry = disambiguate_record(r, split_name)
        cleaned_records.append(cleaned_r)
        if log_entry:
            split_logs.append(log_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in cleaned_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(cleaned_records)} records to: {output_path}")
    print(f"Total modified records in {split_name}: {len(split_logs)}")
    return split_logs


def main():
    print("=" * 80)
    print("STARTING DATA DISAMBIGUATION FOR EXPERIMENT 3")
    print("=" * 80)

    train_logs = process_split(TRAIN_IN, TRAIN_OUT, "train")
    val_logs = process_split(VAL_IN, VAL_OUT, "validation")

    total_logs = train_logs + val_logs

    # Category breakdown of modifications
    cat_counts = {
        "MOVE_SLOW_disambiguation": 0,
        "MOVE_FAST_disambiguation": 0,
        "ROTATE_disambiguation": 0,
        "TURN_disambiguation": 0,
        "SEQUENTIAL_disambiguation": 0,
        "TARGET_casing_normalization": 0
    }

    for item in total_logs:
        rat = item["rationale"]
        if "MOVE_SLOW" in rat:
            cat_counts["MOVE_SLOW_disambiguation"] += 1
        if "MOVE_FAST" in rat:
            cat_counts["MOVE_FAST_disambiguation"] += 1
        if "ROTATE" in rat:
            cat_counts["ROTATE_disambiguation"] += 1
        if "TURN" in rat:
            cat_counts["TURN_disambiguation"] += 1
        if "Sequential" in rat:
            cat_counts["SEQUENTIAL_disambiguation"] += 1
        if "Target casing" in rat:
            cat_counts["TARGET_casing_normalization"] += 1

    summary = {
        "total_records_processed": 3400,
        "train_records": 2800,
        "validation_records": 600,
        "train_modifications_count": len(train_logs),
        "validation_modifications_count": len(val_logs),
        "total_modifications_count": len(total_logs),
        "category_counts": cat_counts,
        "detailed_logs": total_logs
    }

    with open(LOG_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nLabel correction log saved to: {LOG_OUT}")
    print("=" * 80)
    print("DISAMBIGUATION SUMMARY:")
    print(f"  • Train modifications:      {len(train_logs)} / 2800 ({len(train_logs)/2800*100:.1f}%)")
    print(f"  • Validation modifications: {len(val_logs)} / 600 ({len(val_logs)/600*100:.1f}%)")
    print(f"  • Total modifications:      {len(total_logs)} / 3400 ({len(total_logs)/3400*100:.1f}%)")
    print("Category Breakdown:")
    for cat, cnt in cat_counts.items():
        print(f"    - {cat}: {cnt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
