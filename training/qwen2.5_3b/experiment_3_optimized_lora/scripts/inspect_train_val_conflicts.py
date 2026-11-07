"""
Comprehensive Audit of Label Conflicts in Train and Validation Splits.
Analyzes conflicting commands and checks how Intent, Speed Category,
Linear Speed, Angular Speed, Action Sequence, and Step Count correlate.
"""

import os
import sys
import json
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"C:\Users\Reyan\Downloads\Case_Study"
TRAIN_PATH = os.path.join(PROJECT_ROOT, "PREPROCESSING_EXPERIMENT", "output", "splits", "train.jsonl")
VAL_PATH = os.path.join(PROJECT_ROOT, "PREPROCESSING_EXPERIMENT", "output", "splits", "validation.jsonl")

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_records = [json.loads(line) for line in f if line.strip()]

with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_records = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(train_records)} train records and {len(val_records)} validation records.")

all_records = train_records + val_records

# Group by Bangla Command
cmd_groups = defaultdict(list)
for r in all_records:
    cmd_groups[r["Bangla Command"].strip()].append(r)

conflicting = {cmd: rows for cmd, rows in cmd_groups.items() if len(set(r["Intent"] for r in rows)) > 1}
print(f"Total commands with conflicting intents: {len(conflicting)}")

# Classify conflicts into semantic buckets
buckets = {
    "MOVE_SLOW_vs_DECREASE_SPEED": [],
    "MOVE_FAST_vs_INCREASE_SPEED": [],
    "ROTATE_vs_TURN": [],
    "OTHER": []
}

for cmd, rows in sorted(conflicting.items(), key=lambda x: len(x[1]), reverse=True):
    intents = set(r["Intent"] for r in rows)
    if intents.issubset({"MOVE_SLOW", "DECREASE_SPEED"}):
        buckets["MOVE_SLOW_vs_DECREASE_SPEED"].append((cmd, rows))
    elif intents.issubset({"MOVE_FAST", "INCREASE_SPEED"}):
        buckets["MOVE_FAST_vs_INCREASE_SPEED"].append((cmd, rows))
    elif any(("ROTATE" in i and any("TURN" in j for j in intents)) for i in intents):
        buckets["ROTATE_vs_TURN"].append((cmd, rows))
    else:
        buckets["OTHER"].append((cmd, rows))

for bucket_name, items in buckets.items():
    print("=" * 80)
    print(f"BUCKET: {bucket_name} ({len(items)} unique commands, total {sum(len(rows) for _, rows in items)} samples)")
    print("=" * 80)
    for cmd, rows in items[:15]:  # print first 15 for each bucket
        intents_count = defaultdict(int)
        for r in rows:
            intents_count[r["Intent"]] += 1
        print(f"• '{cmd}' (Total: {len(rows)}) -> {dict(intents_count)}")
        # Show sample row details
        for r in rows[:2]:
            print(f"    Sample ID {r['ID']}: Intent={r['Intent']}, SpeedCat={r['Speed Category']}, LinSpd={r['Linear Speed (m/s)']}, AngSpd={r['Angular Speed (rad/s)']}, ActSeq={r['Action Sequence']}")
