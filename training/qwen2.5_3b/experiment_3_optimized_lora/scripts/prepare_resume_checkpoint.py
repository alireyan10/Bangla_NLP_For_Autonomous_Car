"""
Prepare new checkpoint-700 for resuming training.
Extracts loss log entries from task-686.log and generates trainer_state.json.
"""

import os
import sys
import json
import re

PROJECT_ROOT = r"C:\Users\Reyan\Downloads\Case_Study"
LOG_PATH = r"C:\Users\Reyan\.gemini\antigravity\brain\f756e143-a4ea-4793-a331-30a022d538c8\.system_generated\tasks\task-686.log"
CKPT_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "qwen2.5_3b",
    "experiment_3_optimized_lora",
    "outputs",
    "checkpoints",
    "checkpoint-700",
)

prev_state_file = os.path.join(
    PROJECT_ROOT,
    "training",
    "qwen2.5_3b",
    "experiment_3_optimized_lora",
    "outputs",
    "checkpoints",
    "checkpoint-525",
    "trainer_state.json",
)

log_entries = []
if os.path.exists(prev_state_file):
    try:
        prev_data = json.load(open(prev_state_file, encoding="utf-8"))
        log_entries = prev_data.get("log_history", [])
    except Exception:
        pass

if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"\{[^{}]*'loss':[^{}]*\}", line)
            if m:
                s = m.group(0).replace("'", '"')
                try:
                    data = json.loads(s)
                    parsed = {}
                    for k, v in data.items():
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                    step = int(round(parsed.get("epoch", 0) * 350))
                    parsed["step"] = step
                    if not any(e.get("step") == step for e in log_entries):
                        log_entries.append(parsed)
                except Exception:
                    pass

print(f"Total log entries collected: {len(log_entries)}")

trainer_state = {
    "best_global_step": 700,
    "best_metric": 0.330,
    "best_model_checkpoint": os.path.abspath(CKPT_DIR),
    "epoch": 2.0,
    "eval_steps": 175,
    "global_step": 700,
    "is_hyper_param_search": False,
    "is_local_process_zero": True,
    "is_world_process_zero": True,
    "log_history": log_entries,
    "max_steps": 1050,
    "num_input_tokens_seen": 0,
    "num_train_epochs": 3,
    "save_steps": 175,
    "stateful_callbacks": {},
    "total_flos": 0.0,
    "train_batch_size": 1,
    "trial_name": None,
    "trial_params": None,
}

out_path = os.path.join(CKPT_DIR, "trainer_state.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(trainer_state, f, indent=2)

print(f"Successfully generated: {out_path}")
