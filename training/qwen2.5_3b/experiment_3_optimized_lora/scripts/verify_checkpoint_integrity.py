"""
Comprehensive Checkpoint Integrity Verification Script.
Checks both checkpoint-525 and checkpoint-700.
"""

import os
import sys
import json
import torch
from safetensors.torch import load_file

PROJECT_ROOT = r"C:\Users\Reyan\Downloads\Case_Study"
CKPTS_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "qwen2.5_3b",
    "experiment_3_optimized_lora",
    "outputs",
    "checkpoints",
)


def verify_checkpoint(ckpt_name: str) -> dict:
    ckpt_path = os.path.join(CKPTS_DIR, ckpt_name)
    report = {"name": ckpt_name, "path": ckpt_path, "status": "UNKNOWN", "checks": {}}

    if not os.path.exists(ckpt_path):
        report["status"] = "DOES_NOT_EXIST"
        return report

    files = os.listdir(ckpt_path)
    report["files"] = files

    # 1. Check safetensors
    sf_path = os.path.join(ckpt_path, "adapter_model.safetensors")
    if not os.path.exists(sf_path):
        report["checks"]["safetensors"] = "MISSING"
    else:
        try:
            tensors = load_file(sf_path)
            report["checks"]["safetensors"] = f"VALID ({len(tensors)} tensors, {os.path.getsize(sf_path)} bytes)"
        except Exception as e:
            report["checks"]["safetensors"] = f"CORRUPTED: {e}"

    # 2. Check optimizer.pt
    opt_path = os.path.join(ckpt_path, "optimizer.pt")
    if not os.path.exists(opt_path):
        report["checks"]["optimizer"] = "MISSING"
    else:
        try:
            opt_data = torch.load(opt_path, map_location="cpu")
            if isinstance(opt_data, dict):
                report["checks"]["optimizer"] = f"VALID (keys: {list(opt_data.keys())}, {os.path.getsize(opt_path)} bytes)"
            else:
                report["checks"]["optimizer"] = f"VALID ({type(opt_data)}, {os.path.getsize(opt_path)} bytes)"
        except Exception as e:
            report["checks"]["optimizer"] = f"CORRUPTED: {e}"

    # 3. Check adapter_config.json
    cfg_path = os.path.join(ckpt_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        report["checks"]["adapter_config"] = "MISSING"
    else:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            report["checks"]["adapter_config"] = f"VALID (r={cfg_data.get('r')}, alpha={cfg_data.get('lora_alpha')})"
        except Exception as e:
            report["checks"]["adapter_config"] = f"CORRUPTED: {e}"

    # 4. Check trainer_state.json
    ts_path = os.path.join(ckpt_path, "trainer_state.json")
    if not os.path.exists(ts_path):
        report["checks"]["trainer_state"] = "MISSING"
    else:
        try:
            with open(ts_path, "r", encoding="utf-8") as f:
                ts_data = json.load(f)
            report["checks"]["trainer_state"] = f"VALID (global_step={ts_data.get('global_step')}, epoch={ts_data.get('epoch')})"
        except Exception as e:
            report["checks"]["trainer_state"] = f"CORRUPTED: {e}"

    # 5. Check training_args.bin
    args_path = os.path.join(ckpt_path, "training_args.bin")
    report["checks"]["training_args"] = "PRESENT" if os.path.exists(args_path) else "MISSING"

    # Overall judgment
    has_corrupt = any("CORRUPTED" in str(v) for v in report["checks"].values())
    sf_ok = "VALID" in report["checks"].get("safetensors", "")
    opt_ok = "VALID" in report["checks"].get("optimizer", "")
    cfg_ok = "VALID" in report["checks"].get("adapter_config", "")
    ts_ok = "VALID" in report["checks"].get("trainer_state", "")

    if has_corrupt:
        report["status"] = "CORRUPTED"
    elif sf_ok and opt_ok and cfg_ok and ts_ok:
        report["status"] = "FULLY_VALID_AND_READY"
    elif sf_ok and opt_ok and cfg_ok and not ts_ok:
        report["status"] = "WEIGHTS_VALID_BUT_MISSING_TRAINER_STATE"
    else:
        report["status"] = "INCOMPLETE"

    return report


def main():
    print("=" * 80)
    print("CHECKPOINT INTEGRITY AUDIT")
    print("=" * 80)

    for ckpt in ["checkpoint-175", "checkpoint-350", "checkpoint-525", "checkpoint-700"]:
        rep = verify_checkpoint(ckpt)
        print(f"\n[{rep['name']}] Status: {rep['status']}")
        for k, v in rep["checks"].items():
            print(f"  • {k:15}: {v}")


if __name__ == "__main__":
    main()
