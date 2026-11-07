import os
import json
import random
import csv
from collections import Counter
import sys

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "."))
OUT_DIR = os.path.join(ROOT_DIR, "common_split")

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def get_unique_labels(data, key):
    return set(item[key] for item in data)

def create_split():
    # 1. Load original
    train_orig = load_jsonl(os.path.join(ROOT_DIR, "train.jsonl"))
    val_orig = load_jsonl(os.path.join(ROOT_DIR, "validation.jsonl"))
    test_orig = load_jsonl(os.path.join(ROOT_DIR, "test.jsonl"))
    
    all_samples = train_orig + val_orig + test_orig
    
    assert len(all_samples) == 1600, "Must be exactly 1600 samples."
    
    # 2. Extract unique labels across full dataset
    all_targets = get_unique_labels(all_samples, "Target")
    all_intents = get_unique_labels(all_samples, "Intent")
    all_speeds = get_unique_labels(all_samples, "Speed Category")
    
    # We need to guarantee that every label appears in Train.
    # To preserve distribution somewhat and be deterministic:
    random.seed(1234)
    random.shuffle(all_samples)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Force representation of every target, intent, and speed in Train
    needed_targets = set(all_targets)
    needed_intents = set(all_intents)
    needed_speeds = set(all_speeds)
    
    remaining_samples = []
    
    for item in all_samples:
        if item["Target"] in needed_targets or item["Intent"] in needed_intents or item["Speed Category"] in needed_speeds:
            train_samples.append(item)
            needed_targets.discard(item["Target"])
            needed_intents.discard(item["Intent"])
            needed_speeds.discard(item["Speed Category"])
        else:
            remaining_samples.append(item)
            
    # Now fill train_samples up to 1120
    needed_for_train = 1120 - len(train_samples)
    train_samples.extend(remaining_samples[:needed_for_train])
    
    # Next 240 to Validation
    val_samples = remaining_samples[needed_for_train : needed_for_train + 240]
    
    # Next 240 to Test
    test_samples = remaining_samples[needed_for_train + 240 :]
    
    assert len(train_samples) == 1120
    assert len(val_samples) == 240
    assert len(test_samples) == 240
    
    # Write files
    save_jsonl(train_samples, os.path.join(OUT_DIR, "train.jsonl"))
    save_jsonl(val_samples, os.path.join(OUT_DIR, "validation.jsonl"))
    save_jsonl(test_samples, os.path.join(OUT_DIR, "test.jsonl"))
    
    # Create split_assignment.csv
    with open(os.path.join(OUT_DIR, "split_assignment.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Bangla Command", "Split"])
        for split_name, data in [("Train", train_samples), ("Validation", val_samples), ("Test", test_samples)]:
            for item in data:
                writer.writerow([item["ID"], item["Bangla Command"], split_name])
                
    # Gather reports
    def get_counts(data):
        return {
            "Intent": dict(Counter(item["Intent"] for item in data)),
            "Target": dict(Counter(item["Target"] for item in data)),
            "Speed": dict(Counter(item["Speed Category"] for item in data))
        }
        
    orig_dist = get_counts(all_samples)
    new_train_dist = get_counts(train_samples)
    new_val_dist = get_counts(val_samples)
    new_test_dist = get_counts(test_samples)
    
    # Save label_distribution.csv
    with open(os.path.join(OUT_DIR, "label_distribution.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label_Type", "Label_Value", "Global_Count", "Train_Count", "Validation_Count", "Test_Count"])
        
        for key_type, key_name in [("Intent", "Intent"), ("Target", "Target"), ("Speed", "Speed Category")]:
            unique_vals = sorted(list(get_unique_labels(all_samples, key_name)))
            for val in unique_vals:
                g = orig_dist[key_type].get(val, 0)
                tr = new_train_dist[key_type].get(val, 0)
                v = new_val_dist[key_type].get(val, 0)
                te = new_test_dist[key_type].get(val, 0)
                writer.writerow([key_type, val, g, tr, v, te])
                
    # Create split_report.json
    report = {
        "original_sizes": {"Train": 1120, "Validation": 240, "Test": 240, "Total": 1600},
        "new_sizes": {"Train": len(train_samples), "Validation": len(val_samples), "Test": len(test_samples), "Total": 1600},
        "label_closure_verified": True,
        "sample_uniqueness_verified": True,
    }
    
    with open(os.path.join(OUT_DIR, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    readme_content = """# Common Split for Comparative Experiments

This directory contains the unified and corrected label-closed dataset split.

## Reason for New Split
The original Test set contained zero-shot Target labels (e.g., `board`, `wall`) that were entirely absent from the original Training set. A fixed-architecture classification model (like BanglaBERT) cannot dynamically predict classes it has never seen during training. To enable a mathematically fair 2x2 experimental comparison between Generative LLMs and Encoder Classifiers, we completely recreated the Train/Validation/Test split from the 1600-sample pool.

## Guarantee
- Train = 1120
- Validation = 240
- Test = 240
- Every single Intent, Target, and Speed label present in Test and Validation is guaranteed to appear at least once in Train.
- All 1600 samples were used exactly once.
- The new Test set is identical for Qwen Raw, Qwen Preprocessed, BanglaBERT Raw, and BanglaBERT Preprocessed.
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("\n==================================================")
    print("FINAL COMMON SPLIT CREATED & VERIFIED")
    print("==================================================")
    print(f"Train: {len(train_samples)}")
    print(f"Validation: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")
    print("Total assigned:", len(train_samples) + len(val_samples) + len(test_samples))
    
    # Assertions
    train_ids = set(item["ID"] for item in train_samples)
    val_ids = set(item["ID"] for item in val_samples)
    test_ids = set(item["ID"] for item in test_samples)
    
    assert len(train_ids) == 1120
    assert len(val_ids) == 240
    assert len(test_ids) == 240
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(train_ids.union(val_ids).union(test_ids)) == 1600
    print("Check: 1600 samples assigned uniquely with no overlap.")
    
    t_targets = get_unique_labels(train_samples, "Target")
    v_targets = get_unique_labels(val_samples, "Target")
    te_targets = get_unique_labels(test_samples, "Target")
    
    assert v_targets.issubset(t_targets)
    assert te_targets.issubset(t_targets)
    print("Check: All Val/Test Target labels exist in Train.")
    
    prob_targets = ["blocked path", "board", "cupboard", "pole", "road barrier", "wall"]
    print("\n--- Problematic Targets Distribution ---")
    for pt in prob_targets:
        tr = new_train_dist["Target"].get(pt, 0)
        v = new_val_dist["Target"].get(pt, 0)
        te = new_test_dist["Target"].get(pt, 0)
        print(f"'{pt}' -> Train: {tr}, Validation: {v}, Test: {te}")

if __name__ == "__main__":
    create_split()
