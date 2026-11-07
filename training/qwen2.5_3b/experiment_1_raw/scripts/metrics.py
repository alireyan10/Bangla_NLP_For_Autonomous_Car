import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_llm_output(raw_text):
    """Attempt to parse the LLM output into a dictionary."""
    text = raw_text.strip()
    
    # Try to find JSON block if wrapped in markdown
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
        
    text = text.strip()
    
    # Robust extraction: find the first { and the last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        text = text[start_idx:end_idx+1]
    
    try:
        data = json.loads(text)
        return data, True, ""
    except json.JSONDecodeError as e:
        # Some heuristic fallback if slightly malformed
        # For this research, we just record format errors
        return None, False, str(e)

def compute_metrics():
    base_dir = r"C:\Users\Reyan\Downloads\Case_Study\training\qwen2.5_3b\experiment_1_raw"
    predictions_file = os.path.join(base_dir, "evaluation", "predictions.jsonl")
    out_dir = os.path.join(base_dir, "evaluation", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    total_samples = len(results)
    
    # Trackers
    format_errors = 0
    
    y_true_intent = []
    y_pred_intent = []
    
    y_true_target = []
    y_pred_target = []
    
    y_true_speed = []
    y_pred_speed = []
    
    exact_matches = 0
    action_sequence_matches = 0
    
    error_cases = []

    for res in results:
        gt = res['ground_truth']
        raw_pred = res['prediction_raw']
        
        pred_dict, is_valid, err_msg = parse_llm_output(raw_pred)
        
        if not is_valid:
            format_errors += 1
            error_cases.append({
                "id": res['id'],
                "command": res['command'],
                "error_type": "format_error",
                "raw_pred": raw_pred,
                "msg": err_msg
            })
            # To maintain lengths for sklearn metrics, append 'FORMAT_ERROR'
            y_true_intent.append(gt['intent'])
            y_pred_intent.append("FORMAT_ERROR")
            
            y_true_target.append(gt['target'])
            y_pred_target.append("FORMAT_ERROR")
            
            y_true_speed.append(gt['speed_category'])
            y_pred_speed.append("FORMAT_ERROR")
            continue
            
        # Extract fields robustly
        pred_intent = str(pred_dict.get('intent', 'MISSING')).strip()
        pred_target = str(pred_dict.get('target', 'MISSING')).strip()
        pred_speed = str(pred_dict.get('speed_category', 'MISSING')).strip()
        
        y_true_intent.append(gt['intent'])
        y_pred_intent.append(pred_intent)
        
        y_true_target.append(gt['target'])
        y_pred_target.append(pred_target)
        
        y_true_speed.append(gt['speed_category'])
        y_pred_speed.append(pred_speed)
        
        # Check action sequence
        pred_seq = pred_dict.get('action_sequence', [])
        if not isinstance(pred_seq, list):
            pred_seq = []
            
        gt_seq = gt['action_sequence']
        
        seq_match = (pred_seq == gt_seq)
        if seq_match:
            action_sequence_matches += 1
            
        # Check exact match (all main categorical fields match)
        field_match = (pred_intent == gt['intent']) and \
                      (pred_target == gt['target']) and \
                      (pred_speed == gt['speed_category'])
                      
        if field_match and seq_match:
            exact_matches += 1
        else:
            error_cases.append({
                "id": res['id'],
                "command": res['command'],
                "error_type": "value_mismatch",
                "gt": gt,
                "pred": pred_dict
            })
            
    # Calculate Metrics
    acc_intent = accuracy_score(y_true_intent, y_pred_intent)
    acc_target = accuracy_score(y_true_target, y_pred_target)
    acc_speed = accuracy_score(y_true_speed, y_pred_speed)
    acc_exact = exact_matches / total_samples
    acc_seq = action_sequence_matches / total_samples
    format_success_rate = (total_samples - format_errors) / total_samples
    
    # Save Report
    report = {
        "total_samples": total_samples,
        "format_success_rate": format_success_rate,
        "format_errors": format_errors,
        "accuracy_intent": acc_intent,
        "accuracy_target": acc_target,
        "accuracy_speed": acc_speed,
        "accuracy_action_sequence": acc_seq,
        "exact_match_accuracy": acc_exact
    }
    
    with open(os.path.join(out_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    with open(os.path.join(out_dir, "error_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, indent=4, ensure_ascii=False)
        
    print("Metrics Summary:")
    print(json.dumps(report, indent=4))
    
    # Generate Confusion Matrices
    def plot_cm(y_true, y_pred, title, filename):
        labels = sorted(list(set(y_true)))
        if "FORMAT_ERROR" in y_pred and "FORMAT_ERROR" not in labels:
            labels.append("FORMAT_ERROR")
        if "MISSING" in y_pred and "MISSING" not in labels:
            labels.append("MISSING")
            
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()
        
    plot_cm(y_true_intent, y_pred_intent, "Intent Confusion Matrix", "cm_intent.png")
    plot_cm(y_true_target, y_pred_target, "Target Confusion Matrix", "cm_target.png")
    plot_cm(y_true_speed, y_pred_speed, "Speed Confusion Matrix", "cm_speed.png")
    
    # Save text classification reports
    with open(os.path.join(out_dir, "classification_reports.txt"), 'w', encoding='utf-8') as f:
        f.write("=== Intent ===\n")
        f.write(classification_report(y_true_intent, y_pred_intent, zero_division=0))
        f.write("\n\n=== Target ===\n")
        f.write(classification_report(y_true_target, y_pred_target, zero_division=0))
        f.write("\n\n=== Speed ===\n")
        f.write(classification_report(y_true_speed, y_pred_speed, zero_division=0))
        
    print(f"Metrics, plots, and error analysis saved to {out_dir}")

if __name__ == "__main__":
    compute_metrics()
