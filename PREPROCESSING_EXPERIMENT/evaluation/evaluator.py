"""
Standardized Evaluation Engine for Robot Command Mapping Models.

Calculates key performance metrics consistently:
- JSON Validity
- Intent Accuracy
- Target Accuracy
- Speed Category Accuracy
- Action Sequence Accuracy
- Step Count Accuracy
- Linear Speed Accuracy (tolerance <= 1e-3)
- Angular Speed Accuracy (tolerance <= 1e-3)
- Primary 4-Field Categorical Exact Match (intent, target, speed_category, action_sequence)
- Comprehensive 7-Field Exact Match (4-field + step_count + linear_speed + angular_speed)

Guarantees:
- Never scores invalid JSON as correct for any metric.
- Never silently invents missing values.
- Cleanly separates format errors from semantic prediction mismatches.
- Handles extra and missing fields predictably.
"""

import json
import math
from typing import Dict, Any, List, Optional, Tuple
from .robust_json_parser import parse_robot_json

REQUIRED_FIELDS = [
    "intent",
    "target",
    "speed_category",
    "action_sequence",
    "step_count",
    "linear_speed",
    "angular_speed"
]

CATEGORICAL_4_FIELDS = [
    "intent",
    "target",
    "speed_category",
    "action_sequence"
]


def _is_valid_number(val: Any) -> bool:
    """Check if value is a finite real number, explicitly excluding booleans."""
    if isinstance(val, bool) or val is None:
        return False
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def evaluate_single_sample(
    raw_prediction: Any,
    ground_truth: Dict[str, Any],
    tolerance: float = 1e-3
) -> Dict[str, Any]:
    """
    Evaluate a single model generation against ground truth.

    Parameters
    ----------
    raw_prediction : Any
        The raw LLM output string.
    ground_truth : Dict[str, Any]
        Dictionary of ground truth values.
    tolerance : float
        Numeric tolerance for linear and angular speed comparisons.

    Returns
    -------
    Dict[str, Any]
        Detailed evaluation record for this sample.
    """
    parsed_dict, is_valid_json, err_msg = parse_robot_json(raw_prediction)

    record: Dict[str, Any] = {
        "is_valid_json": is_valid_json,
        "format_error": not is_valid_json,
        "parser_error_message": err_msg,
        "parsed_json": parsed_dict,
        "missing_fields": [],
        "extra_fields": [],
        "intent_match": False,
        "target_match": False,
        "speed_category_match": False,
        "action_sequence_match": False,
        "step_count_match": False,
        "linear_speed_match": False,
        "angular_speed_match": False,
        "exact_match_4field": False,
        "exact_match_7field": False,
        "linear_speed_diff": None,
        "angular_speed_diff": None,
    }

    # If JSON is invalid, never score as correct for any metric
    if not is_valid_json or not isinstance(parsed_dict, dict):
        record["missing_fields"] = list(REQUIRED_FIELDS)
        return record

    # Inspect field presence
    for req in REQUIRED_FIELDS:
        if req not in parsed_dict:
            record["missing_fields"].append(req)

    for key in parsed_dict.keys():
        if key not in REQUIRED_FIELDS:
            record["extra_fields"].append(key)

    # 1. Intent match
    gt_intent = str(ground_truth.get("intent", "")).strip()
    if "intent" in parsed_dict:
        pred_intent = str(parsed_dict["intent"]).strip()
        record["intent_match"] = (pred_intent == gt_intent)
    else:
        record["intent_match"] = False

    # 2. Target match
    gt_target = str(ground_truth.get("target", "")).strip()
    if "target" in parsed_dict:
        pred_target = str(parsed_dict["target"]).strip()
        record["target_match"] = (pred_target == gt_target)
    else:
        record["target_match"] = False

    # 3. Speed category match
    gt_speed = str(ground_truth.get("speed_category", "")).strip()
    if "speed_category" in parsed_dict:
        pred_speed = str(parsed_dict["speed_category"]).strip()
        record["speed_category_match"] = (pred_speed == gt_speed)
    else:
        record["speed_category_match"] = False

    # 4. Action sequence match
    gt_seq = ground_truth.get("action_sequence", [])
    if isinstance(gt_seq, str):
        # Handle string serialization if present
        if "->" in gt_seq:
            gt_seq = [s.strip() for s in gt_seq.split("->")]
        else:
            gt_seq = [gt_seq.strip()]
    elif not isinstance(gt_seq, list):
        gt_seq = [str(gt_seq)]

    if "action_sequence" in parsed_dict and isinstance(parsed_dict["action_sequence"], list):
        pred_seq = [str(a).strip() for a in parsed_dict["action_sequence"]]
        record["action_sequence_match"] = (pred_seq == gt_seq)
    else:
        record["action_sequence_match"] = False

    # Primary 4-Field Exact Match
    record["exact_match_4field"] = (
        record["intent_match"]
        and record["target_match"]
        and record["speed_category_match"]
        and record["action_sequence_match"]
    )

    # 5. Step count match
    try:
        gt_step = int(ground_truth.get("step_count", 0))
    except (ValueError, TypeError):
        gt_step = 0

    if "step_count" in parsed_dict and _is_valid_number(parsed_dict["step_count"]):
        try:
            pred_step = int(parsed_dict["step_count"])
            record["step_count_match"] = (pred_step == gt_step)
        except (ValueError, TypeError):
            record["step_count_match"] = False
    else:
        record["step_count_match"] = False

    # 6. Linear speed match
    try:
        gt_lin = float(ground_truth.get("linear_speed", 0.0))
    except (ValueError, TypeError):
        gt_lin = 0.0

    if "linear_speed" in parsed_dict and _is_valid_number(parsed_dict["linear_speed"]):
        try:
            pred_lin = float(parsed_dict["linear_speed"])
            diff = abs(pred_lin - gt_lin)
            record["linear_speed_diff"] = diff
            record["linear_speed_match"] = (diff <= tolerance)
        except (ValueError, TypeError):
            record["linear_speed_match"] = False
    else:
        record["linear_speed_match"] = False

    # 7. Angular speed match
    try:
        gt_ang = float(ground_truth.get("angular_speed", 0.0))
    except (ValueError, TypeError):
        gt_ang = 0.0

    if "angular_speed" in parsed_dict and _is_valid_number(parsed_dict["angular_speed"]):
        try:
            pred_ang = float(parsed_dict["angular_speed"])
            diff = abs(pred_ang - gt_ang)
            record["angular_speed_diff"] = diff
            record["angular_speed_match"] = (diff <= tolerance)
        except (ValueError, TypeError):
            record["angular_speed_match"] = False
    else:
        record["angular_speed_match"] = False

    # Comprehensive 7-Field Exact Match
    record["exact_match_7field"] = (
        record["exact_match_4field"]
        and record["step_count_match"]
        and record["linear_speed_match"]
        and record["angular_speed_match"]
    )

    return record


def evaluate_dataset(
    items: List[Dict[str, Any]],
    tolerance: float = 1e-3
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate a list of sample dictionaries.
    Each item must contain:
      - 'raw_prediction': str
      - 'ground_truth': dict (or json string)
      - optional 'id', 'command'

    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]]]
        - summary_metrics
        - detailed_records
    """
    total = len(items)
    if total == 0:
        return {}, []

    valid_json_count = 0
    format_errors_count = 0
    intent_matches = 0
    target_matches = 0
    speed_matches = 0
    seq_matches = 0
    step_matches = 0
    lin_matches = 0
    ang_matches = 0
    exact_4_matches = 0
    exact_7_matches = 0

    lin_diffs: List[float] = []
    ang_diffs: List[float] = []
    detailed_records: List[Dict[str, Any]] = []

    for item in items:
        raw_pred = item.get("raw_prediction", item.get("prediction", item.get("prediction_raw", "")))
        gt = item.get("ground_truth", {})
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except Exception:
                gt = {}

        res = evaluate_single_sample(raw_pred, gt, tolerance=tolerance)
        res["id"] = item.get("id")
        res["command"] = item.get("command", item.get("input", ""))
        detailed_records.append(res)

        if res["is_valid_json"]:
            valid_json_count += 1
        else:
            format_errors_count += 1

        if res["intent_match"]:
            intent_matches += 1
        if res["target_match"]:
            target_matches += 1
        if res["speed_category_match"]:
            speed_matches += 1
        if res["action_sequence_match"]:
            seq_matches += 1
        if res["step_count_match"]:
            step_matches += 1
        if res["linear_speed_match"]:
            lin_matches += 1
        if res["angular_speed_match"]:
            ang_matches += 1
        if res["exact_match_4field"]:
            exact_4_matches += 1
        if res["exact_match_7field"]:
            exact_7_matches += 1

        if res["linear_speed_diff"] is not None:
            lin_diffs.append(res["linear_speed_diff"])
        if res["angular_speed_diff"] is not None:
            ang_diffs.append(res["angular_speed_diff"])

    summary = {
        "total_samples": total,
        "valid_json_count": valid_json_count,
        "format_error_count": format_errors_count,
        "json_validity_rate": valid_json_count / total,
        "intent_accuracy": intent_matches / total,
        "target_accuracy": target_matches / total,
        "speed_category_accuracy": speed_matches / total,
        "action_sequence_accuracy": seq_matches / total,
        "step_count_accuracy": step_matches / total,
        "linear_speed_accuracy": lin_matches / total,
        "angular_speed_accuracy": ang_matches / total,
        "exact_match_4field_count": exact_4_matches,
        "exact_match_4field": exact_4_matches / total,
        "exact_match_7field_count": exact_7_matches,
        "exact_match_7field": exact_7_matches / total,
        "linear_speed_mae": (sum(lin_diffs) / len(lin_diffs)) if lin_diffs else 0.0,
        "angular_speed_mae": (sum(ang_diffs) / len(ang_diffs)) if ang_diffs else 0.0,
        "tolerance": tolerance
    }

    return summary, detailed_records
