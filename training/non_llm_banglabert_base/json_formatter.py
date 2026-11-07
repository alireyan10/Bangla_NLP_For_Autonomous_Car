"""
JSON Formatter and Schema Validator for Non-LLM BanglaBERT-Base Predictions.
Assembles raw multi-head predictions into valid, standardized robot-action JSON.
"""

import json
from typing import Dict, Any, Tuple
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PREPROCESSING_EXPERIMENT.evaluation.robust_json_parser import parse_robot_json


def assemble_robot_json(
    pred_dict: Dict[str, Any],
    label_mappings: Dict[str, Any],
    use_setpoint_speeds: bool = True
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Assembles model head outputs into structured robot action JSON.
    """
    intents = label_mappings["intents"]
    targets = label_mappings["targets"]
    speed_cats = label_mappings["speed_categories"]
    atomic_actions = label_mappings["atomic_actions"]
    step_actions = label_mappings["step_actions"]
    linear_speeds = label_mappings["linear_speeds"]
    angular_speeds = label_mappings["angular_speeds"]
    step_counts = label_mappings["step_counts"]

    # 1. Intent
    intent_id = pred_dict.get("intent_id", 0)
    intent = intents[intent_id] if 0 <= intent_id < len(intents) else "MOVE_FORWARD"

    # 2. Target
    target_id = pred_dict.get("target_id", 0)
    target = targets[target_id] if 0 <= target_id < len(targets) else "none"

    # 3. Speed Category
    speed_cat_id = pred_dict.get("speed_cat_id", 0)
    speed_category = speed_cats[speed_cat_id] if 0 <= speed_cat_id < len(speed_cats) else "NORMAL"

    # 4. Step Count & Step Actions (Derived via Documented Ontology Rules)
    step_count_id = pred_dict.get("step_count_id", 0)
    raw_step_count = step_counts[step_count_id] if 0 <= step_count_id < len(step_counts) else 1

    s1_id = pred_dict.get("step1_id", 0)
    s2_id = pred_dict.get("step2_id", 0)
    s3_id = pred_dict.get("step3_id", 0)
    s4_id = pred_dict.get("step4_id", 0)

    s1 = atomic_actions[s1_id] if 0 <= s1_id < len(atomic_actions) else "MOVE_FORWARD"
    s2 = step_actions[s2_id] if 0 <= s2_id < len(step_actions) else "NONE"
    s3 = step_actions[s3_id] if 0 <= s3_id < len(step_actions) else "NONE"
    s4 = step_actions[s4_id] if 0 <= s4_id < len(step_actions) else "NONE"

    # Strict Robotic Ontology Alignment:
    # Rule 1: If Intent != 'SEQUENTIAL', the command is single-step. Action Sequence == [Intent], Step Count == 1.
    # Rule 2: If Intent == 'SEQUENTIAL', the command is multi-step (Step Count >= 2), assembled from step slots.
    if intent != "SEQUENTIAL":
        step_count = 1
        action_sequence = [intent]
    else:
        # Multi-step command
        step_count = max(2, raw_step_count)
        candidates = [s1, s2, s3, s4][:step_count]
        action_sequence = [s for s in candidates if s != "NONE"]
        if not action_sequence:
            action_sequence = [s1, s2 if s2 != "NONE" else "STOP"]
        step_count = len(action_sequence)

    # 5. Linear Speed
    if use_setpoint_speeds and "linear_speed_class_id" in pred_dict:
        lin_id = pred_dict["linear_speed_class_id"]
        linear_speed = float(linear_speeds[lin_id]) if 0 <= lin_id < len(linear_speeds) else 0.15
    else:
        linear_speed = float(pred_dict.get("linear_speed_reg", 0.15))

    # 6. Angular Speed
    if use_setpoint_speeds and "angular_speed_class_id" in pred_dict:
        ang_id = pred_dict["angular_speed_class_id"]
        angular_speed = float(angular_speeds[ang_id]) if 0 <= ang_id < len(angular_speeds) else 0.0
    else:
        angular_speed = float(pred_dict.get("angular_speed_reg", 0.0))

    action_plan = {
        "intent": intent,
        "target": target,
        "speed_category": speed_category,
        "linear_speed": round(linear_speed, 4),
        "angular_speed": round(angular_speed, 4),
        "action_sequence": action_sequence,
        "step_count": step_count,
    }

    # Format into standardized JSON string
    json_str = json.dumps(action_plan, ensure_ascii=False, indent=2)

    # Validate syntax using existing project validator
    parsed_obj, is_valid_json, err_msg = parse_robot_json(json_str)

    # Domain validation checks
    unsupported_target = target not in targets
    physical_limits_ok = (0.0 <= linear_speed <= 1.0) and (-2.0 <= angular_speed <= 2.0)

    validation_status = {
        "is_valid_json": is_valid_json,
        "syntax_error": err_msg if not is_valid_json else None,
        "has_unsupported_target": unsupported_target,
        "physical_limits_ok": physical_limits_ok,
        "semantic_warning": "Target unsupported" if unsupported_target else ("Speed out of bounds" if not physical_limits_ok else None),
    }

    return json_str, action_plan, validation_status
