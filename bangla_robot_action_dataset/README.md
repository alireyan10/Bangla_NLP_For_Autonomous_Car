# Bangla Robot Action Dataset

Synthetic Bangla natural-language to high-level robot-action dataset (1,600 samples) for indoor and outdoor mobile robots. It is designed around structured intent-to-action mapping, ordered behavior plans, and tunable motion parameters informed by the project literature on language-guided autonomy. Validate all commands, safety constraints, and speed values before real-world deployment.

## Fields

- **ID**: unique record identifier.
- **Environment**: Room, Classroom, Corridor, or Outdoor.
- **Bangla Command / English Translation**: paired natural-language instruction and verified translation.
- **Intent**: semantic command class.
- **Action**: controlled primary action label.
- **Action Sequence**: ordered high-level plan using ` → `, for example `FORWARD → TURN_LEFT`.
- **Speed Category / Linear Speed / Angular Speed**: initial tunable motion values. SLOW = 0.10 m/s and 0.20 rad/s; NORMAL = 0.15 m/s and 0.30 rad/s; FAST = 0.20 m/s and 0.40 rad/s. STOP is zero. Pure left turns have positive angular speed; pure right turns negative.
- **Direction, Distance, Target, Obstacle, Difficulty**: structured execution context.

## Labels

Actions: FORWARD, BACKWARD, STOP, TURN_LEFT, TURN_RIGHT, ROTATE_LEFT, ROTATE_RIGHT, INCREASE_SPEED, DECREASE_SPEED, MOVE_TOWARD, MOVE_NEAR, MOVE_IN_FRONT, MOVE_PAST. Difficulty is SIMPLE, MEDIUM, or HARD.

## Splits

Training: 1,120 (70%); Validation: 240 (15%); Testing: 240 (15%). The Excel workbook provides All_Data and split sheets; CSV/JSON/JSONL use identical records.

## Qwen3 fine-tuning

Use each JSONL line as a supervised example. A practical prompt contains Environment and Bangla Command; the target contains Intent, Action, Action Sequence, speed fields, target, and obstacle. Hold out `test.jsonl` for final evaluation, and validate structured output against the controlled label list before execution.
