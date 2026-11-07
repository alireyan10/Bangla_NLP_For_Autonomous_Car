# Bangla Robot Command-to-Action JSON Dataset (v2)

This repository contains the curated and structured **Bangla Robot Voice & Text Command Dataset** for autonomous mobile robot navigation and ROS2 action planning.

---

## 1. Dataset Overview

* **Total Samples:** 4,000 commands
* **Languages:** Bengali (বাংলা) with English translations
* **Environments:** Hospital, Warehouse, Home, Office
* **Command Types:** CLEAN & NOISY
* **Benchmark Splits:**
  * **Train Set:** 2,800 samples (70%)
  * **Validation Set:** 600 samples (15%)
  * **Test Set (Held-Out Benchmark):** 600 samples (15%, IDs 3401–4000)

---

## 2. Directory Structure

```text
NEW dataset/
│
├── README.md                          # Dataset Documentation
└── data/
    ├── dataset_v2.csv                 # Primary Tabular Dataset (4,000 rows)
    ├── dataset_v2.json                # JSON format of the dataset
    └── dataset_v2.xlsx                # Excel Spreadsheet format
```

---

## 3. Data Schema & Column Dictionary

The dataset contains **14 feature columns** with zero missing/null values:

| Column Name | Data Type | Description & Example Values |
| :--- | :--- | :--- |
| `ID` | Integer | Unique identifier (1 to 4000) |
| `Environment` | String | Operational environment (`Hospital`, `Warehouse`, `Home`, etc.) |
| `Command Type` | String | Audio condition (`CLEAN`, `NOISY`) |
| `Bangla Command` | String | Raw Bengali voice command (e.g., `রোবট ধীরে টেবিলের কাছে যাও`) |
| `English Translation`| String | Semantic English translation |
| `Intent` | String | Primary intention (17 distinct classes: `MOVE_FORWARD`, `SEQUENTIAL`, etc.) |
| `Target` | String | Destination entity (36 physical classes: `table`, `chair`, `box`, `NONE`) |
| `Speed Category` | String | Kinematic modifier (`NORMAL`, `SLOW`, `FAST`, `STOP`) |
| `Linear Speed (m/s)`| Float | Motor linear velocity setpoint (`0.0`, `0.1`, `0.15`, `0.2`) |
| `Angular Speed (rad/s)`| Float | Motor turning velocity setpoint (`-0.2`, `-0.15`, `0.0`, `0.15`, `0.2`) |
| `Action Sequence` | String | Ordered execution steps (e.g., `MOVE_FORWARD -> ROTATE_LEFT`) |
| `Step Count` | Integer | Total steps in sequence (1 to 4) |
| `Difficulty` | String | Complexity rating (`EASY`, `MEDIUM`, `HARD`) |
| `Noise Type` | String | Acoustic noise category (`NONE`, `BACKGROUND_SPEECH`, `STREET_NOISE`) |

---

## 4. Output JSON Format (ROS2 Schema)

Each command maps to a deterministic JSON schema executable by ROS2 `/cmd_vel` motor controllers:

```json
{
  "intent": "SEQUENTIAL",
  "target": "table",
  "speed_category": "SLOW",
  "linear_speed": 0.1,
  "angular_speed": 0.0,
  "action_sequence": [
    "MOVE_TOWARD_TARGET",
    "STOP"
  ],
  "step_count": 2
}
```

---

## 5. Usage in Python

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/dataset_v2.csv')
print(f"Total commands loaded: {len(df)}")
print(df[['Bangla Command', 'Intent', 'Target', 'Speed Category']].head())
```
