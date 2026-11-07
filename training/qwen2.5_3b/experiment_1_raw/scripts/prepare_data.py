import pandas as pd
import json
import os
import ast

def verify_dataset(df_path):
    print(f"Loading and verifying RAW dataset from: {df_path}")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"RAW dataset not found at {df_path}")
        
    df = pd.read_csv(df_path, encoding='utf-8')
    
    # Validation 1: 4000 rows
    if len(df) != 4000:
        raise ValueError(f"Expected 4000 rows, got {len(df)}")
        
    # Validation 2: No duplicate IDs
    if df['ID'].duplicated().any():
        raise ValueError("Duplicate IDs found in the dataset")
        
    # Validation 3: No missing required values
    required_cols = ['ID', 'Bangla Command', 'Intent', 'Target', 'Speed Category', 'Linear Speed (m/s)', 'Angular Speed (rad/s)', 'Action Sequence', 'Step Count']
    if df[required_cols].isnull().any().any():
        raise ValueError("Missing values found in required columns")
        
    print("[PASS] Dataset integrity verified (4000 rows, unique IDs, no missing required values).")
    return df

def format_prompt(row):
    # Parse Action Sequence list
    try:
        seq_str = str(row['Action Sequence'])
        if seq_str.startswith('['):
            # It's a string representation of a list
            action_sequence = ast.literal_eval(seq_str)
        else:
            # It's a single action or arrow separated, wait, let's check the RAW dataset
            # Actually, the user's prompt says:
            # 'action_sequence': ['MOVE_FORWARD']
            # Let's handle ' -> ' format if it exists, or list format
            if ' -> ' in seq_str:
                action_sequence = [s.strip() for s in seq_str.split('->')]
            elif ',' in seq_str and not seq_str.startswith('['):
                action_sequence = [s.strip() for s in seq_str.split(',')]
            else:
                action_sequence = [seq_str.strip()]
    except Exception:
        # Fallback to just wrapping it in a list if literal_eval fails
        seq_str = str(row['Action Sequence'])
        if ' -> ' in seq_str:
            action_sequence = [s.strip() for s in seq_str.split('->')]
        else:
            action_sequence = [seq_str.strip()]

    # Construct the JSON output string
    output_dict = {
        "intent": str(row['Intent']).strip(),
        "target": str(row['Target']).strip(),
        "speed_category": str(row['Speed Category']).strip(),
        "linear_speed": float(row['Linear Speed (m/s)']),
        "angular_speed": float(row['Angular Speed (rad/s)']),
        "action_sequence": action_sequence,
        "step_count": int(row['Step Count'])
    }
    
    # Format messages array
    messages = [
        {"role": "system", "content": "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan."},
        {"role": "user", "content": str(row['Bangla Command']).strip()},
        {"role": "assistant", "content": json.dumps(output_dict, ensure_ascii=False, indent=2)}
    ]
    
    return {"messages": messages}

def main():
    raw_csv_path = r"C:\Users\Reyan\Downloads\Case_Study\NEW dataset\data\dataset_v2.csv"
    out_dir = r"C:\Users\Reyan\Downloads\Case_Study\training\qwen2.5_3b\experiment_1_raw\data"
    
    df = verify_dataset(raw_csv_path)
    
    # Enforce existing official splits based on row indices
    # Train = 2800, Val = 600, Test = 600
    df_train = df.iloc[:2800].copy()
    df_val = df.iloc[2800:3400].copy()
    df_test = df.iloc[3400:4000].copy()
    
    if len(df_train) != 2800 or len(df_val) != 600 or len(df_test) != 600:
        raise ValueError("Split sizes do not match expected 2800/600/600")
        
    print("[PASS] Official split sizing preserved.")
    
    # Format and save
    datasets = {
        "train.jsonl": df_train,
        "validation.jsonl": df_val,
        "test.jsonl": df_test
    }
    
    for filename, split_df in datasets.items():
        out_path = os.path.join(out_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                formatted = format_prompt(row)
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                
        print(f"Saved {len(split_df)} formatted samples to {out_path}")
        
    print("Data preparation complete.")
    
    # Save a reference txt
    with open(os.path.join(out_dir, "dataset_reference.txt"), 'w', encoding='utf-8') as f:
        f.write("RAW Dataset V2\nPath: C:\\Users\\Reyan\\Downloads\\Case_Study\\NEW dataset\\data\\dataset_v2.csv\nSplits: Train=2800, Val=600, Test=600\n")

if __name__ == "__main__":
    main()
