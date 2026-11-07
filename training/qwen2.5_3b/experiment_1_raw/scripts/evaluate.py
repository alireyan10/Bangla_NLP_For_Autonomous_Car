import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
import ast

def main():
    base_dir = r"C:\Users\Reyan\Downloads\Case_Study\training\qwen2.5_3b\experiment_1_raw"
    adapter_path = os.path.join(base_dir, "outputs", "adapter")
    raw_csv_path = r"C:\Users\Reyan\Downloads\Case_Study\NEW dataset\data\dataset_v2.csv"
    out_file = os.path.join(base_dir, "evaluation", "predictions.jsonl")
    
    # 1. Load Data
    print(f"Loading raw dataset from {raw_csv_path}", flush=True)
    df = pd.read_csv(raw_csv_path, encoding='utf-8')
    df_test = df.iloc[3400:4000].copy()
    print(f"Test set size: {len(df_test)}", flush=True)
    
    # Check for existing progress
    existing_ids = set()
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data['id'])
                except:
                    pass
        print(f"Found existing predictions.jsonl with {len(existing_ids)} completed samples.", flush=True)
    
    # 2. Load Model & Tokenizer
    model_name = "Qwen/Qwen2.5-3B"
    print(f"Loading tokenizer {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading base model in 4-bit...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # 3. Inference Loop
    print("Starting inference...", flush=True)
    total_latency = 0
    processed_count = len(existing_ids)
    
    # Open file in append mode
    with open(out_file, 'a', encoding='utf-8') as out_f:
        for idx, row in df_test.iterrows():
            if row['ID'] in existing_ids:
                continue
                
            cmd = str(row['Bangla Command']).strip()
            
            messages = [
                {"role": "system", "content": "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan."},
                {"role": "user", "content": cmd}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            start_t = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_t = time.time()
            latency = end_t - start_t
            total_latency += latency
            
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            seq_str = str(row['Action Sequence'])
            if ' -> ' in seq_str:
                action_sequence = [s.strip() for s in seq_str.split('->')]
            else:
                action_sequence = [seq_str.strip()]
                
            gt_dict = {
                "intent": str(row['Intent']).strip(),
                "target": str(row['Target']).strip(),
                "speed_category": str(row['Speed Category']).strip(),
                "linear_speed": float(row['Linear Speed (m/s)']),
                "angular_speed": float(row['Angular Speed (rad/s)']),
                "action_sequence": action_sequence,
                "step_count": int(row['Step Count'])
            }
            
            res = {
                "id": row['ID'],
                "command": cmd,
                "ground_truth": gt_dict,
                "prediction_raw": prediction_str,
                "latency": latency
            }
            
            out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
            out_f.flush() # Force write to disk immediately
            
            processed_count += 1
            if (processed_count % 10) == 0:
                print(f"Processed {processed_count}/600 (Recent latency: {latency:.2f}s)", flush=True)
                
    if (processed_count - len(existing_ids)) > 0:
        avg_latency = total_latency / (processed_count - len(existing_ids))
        print(f"Inference complete. New avg latency: {avg_latency:.3f}s/sample", flush=True)
    else:
        print("All samples already evaluated.", flush=True)

if __name__ == "__main__":
    main()
