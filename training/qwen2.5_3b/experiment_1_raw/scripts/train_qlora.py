import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import time

def main():
    # 1. Reproducibility
    set_seed(42)
    
    # Paths
    base_dir = r"C:\Users\Reyan\Downloads\Case_Study\training\qwen2.5_3b\experiment_1_raw"
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "outputs")
    report_dir = os.path.join(base_dir, "report")
    
    # 2. Load Dataset
    print("Loading datasets...")
    dataset = load_dataset(
        'json',
        data_files={
            'train': os.path.join(data_dir, "train.jsonl"),
            'validation': os.path.join(data_dir, "validation.jsonl")
        }
    )
    
    # 3. Model & Tokenizer Configuration
    model_name = "Qwen/Qwen2.5-3B"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Configuring BitsAndBytes for 4-bit QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print("Loading base model...")
    # Load model. We do not use device_map="auto" if we want to ensure it all goes to GPU 0 for a single 4GB GPU,
    # but the prompt requested device_map="auto", so we will use that.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    # 4. LoRA Configuration
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 5. Training Arguments
    # The prompt explicitly warns about fp16=True causing issues with bitsandbytes/AMP.
    # So we disable fp16 and bf16 to ensure stability.
    from trl import SFTTrainer, SFTConfig
    
    training_args = SFTConfig(
        output_dir=os.path.join(out_dir, "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        optim="paged_adamw_32bit",
        report_to="none",
        max_length=512
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
    )
    
    # 7. Train
    print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # 8. Save adapter
    final_adapter_dir = os.path.join(out_dir, "adapter")
    print(f"Saving final adapter to {final_adapter_dir}...")
    trainer.save_model(final_adapter_dir)
    
    # 9. Extract and save logs
    print("Extracting training history...")
    log_history = trainer.state.log_history
    history_df = pd.DataFrame(log_history)
    history_path = os.path.join(report_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    
    # Determine best eval loss
    eval_logs = [log for log in log_history if 'eval_loss' in log]
    if eval_logs:
        best_eval_loss = min([log['eval_loss'] for log in eval_logs])
    else:
        best_eval_loss = "N/A"
        
    final_train_loss = train_result.training_loss
    
    # Memory metrics
    peak_memory = "Unknown"
    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory = f"{peak_memory_bytes / (1024**3):.2f} GB"
        
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in trainer.model.parameters())
    
    # Save a summary txt
    summary = f"""TRAINING SUMMARY
================
Model: {model_name}
Training Time: {training_time:.2f} seconds
Final Train Loss: {final_train_loss}
Best Eval Loss: {best_eval_loss}
Trainable Parameters: {trainable_params}
Total Parameters: {all_params}
Peak Memory: {peak_memory}
GPU Used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
"""
    with open(os.path.join(report_dir, "training_summary.txt"), "w") as f:
        f.write(summary)
        
    print("Script finished successfully.")

if __name__ == "__main__":
    main()
