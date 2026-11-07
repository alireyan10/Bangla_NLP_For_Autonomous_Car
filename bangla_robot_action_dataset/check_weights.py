import torch
from transformers import AutoModel
import os

def check():
    base_model = AutoModel.from_pretrained("csebuetnlp/banglabert_small")
    base_weight = base_model.encoder.layer[0].attention.self.query.weight.data.clone()
    
    # Load Exp 3
    exp3 = torch.load(r"C:\Users\Reyan\Downloads\Case_Study\bangla_robot_action_dataset\training\banglabert_small\raw\outputs\best_model\model_weights.bin", map_location="cpu", weights_only=True)
    
    # Load Exp 4
    exp4 = torch.load(r"C:\Users\Reyan\Downloads\Case_Study\bangla_robot_action_dataset\training\banglabert_small\preprocessed\outputs\best_model\model_weights.bin", map_location="cpu", weights_only=True)
    
    exp3_weight = exp3["bert.encoder.layer.0.attention.self.query.weight"]
    exp4_weight = exp4["bert.encoder.layer.0.attention.self.query.weight"]
    
    print("Exp 3 changed from base:", not torch.equal(base_weight, exp3_weight))
    print("Exp 4 changed from base:", not torch.equal(base_weight, exp4_weight))

check()
