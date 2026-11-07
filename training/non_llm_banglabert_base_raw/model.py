"""
BanglaBERT-Base Multi-Head Architecture for Robot Command Parsing.
Encoder: csebuetnlp/banglabert (ELECTRA architecture, hidden_size=768, ~110M params)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BanglaBERTMultiHead(nn.Module):
    def __init__(self, model_name: str, label_mappings: dict, dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for banglabert Base

        self.label_mappings = label_mappings
        num_intents = len(label_mappings["intents"])
        num_targets = len(label_mappings["targets"])
        num_speed_cats = len(label_mappings["speed_categories"])
        num_step_counts = len(label_mappings["step_counts"])
        num_atomic = len(label_mappings["atomic_actions"])
        num_step_actions = len(label_mappings["step_actions"])
        num_linear_speeds = len(label_mappings["linear_speeds"])
        num_angular_speeds = len(label_mappings["angular_speeds"])

        self.dropout = nn.Dropout(dropout_prob)

        # 1. Intent Head
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_intents),
        )

        # 2. Target Head
        self.target_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_targets),
        )

        # 3. Speed Category Head
        self.speed_category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_speed_cats),
        )

        # 4. Step Count Head
        self.step_count_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_step_counts),
        )

        # 5. Slot Action Heads for Steps 1..4
        self.step1_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_atomic),
        )
        self.step2_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_step_actions),
        )
        self.step3_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_step_actions),
        )
        self.step4_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_step_actions),
        )

        # 6. Speed Setpoint Classification Heads
        self.linear_speed_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_linear_speeds),
        )
        self.angular_speed_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_angular_speeds),
        )

        # 7. Speed Continuous Regression Heads
        self.linear_reg_head = nn.Linear(hidden_size, 1)
        self.angular_reg_head = nn.Linear(hidden_size, 1)

        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: dict = None,
        loss_weights: dict = None,
    ):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        cls_rep = encoder_outputs[0][:, 0, :]
        features = self.dropout(cls_rep)

        intent_logits = self.intent_head(features)
        target_logits = self.target_head(features)
        speed_cat_logits = self.speed_category_head(features)
        step_count_logits = self.step_count_head(features)
        step1_logits = self.step1_head(features)
        step2_logits = self.step2_head(features)
        step3_logits = self.step3_head(features)
        step4_logits = self.step4_head(features)
        linear_speed_logits = self.linear_speed_head(features)
        angular_speed_logits = self.angular_speed_head(features)
        linear_reg_out = self.linear_reg_head(features).squeeze(-1)
        angular_reg_out = self.angular_reg_head(features).squeeze(-1)

        output_dict = {
            "intent_logits": intent_logits,
            "target_logits": target_logits,
            "speed_category_logits": speed_cat_logits,
            "step_count_logits": step_count_logits,
            "step1_logits": step1_logits,
            "step2_logits": step2_logits,
            "step3_logits": step3_logits,
            "step4_logits": step4_logits,
            "linear_speed_logits": linear_speed_logits,
            "angular_speed_logits": angular_speed_logits,
            "linear_reg": linear_reg_out,
            "angular_reg": angular_reg_out,
        }

        total_loss = None
        loss_breakdown = {}

        if labels is not None:
            weights = loss_weights or {
                "intent": 1.0,
                "target": 1.0,
                "speed_category": 0.8,
                "step_count": 0.8,
                "step1": 1.0,
                "step2": 0.8,
                "step3": 0.6,
                "step4": 0.4,
                "linear_speed": 1.0,
                "angular_speed": 1.0,
            }

            l_intent = self.ce_loss(intent_logits, labels["intent"])
            l_target = self.ce_loss(target_logits, labels["target"])
            l_speed_cat = self.ce_loss(speed_cat_logits, labels["speed_category"])
            l_step_count = self.ce_loss(step_count_logits, labels["step_count"])
            l_step1 = self.ce_loss(step1_logits, labels["step1"])
            l_step2 = self.ce_loss(step2_logits, labels["step2"])
            l_step3 = self.ce_loss(step3_logits, labels["step3"])
            l_step4 = self.ce_loss(step4_logits, labels["step4"])
            l_linear_speed = self.ce_loss(linear_speed_logits, labels["linear_speed_class"])
            l_angular_speed = self.ce_loss(angular_speed_logits, labels["angular_speed_class"])

            l_linear_reg = self.smooth_l1_loss(linear_reg_out, labels["linear_speed_val"])
            l_angular_reg = self.smooth_l1_loss(angular_reg_out, labels["angular_speed_val"])

            total_loss = (
                weights["intent"] * l_intent
                + weights["target"] * l_target
                + weights["speed_category"] * l_speed_cat
                + weights["step_count"] * l_step_count
                + weights["step1"] * l_step1
                + weights["step2"] * l_step2
                + weights["step3"] * l_step3
                + weights["step4"] * l_step4
                + weights["linear_speed"] * (l_linear_speed + 0.5 * l_linear_reg)
                + weights["angular_speed"] * (l_angular_speed + 0.5 * l_angular_reg)
            )

            loss_breakdown = {
                "loss_intent": l_intent.item(),
                "loss_target": l_target.item(),
                "loss_speed_cat": l_speed_cat.item(),
                "loss_step_count": l_step_count.item(),
                "loss_step1": l_step1.item(),
                "loss_step2": l_step2.item(),
                "loss_step3": l_step3.item(),
                "loss_step4": l_step4.item(),
                "loss_linear_speed": l_linear_speed.item(),
                "loss_angular_speed": l_angular_speed.item(),
                "loss_linear_reg": l_linear_reg.item(),
                "loss_angular_reg": l_angular_reg.item(),
                "total_loss": total_loss.item(),
            }

            output_dict["loss"] = total_loss
            output_dict["loss_breakdown"] = loss_breakdown

        return output_dict
