import torch
import torch.nn as nn
from src import config


class WeatherLoss:
    def __init__(self, pad_id: int, lm_weight: float = config.LM_LOSS_WEIGHT,
                 cls_weight: float = config.CLS_LOSS_WEIGHT, label_smoothing: float = 0.05):
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lm_weight = lm_weight
        self.cls_weight = cls_weight

    def compute(self, lm_logits, target_ids, temp_logits, temp_label, hum_logits, hum_label, vocab_size):
        lm_loss = self.lm_criterion(lm_logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        cls_loss = (
            self.cls_criterion(temp_logits, temp_label) + self.cls_criterion(hum_logits, hum_label)
        )
        total_loss = self.lm_weight * lm_loss + self.cls_weight * cls_loss
        return total_loss, lm_loss, cls_loss
