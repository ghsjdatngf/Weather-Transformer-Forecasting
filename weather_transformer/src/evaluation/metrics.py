import math
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_perplexity(model, data_loader, pad_id, vocab_size, device):
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for input_ids, target_ids, _context_ids, _t, _h in data_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
            n_real_tokens = (target_ids != pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += n_real_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def compute_classification_metrics(labels, preds, label_names):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm.tolist(), "labels": label_names,
    }


def collect_forecast_predictions(model, data_loader, device):
    """Runs the classification heads over a loader and returns predictions/labels
    for both Temp Category and Humidity Category."""
    model.eval()
    all_temp_preds, all_temp_labels = [], []
    all_hum_preds, all_hum_labels = [], []

    with torch.no_grad():
        for _input_ids, _target_ids, context_ids, temp_label, humidity_label in data_loader:
            context_ids = context_ids.to(device)
            temp_logits, hum_logits = model.classify(context_ids)

            all_temp_preds.extend(temp_logits.argmax(dim=-1).cpu().tolist())
            all_temp_labels.extend(temp_label.tolist())
            all_hum_preds.extend(hum_logits.argmax(dim=-1).cpu().tolist())
            all_hum_labels.extend(humidity_label.tolist())

    return all_temp_preds, all_temp_labels, all_hum_preds, all_hum_labels
