import torch
def validate(model, val_loader, loss_fn, vocab_size, device):
    model.eval()
    total_loss, total_lm_loss, total_cls_loss, n_batches = 0.0, 0.0, 0.0, 0

    temp_correct, temp_total = 0, 0
    hum_correct, hum_total = 0, 0

    with torch.no_grad():
        for input_ids, target_ids, context_ids, temp_label, humidity_label in val_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            context_ids = context_ids.to(device)
            temp_label, humidity_label = temp_label.to(device), humidity_label.to(device)

            lm_logits = model(input_ids)
            temp_logits, hum_logits = model.classify(context_ids)

            loss, lm_loss, cls_loss = loss_fn.compute(
                lm_logits, target_ids, temp_logits, temp_label, hum_logits, humidity_label, vocab_size
            )

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_cls_loss += cls_loss.item()
            n_batches += 1

            temp_preds = temp_logits.argmax(dim=-1)
            temp_correct += (temp_preds == temp_label).sum().item()
            temp_total += temp_label.size(0)

            hum_preds = hum_logits.argmax(dim=-1)
            hum_correct += (hum_preds == humidity_label).sum().item()
            hum_total += humidity_label.size(0)

    val_temp_acc = temp_correct / temp_total if temp_total > 0 else 0.0
    val_hum_acc = hum_correct / hum_total if hum_total > 0 else 0.0
    val_overall_acc = (val_temp_acc + val_hum_acc) / 2.0

    return {
        "val_loss": total_loss / n_batches,
        "val_lm_loss": total_lm_loss / n_batches,
        "val_cls_loss": total_cls_loss / n_batches,
        "val_temp_acc": val_temp_acc,
        "val_hum_acc": val_hum_acc,
        "val_overall_acc": val_overall_acc,
    }