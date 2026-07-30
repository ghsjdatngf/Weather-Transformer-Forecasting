import json
import os
import torch
from torch.utils.data import DataLoader

from src import config
from src.tokenizer import WeatherTokenizer
from src.preprocessing import WeatherDataset
from src.models.weather_transformer import WeatherTransformerLM
from src.training.loss import WeatherLoss
from src.training.validation import validate
from src.evaluation.visualization import plot_loss_curves


def train(epochs: int = config.EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = WeatherTokenizer.load(config.VOCAB_PATH)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.vocab["<pad>"]

    train_ds = WeatherDataset(config.TRAIN_LM_CSV, config.TRAIN_FORECAST_CSV, tokenizer)
    val_ds = WeatherDataset(config.VAL_LM_CSV, config.VAL_FORECAST_CSV, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model_max_seq_len = max(config.MAX_SEQ_LEN, config.CLS_SEQ_LEN)
    model = WeatherTransformerLM(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        max_seq_len=model_max_seq_len,
        dropout=config.DROPOUT,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loss_fn = WeatherLoss(pad_id=pad_id)

    history = {
        "train_loss": [], "val_loss": [],
        "train_lm_loss": [], "val_lm_loss": [],
        "train_cls_loss": [], "val_cls_loss": [],
        "val_temp_acc": [], "val_hum_acc": [], "val_overall_acc": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_lm_loss, total_cls_loss, n_batches = 0.0, 0.0, 0.0, 0

        for input_ids, target_ids, context_ids, temp_label, humidity_label in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            context_ids = context_ids.to(device)
            temp_label = temp_label.to(device)
            humidity_label = humidity_label.to(device)

            optimizer.zero_grad()

            lm_logits = model(input_ids)
            temp_logits, hum_logits = model.classify(context_ids)

            loss, lm_loss, cls_loss = loss_fn.compute(
                lm_logits, target_ids,
                temp_logits, temp_label,
                hum_logits, humidity_label,
                vocab_size,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_cls_loss += cls_loss.item()
            n_batches += 1

        train_metrics = {
            "train_loss": total_loss / n_batches,
            "train_lm_loss": total_lm_loss / n_batches,
            "train_cls_loss": total_cls_loss / n_batches,
        }
        val_metrics = validate(model, val_loader, loss_fn, vocab_size, device)

        for k, v in train_metrics.items():
            history[k].append(v)
        for k, v in val_metrics.items():
            history[k].append(v)

        gap = val_metrics["val_loss"] - train_metrics["train_loss"]
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Train-Val Gap: {gap:+.4f}"
        )
        
        print(
            f"  Val Temp Acc: {val_metrics['val_temp_acc']:.4f} | "
            f"Val Hum Acc: {val_metrics['val_hum_acc']:.4f} | "
            f"Val Overall Acc: {val_metrics['val_overall_acc']:.4f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "d_model": config.D_MODEL,
                        "n_heads": config.N_HEADS,
                        "n_layers": config.N_LAYERS,
                        "d_ff": config.D_FF,
                        "max_seq_len": model_max_seq_len,
                        "dropout": config.DROPOUT,
                    },
                },
                config.BEST_MODEL_PATH,
            )
            print(f"  → Best model updated (val_loss = {best_val_loss:.4f})")

    # Save history and plots
    with open(config.LOSS_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    

    plot_loss_curves(history)
    print(
        f"\nTraining complete. Best model + loss curves saved in "
        f"'{config.CHECKPOINT_DIR}/' and '{config.PLOTS_DIR}/'"
    )


if __name__ == "__main__":
    train()
    