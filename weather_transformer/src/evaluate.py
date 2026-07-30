import os
import json
import torch
from torch.utils.data import DataLoader

from src import config
from src.tokenizer import WeatherTokenizer
from src.preprocessing import WeatherDataset
from src.models.weather_transformer import WeatherTransformerLM
from src.evaluation.metrics import (
    compute_perplexity,
    compute_classification_metrics,
    collect_forecast_predictions,
)
from src.evaluation.visualization import plot_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load best model
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model_config = checkpoint["config"]

    tokenizer = WeatherTokenizer.load(config.VOCAB_PATH)
    pad_id = tokenizer.vocab["<pad>"]

    model = WeatherTransformerLM(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test data
    test_ds = WeatherDataset(config.TEST_LM_CSV, config.TEST_FORECAST_CSV, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Language modeling metrics
    avg_lm_loss, perplexity = compute_perplexity(
        model, test_loader, pad_id, model_config["vocab_size"], device
    )
    print(f"\nTest Cross-Entropy Loss: {avg_lm_loss:.4f}")
    print(f"Perplexity:               {perplexity:.4f}")

    # Classification metrics
    temp_preds, temp_labels, hum_preds, hum_labels = collect_forecast_predictions(
        model, test_loader, device
    )

    results = {
        "lm_loss": avg_lm_loss,
        "perplexity": perplexity,
    }

    for name, preds, labels, label_map in [
        ("Temperature Category", temp_preds, temp_labels, config.TEMP_LABELS),
        ("Humidity Category", hum_preds, hum_labels, config.HUMIDITY_LABELS),
    ]:
        label_names = list(label_map.keys())
        metrics = compute_classification_metrics(labels, preds, label_names)

        print(f"\n{name}:")
        print(f"  Accuracy  = {metrics['accuracy']:.4f}")
        print(f"  Precision = {metrics['precision']:.4f}")
        print(f"  Recall    = {metrics['recall']:.4f}")
        print(f"  F1 Score  = {metrics['f1']:.4f}")

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            label_names,
            f"{name} Confusion Matrix",
            f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        )
        results[name] = metrics

    # Save report
    report_path = os.path.join(config.REPORTS_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()