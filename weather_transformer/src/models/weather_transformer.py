import torch
import torch.nn as nn
from src.models.embedding import WeatherEmbedding
from src.models.transformer_block import TransformerBlock


class WeatherTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512,
                 max_seq_len=256, dropout=0.1, n_temp_classes=3, n_humidity_classes=3):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.embedding = WeatherEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)
        self.temp_classifier = nn.Linear(d_model, n_temp_classes)
        self.humidity_classifier = nn.Linear(d_model, n_humidity_classes)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
        )

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        x = self.embedding(idx)
        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Language modeling forward pass -> (B, T, vocab_size) logits."""
        x = self.encode(idx)
        return self.lm_head(x)

    def classify(self, idx: torch.Tensor):
        """Forecasting forward pass -> (temp_logits, humidity_logits), each (B, n_classes).
        Uses mean-pooled representation over the input sequence (the past-days context)."""
        x = self.encode(idx)
        pooled = x.mean(dim=1)
        return self.temp_classifier(pooled), self.humidity_classifier(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = WeatherTransformerLM(vocab_size=1000, max_seq_len=64)
    dummy = torch.randint(0, 1000, (2, 40))
    print("LM logits:", model(dummy).shape)
    print("Classify logits:", [t.shape for t in model.classify(dummy)])
    print(f"Parameters: {model.count_parameters():,}")
