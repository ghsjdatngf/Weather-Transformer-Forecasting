import json
import re
from src import config

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def simple_tokenize(text: str):
    text = text.lower()
    return re.findall(r"\d+\.\d+|\d+|[a-z]+|[^\sa-z0-9]", text)


class WeatherTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()} if vocab else {}

    def build_vocab(self, texts):
        counter = {}
        for line in texts:
            for tok in simple_tokenize(line):
                counter[tok] = counter.get(tok, 0) + 1

        sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])
        vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for tok, _ in sorted_tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        return self.vocab

    def encode(self, text, add_special_tokens=True):
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        if add_special_tokens:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        return ids

    def encode_fixed(self, text, seq_len, add_special_tokens=True):
        """Encode and pad/truncate to a fixed length (used for batching)."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        pad_id = self.vocab["<pad>"]
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad_id] * (seq_len - len(ids))
        return ids

    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        tokens = [t for t in tokens if t not in ("<pad>", "<bos>", "<eos>")]
        return " ".join(tokens)

    def save(self, path=None):
        path = path or config.VOCAB_PATH
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path=None):
        path = path or config.VOCAB_PATH
        with open(path, "r") as f:
            vocab = json.load(f)
        return cls(vocab)

    def __len__(self):
        return len(self.vocab)


if __name__ == "__main__":
    with open(config.TRAIN_LM_CSV.replace(".csv", ".txt")) as f:
        train_lines = f.read().split("\n")

    tokenizer = WeatherTokenizer()
    tokenizer.build_vocab(train_lines)
    tokenizer.save()
    print(f"Vocabulary size: {len(tokenizer)} (saved to {config.VOCAB_PATH})")
