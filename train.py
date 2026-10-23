"""Train and evaluate the toy symbol-to-reading sequence model on CPU or CUDA."""
import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import Seq2Seq

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "toy_pairs.tsv"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_pairs(path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        source, target = line.split("\t", maxsplit=1)
        pairs.append((source.strip().split(), target.strip()))
    if len(pairs) < 4:
        raise ValueError("Need at least four tab-separated source/target examples.")
    return pairs


def build_vocab(tokens):
    return {token: i for i, token in enumerate([PAD, BOS, EOS, UNK] + sorted(set(tokens)))}


def encode_source(signs, vocab):
    return [vocab.get(token, vocab[UNK]) for token in signs]


def encode_target(text, vocab):
    # Target is character-level so the decoder learns spelling/transduction.
    return [vocab[BOS]] + [vocab.get(char, vocab[UNK]) for char in text] + [vocab[EOS]]


class PairDataset(Dataset):
    def __init__(self, pairs, source_vocab, target_vocab):
        self.items = [
            (encode_source(source, source_vocab), encode_target(target, target_vocab))
            for source, target in pairs
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    max_src = max(len(src) for src, _ in batch)
    max_tgt = max(len(tgt) for _, tgt in batch)
    sources = torch.zeros((len(batch), max_src), dtype=torch.long)
    targets = torch.zeros((len(batch), max_tgt), dtype=torch.long)
    for i, (src, tgt) in enumerate(batch):
        sources[i, :len(src)] = torch.tensor(src)
        targets[i, :len(tgt)] = torch.tensor(tgt)
    return sources, targets


def decode(ids, inverse, eos_id):
    chars = []
    for idx in ids:
        token = inverse[int(idx)]
        if token == EOS:
            break
        if token not in (PAD, BOS, UNK):
            chars.append(token)
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save", type=Path, default=ROOT / "seq2seq_model.pt")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available in this PyTorch environment.")
    set_seed(args.seed)
    device = torch.device(args.device)

    pairs = read_pairs(args.data)
    random.shuffle(pairs)
    split = max(1, int(0.8 * len(pairs)))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    if not test_pairs:
        train_pairs, test_pairs = pairs[:-1], pairs[-1:]

    source_vocab = build_vocab(token for src, _ in pairs for token in src)
    target_vocab = build_vocab(char for _, text in pairs for char in text)
    source_inverse = {i: t for t, i in source_vocab.items()}
    target_inverse = {i: t for t, i in target_vocab.items()}

    train_loader = DataLoader(
        PairDataset(train_pairs, source_vocab, target_vocab),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        PairDataset(test_pairs, source_vocab, target_vocab),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    model = Seq2Seq(len(source_vocab), len(target_vocab),
                    source_pad_id=source_vocab[PAD],
                    target_pad_id=target_vocab[PAD]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab[PAD])

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for source, target in train_loader:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(source, target[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             target[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | loss={total_loss / batches:.4f}")

    model.eval()
    exact, total = 0, 0
    print("\nHELD-OUT EXAMPLES (tiny toy split; not a benchmark)")
    with torch.no_grad():
        for source, target in test_loader:
            source, target = source.to(device), target.to(device)
            predictions = model.greedy_decode(
                source, target_vocab[BOS], target_vocab[EOS],
                max_length=max(8, target.size(1) + 4)
            )
            for i in range(source.size(0)):
                pred = decode(predictions[i].cpu().tolist(), target_inverse, target_vocab[EOS])
                gold = decode(target[i, 1:].cpu().tolist(), target_inverse, target_vocab[EOS])
                exact += int(pred == gold)
                total += 1
                sign_text = " ".join(source_inverse[int(x)] for x in source[i].cpu().tolist()
                                     if int(x) != source_vocab[PAD])
                print(f"Signs: {sign_text}\nGold:  {gold}\nPred:  {pred}\n")
    print(f"Exact match: {exact}/{total} = {exact / max(total, 1):.1%}")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "config": {"embedding_dim": 48, "hidden_dim": 96}
    }, args.save)
    print(f"Saved checkpoint to: {args.save}")


if __name__ == "__main__":
    main()
