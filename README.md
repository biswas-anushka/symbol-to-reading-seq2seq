# Symbol-to-Reading Transduction with Seq2Seq Attention

A from-scratch learning project implementing a small encoder-decoder GRU with additive attention in PyTorch. The model maps sequences of symbolic tokens to character-level text output and runs on CPU by default.

## Important dataset note

The included examples are **synthetic toy data** created to test the code. They are not authentic Egyptian hieroglyphs, real readings, or a reproduction of the referenced challenge. To study actual hieroglyphic transliteration, replace the toy examples with a properly sourced, licensed dataset and document its schema and provenance.

## What is implemented

- Custom GRU encoder and decoder
- Additive attention implemented with PyTorch layers
- Character-level target vocabulary
- Teacher forcing during training
- Greedy autoregressive decoding
- Gradient clipping and deterministic seed setting
- Tiny held-out split and exact-match reporting
- CPU support; optional CUDA if the local PyTorch build supports it

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install PyTorch from the official selector if you need a specific CPU/CUDA build, then install requirements:

```bash
pip install -r requirements.txt
```

## Train on CPU

```bash
python train.py --device cpu --epochs 120
```

Optional CUDA:
```bash
python train.py --device cuda --epochs 120
```

The script prints training loss, predictions on a small held-out split, exact-match accuracy, and saves `seq2seq_model.pt`. The checkpoint is ignored by Git by default.

## Data format

Edit `data/toy_pairs.tsv`. Each non-comment line contains source symbols separated by spaces, a tab, and the target string:

```text
A1 B1<TAB>ka
```

The sample dataset is deliberately tiny and synthetic. Results on it do not indicate real-world transliteration quality.

## Suggested next steps

- Use a real, documented and licensed dataset.
- Create train/validation/test splits by headword or symbol combination to prevent leakage.
- Compare against a simple lookup-table baseline.
- Add teacher-forcing scheduling and beam search.
- Report character error rate and exact match on a meaningful held-out set.
- Profile CPU inference time with repeated runs and report the hardware/software setup.

## Resume wording

After running and understanding the implementation, a fair description is:

"Implemented a GRU encoder-decoder with additive attention in PyTorch for symbolic sequence-to-text transduction; added teacher-forced training, greedy decoding, and held-out evaluation on a synthetic dataset."

Update this statement only when it accurately reflects your implementation and results.
