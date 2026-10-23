"""A small character-level encoder-decoder with additive attention, built in PyTorch."""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, source):
        embedded = self.embedding(source)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, source_mask):
        # decoder_hidden: [1, batch, hidden]
        query = self.query(decoder_hidden[-1]).unsqueeze(1)
        keys = self.key(encoder_outputs)
        scores = self.energy(torch.tanh(keys + query)).squeeze(-1)
        scores = scores.masked_fill(~source_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_id, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.attention = attention
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_step(self, token, hidden, encoder_outputs, source_mask):
        embedded = self.embedding(token.unsqueeze(1))
        context, weights = self.attention(hidden, encoder_outputs, source_mask)
        decoder_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.gru(decoder_input, hidden)
        logits = self.output(torch.cat([output, context], dim=-1).squeeze(1))
        return logits, hidden, weights


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim=48,
                 hidden_dim=96, source_pad_id=0, target_pad_id=0):
        super().__init__()
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        self.encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim, source_pad_id)
        attention = AdditiveAttention(hidden_dim)
        self.decoder = Decoder(target_vocab_size, embedding_dim, hidden_dim,
                               target_pad_id, attention)

    def encode(self, source):
        source_mask = source.ne(self.source_pad_id)
        encoder_outputs, hidden = self.encoder(source)
        return encoder_outputs, hidden, source_mask

    def forward(self, source, target_input):
        encoder_outputs, hidden, source_mask = self.encode(source)
        logits = []
        for t in range(target_input.size(1)):
            step_logits, hidden, _ = self.decoder.forward_step(
                target_input[:, t], hidden, encoder_outputs, source_mask
            )
            logits.append(step_logits.unsqueeze(1))
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, source, bos_id, eos_id, max_length=32):
        self.eval()
        encoder_outputs, hidden, source_mask = self.encode(source)
        token = torch.full((source.size(0),), bos_id, dtype=torch.long,
                           device=source.device)
        generated = []
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_length):
            logits, hidden, _ = self.decoder.forward_step(
                token, hidden, encoder_outputs, source_mask
            )
            token = logits.argmax(dim=-1)
            generated.append(token)
            finished |= token.eq(eos_id)
            if finished.all():
                break
        return torch.stack(generated, dim=1) if generated else source.new_empty((source.size(0), 0))
