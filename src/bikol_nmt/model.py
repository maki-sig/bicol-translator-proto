import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, UNK, BOS, EOS = range(4)


def threads(count):
    torch.set_num_threads(count)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)


def length_normalizer(length, penalty):
    return ((5 + length) / 6) ** penalty


class MultiHeadAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q, self.k, self.v, self.out = [nn.Linear(width, width) for _ in range(4)]

    def forward(self, query, memory, mask):
        def heads(tensor):
            return tensor.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        output = scaled_dot_product_attention(heads(self.q(query)), heads(self.k(memory)),
                                              heads(self.v(memory)), mask,
                                              self.dropout if self.training else 0.)
        return self.out(output.transpose(1, 2).flatten(2))


class PositionalEncoding(nn.Module):
    def __init__(self, width, max_len):
        super().__init__()
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        angles = torch.arange(max_len)[:, None] * frequencies
        positions = torch.empty(max_len, width)
        positions[:, 0::2], positions[:, 1::2] = angles.sin(), angles[:, :width // 2].cos()
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, x):
        return x + self.positions[:x.size(1)]


class Layer(nn.Module):
    def __init__(self, width, heads, ffn, dropout, decoder=False):
        super().__init__()
        self.self_attention = MultiHeadAttention(width, heads, dropout)
        self.self_norm, self.ff_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.cross_attention = MultiHeadAttention(width, heads, dropout) if decoder else None
        self.cross_norm = nn.LayerNorm(width) if decoder else None
        self.ff = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, memory=None, source_mask=None):
        z = self.self_norm(x)
        x = x + self.dropout(self.self_attention(z, z, mask))
        if self.cross_attention is not None:
            x = x + self.dropout(self.cross_attention(self.cross_norm(x), memory, source_mask))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class TransformerNMT(nn.Module):
    def __init__(self, vocab_size, d_model=192, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=768, dropout=.25, max_len=128):
        super().__init__()
        if d_model % nhead or max_len < 2:
            raise ValueError("Invalid attention dimensions or maximum sequence length")
        self.max_len = max_len
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.positions = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(Layer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_encoder_layers))
        self.decoder = nn.ModuleList(Layer(d_model, nhead, dim_feedforward, dropout, decoder=True)
                                     for _ in range(num_decoder_layers))
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def embed(self, tokens):
        return self.dropout(self.positions(self.embedding(tokens) * self.scale))

    def encode(self, src):
        mask = src.ne(PAD)[:, None, None, :]
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.encoder_norm(x), mask

    def decode(self, trg, memory, source_mask, last_only=False):
        length = trg.size(1)
        causal = torch.ones((length, length), device=trg.device, dtype=torch.bool).tril()
        mask = trg.ne(PAD)[:, None, None, :] & causal
        x = self.embed(trg)
        for layer in self.decoder:
            x = layer(x, mask, memory, source_mask)
        if last_only:
            x = x[:, -1]
        return F.linear(self.decoder_norm(x), self.embedding.weight)

    def forward(self, src, trg):
        return self.decode(trg, *self.encode(src))

    @torch.inference_mode()
    def greedy_decode(self, source, max_len=48):
        memory, mask = self.encode(source)
        result = source.new_full((source.size(0), 1), BOS)
        done = source.new_zeros(source.size(0), dtype=torch.bool)
        for _ in range(min(max_len, self.max_len) - 1):
            logits = self.decode(result, memory, mask, last_only=True)
            logits[:, [PAD, UNK, BOS]] = -torch.inf
            result = torch.cat((result, logits.argmax(-1).masked_fill(done, EOS)[:, None]), dim=1)
            done |= result[:, -1].eq(EOS)
            if done.all():
                break
        return result

    @torch.inference_mode()
    def beam_search(self, source, beam_size=5, top_k=3, max_len=128, min_len=1, length_penalty=.6):
        memory, mask = self.encode(source)
        return [search(self, memory[i:i + 1], mask[i:i + 1], beam_size, top_k,
                       min(max_len, self.max_len), min_len, length_penalty)
                for i in range(source.size(0))]


def search(model, memory, mask, width, top_k, max_len, min_len, penalty):
    device = memory.device
    sequences = torch.tensor([[BOS]], device=device)
    scores = torch.zeros(1, device=device)
    done = []
    for length in range(1, max_len):
        count = len(sequences)
        logits = model.decode(sequences, memory.expand(count, -1, -1),
                              mask.expand(count, -1, -1, -1), last_only=True)
        probs = logits.log_softmax(-1)
        probs[:, [PAD, UNK, BOS]] = -torch.inf
        if length <= min_len:
            probs[:, EOS] = -torch.inf
        probabilities = probs + scores[:, None]
        normalized = length_normalizer(length, penalty)
        values, indices = probabilities.flatten().topk(min(2 * width, probabilities.numel()))
        active = []
        for value, index in zip(values.tolist(), indices.tolist()):
            if not math.isfinite(value):
                continue
            parent, token = divmod(index, probs.size(1))
            if token == EOS:
                done.append((tuple(sequences[parent, 1:].tolist()), value / normalized))
            elif length < max_len - 1 and len(active) < width:
                active.append((parent, token, value))
            if width == 1:
                break
        if not active:
            break
        parents, tokens, next_scores = zip(*active)
        sequences = torch.cat((sequences[list(parents)], torch.tensor(tokens, device=device)[:, None]), dim=1)
        scores = torch.tensor(next_scores, device=device)
        if len(done) >= top_k:
            bound = scores.max().item() / length_normalizer(max_len - 1, penalty)
            if sorted((score for _, score in done), reverse=True)[top_k - 1] >= bound:
                break
    dedup = {}
    for tokens, score in sorted(done, key=lambda item: -item[1]):
        if tokens:
            dedup.setdefault(tokens, score)
    return list(dedup.items())[:top_k]
