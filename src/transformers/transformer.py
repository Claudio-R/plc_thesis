import math
import torch
import torch.nn as nn
from typing import Optional
# from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    """Linear layer used for input projections """

    def __init__(self, n_codebooks: int, codebook_size: int, d_model: int) -> None:
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embed_layers = nn.ModuleList([nn.Embedding(codebook_size, d_model) for _ in range(n_codebooks)])
        # self.embedding = nn.Embedding(codebook_size, d_model)

    def forward(self, x):
        """ x: (B, N, T) --> embeddings: (B, T, D) """
        embeddings = sum([embed(x[:, i, :]) for i, embed in enumerate(self.embed_layers)])
        return embeddings

class OutputProjections(nn.Module):
    """ Linear layer used for output projections """

    def __init__(self, n_codebooks: int, codebook_size: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.ModuleList([nn.Linear(d_model, codebook_size) for _ in range(n_codebooks)])

    def forward(self, x):
        """ x: (B, T, D) --> logits: (B, N, T, C) """
        logits = [proj(x) for proj in self.projection]
        logits = torch.stack(logits, dim=1) # logits.shape==(B, n_codebooks, S, C)
        return logits

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class ResCumSumLayer(nn.Module):
    def __init__(self, dim: int, mult: float = 1., dropout: float = 0.):
        super().__init__()
        dim_inner = int(mult * dim)
        self.block = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # input size: (B, seq_len, dim)
        x = torch.cumsum(x, dim=-2)
        x = self.block(x)
        return x


class Transformer(nn.Module):
    """ Transformer model """
    def __init__(self, config, n_codebooks, codebook_size, sample_rate, frame_dim):
        super().__init__()
        self.version = config["version"]
        self.d_model = config["transformer"]["d_model"]
        self.n_layers = config["transformer"]["n_layers"]
        self.n_heads = config["transformer"]["n_heads"]
        self.segment_duration = config["segment_dur"]
        self.context_length = int(sample_rate * config["segment_dur"] / frame_dim)

        self.input_embeddings = InputEmbeddings(n_codebooks, codebook_size, self.d_model)
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(self.d_model, self.n_heads, norm_first=True),
            num_layers=self.n_layers,
            norm=torch.nn.LayerNorm(self.d_model))
        self.output_projection = OutputProjections(n_codebooks, codebook_size, self.d_model)

    def forward(self, x):
        """ codes: (B, N, T) --> codes: (B, N, T) """
        mask = nn.Transformer.generate_square_subsequent_mask(sz=x.size(-1))
        x = self.input_embeddings(x)  # (B, N, T) --> (B, T, D)
        x = self.decoder(tgt=x, memory=x, tgt_mask=mask, memory_mask=mask, tgt_is_causal=True, memory_is_causal=True)  # (B, T, D) --> (B, T, D)
        x = self.output_projection(x)  # (B, T, D) --> (B, N, T, C)
        return x

    def predict(self, x):
        logits = self.forward(x)
        codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (B, n_codebooks, S, C)
        pred_codes = torch.argmax(codebook_index_probs, dim=-1)
        return logits, pred_codes
