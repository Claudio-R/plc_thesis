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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(1, 0)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)
        return self.dropout(x)

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
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.context_length)
        self.decoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, norm_first=True),
            num_layers=self.n_layers,
            norm=torch.nn.LayerNorm(self.d_model),
            mask_check=False,
        )
        self.output_projection = OutputProjections(n_codebooks, codebook_size, self.d_model)

    def forward(self, x):
        """ codes: (B, N, T) --> codes: (B, N, T) """
        mask = nn.Transformer.generate_square_subsequent_mask(sz=x.size(-1))
        x = self.input_embeddings(x)  # (B, N, T) --> (B, T, D)
        x = self.positional_encoding(x)
        x = self.decoder(x, mask, is_causal=True)  # (B, T, D) --> (B, T, D)
        x = self.output_projection(x)  # (B, T, D) --> (B, N, T, C)
        return x

    def predict(self, x):
        logits = self.forward(x)
        codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (B, n_codebooks, S, C)
        pred_codes = torch.argmax(codebook_index_probs, dim=-1)
        return logits, pred_codes
