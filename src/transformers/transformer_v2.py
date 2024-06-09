import math
import torch
import torch.nn as nn
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

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


class LayerNormalization(nn.Module):
    # TODO: substitute with  torch.norm

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """ (B, T, D) --> (B, T, D) """
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
        std = x.std(dim=-1, keepdim=True)  # (B, T, 1)
        x = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return x


class FeedForwardBlock(nn.Module):
    """ Feed Forward Block """

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: nn.Module = nn.ReLU()) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ (B, T, D) --> (B, T, d_ff) --> (B, T, D) """
        return self.block(x)


class MultiHeadAttentionBlock(nn.Module):
    """ Multi-Head Attention Block with rotary embeddings """
    # New: output dropout is now part of the block. There are now two dropouts, dropout_attn for the attention scores,
    # and dropout for the output of the block. This respects your implementation; we might remove one of the two.
    # Also, d_model and d_attn are decoupled. Theoretically, you may want to compute attention in a higher- or lower-
    # dimensional space than what determined by d_model.
    def __init__(self, d_model: int, d_attn: Optional[int], n_heads: int, dropout: float, dropout_attn: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn if d_attn is not None else d_model
        self.n_heads = n_heads
        assert self.d_attn % self.n_heads == 0, "d_attn is not divisible by n_heads"

        # self.norm = LayerNormalization(self.d_model)  # New: norm is now part of attention block
        self.norm = nn.LayerNorm(self.d_model)

        self.d_h = self.d_attn // n_heads
        self.w_q = nn.Linear(self.d_model, self.d_attn, bias=False)  # Wq
        self.w_k = nn.Linear(self.d_model, self.d_attn, bias=False)  # Wk
        self.w_v = nn.Linear(self.d_model, self.d_attn, bias=False)  # Wv
        self.w_o = nn.Linear(self.d_attn, self.d_model, bias=False)  # Wo
        self.rotary_emb = RotaryEmbedding(dim=self.d_h)
        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout_attn)

    def attention(self, query, key, value, mask):
        """
            params:
            query: (B, h, T, d_h) = (1, 8, 10, 64)
            key: (B, h, T, d_h)
            value: (B, h, T, d_h)
            mask: (T, T)
            dropout: nn.Dropout

            returns:
            x: (B, h, T, d_h)
            attention_scores: (B, h, T, T)
        """
        # (B, h, T, d_h) --> (B, h, T, T)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_attn)
        # Apply additive masking
        if mask is not None:
            attention_scores = attention_scores + mask

        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.dropout_attn(attention_scores)
        # (B, h, T, T) @ (B, h, T, d_h) --> (B, h, T, d_h)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
            params:
            q: (B, T, D)
            k: (B, T, D)
            v: (B, T, D)
            mask: (T, T)

            returns:
            x: (B, T, D)
        """

        # Apply layer normalization
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)

        # Apply weights to q, k and v: (B, T, D) --> (B, T, D)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split into h heads: (B, T, D) --> (B, T, h, d_h) --> (B, h, T, d_h)
        query = query.view(*query.shape[:-1], self.n_heads, self.d_h).transpose(-3, -2)
        key = key.view(*key.shape[:-1], self.n_heads, self.d_h).transpose(-3, -2)
        value = value.view(*value.shape[:-1], self.n_heads, self.d_h).transpose(-3, -2)

        # Apply rotary embeddings to the query, key and value: (B, h, T, d_h) --> (B, h, T, d_h)
        query = self.rotary_emb.rotate_queries_or_keys(query)
        key = self.rotary_emb.rotate_queries_or_keys(key)

        # Calculate attention: (B, h, T, d_h) --> (B, h, T, d_h)
        x, __ = self.attention(query, key, value, mask)

        # Combine all the heads together: (B, h, T, d_h) --> (B, T, h, d_h) --> (B, T, D)
        x = x.transpose(-3, -2)
        x = x.contiguous()
        x = x.view(*x.shape[:-2], self.n_heads * self.d_h)

        # Multiply by Wo: (B, T, D) --> (B, T, D)
        x = self.w_o(x)

        x = self.dropout(x)

        return x


class TransformerDecoderBlock(nn.Module):
    """ Decoder Block
        1. Self-Attention Block
        2. Feed Forward Block
    """

    def __init__(self, d_model: int, d_attn: int, n_heads: int, dropout: float, dropout_attn: float) -> None:
        super().__init__()
        self.attention_block = MultiHeadAttentionBlock(d_model, d_attn, n_heads, dropout, dropout_attn)
        self.feed_forward_block = FeedForwardBlock(d_model, d_model * 4, dropout)

    def forward(self, x, mask):
        """
            params:
            x: (B, T, D)
            mask: (T, T)

            returns:
            x: (B, T, D)
        """
        # New: Moved here as we could think of a scenario in which x.size() changes between TransformerDecoderBlocks
        x = x + self.attention_block(x, x, x, mask)
        x = x + self.feed_forward_block(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_attn: int, n_heads: int, dropout: float, dropout_attn: float) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, d_attn, n_heads, dropout, dropout_attn) for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        """
            params:
            x: FloatTensor(B, T, D)

            returns:
            x: FloatTensor(B, T, D)
        """
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, mask)
        return x


class Transformer(nn.Module):
    """ Transformer model """
    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["codec"]["n_codebooks"]
        self.codebook_size = config["codec"]["codebook_size"]
        self.d_model = config["transformer"]["d_model"]
        self.d_attn = config["transformer"]["d_attn"]
        self.n_heads = config["transformer"]["n_heads"]
        self.n_layers = config["transformer"]["n_layers"]
        self.dropout = config["transformer"]["dropout"]
        self.dropout_attn = config["transformer"]["dropout_attn"]
        self.device = config["device"]
        self.context_length = config["codec"]["sample_rate"] * config["segment_dur"]

        self.input_embeddings = InputEmbeddings(self.n_codebooks, self.codebook_size, self.d_model)
        self.decoder = TransformerDecoder(self.n_layers, self.d_model, self.d_attn, self.n_heads, self.dropout, self.dropout_attn)
        self.output_projection = OutputProjections(self.n_codebooks, self.codebook_size, self.d_model)
    def forward(self, x):
        """ codes: (B, N, T) --> codes: (B, N, T) """
        # At inference time, T might be less than context length
        mask = nn.Transformer.generate_square_subsequent_mask(sz=x.size(-1), device=self.device)
        x = self.input_embeddings(x)  # (B, N, T) --> (B, T, D)
        x = self.decoder(x, mask)  # (B, T, D) --> (B, T, D)
        x = self.output_projection(x)  # (B, T, D) --> (B, N, T, C)
        return x

    def predict(self, x):
        logits = self.forward(x)
        codebook_index_probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: (B, n_codebooks, S, C)
        pred_codes = torch.argmax(codebook_index_probs, dim=-1)
        return pred_codes
