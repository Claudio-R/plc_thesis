import math
import torch
import torch.nn as nn
from typing import Optional
from rotary_embedding_torch import RotaryEmbedding


class InputProjections(nn.Module):
    """Linear layer used for input projections """

    def __init__(self, n_codebooks: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(n_codebooks, d_model, bias=False)

    def forward(self, x):
        """ (B, N, T) --> (B, T, N) --> (B, T, D) """
        x = x.transpose(-2, -1)  # In general, transposing with negative indices make it work also for unbatched tensors
        x = self.projection(x)
        return x


class OutputProjections(nn.Module):
    """ Linear layer used for output projections """

    def __init__(self, d_model: int, n_codebooks: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, n_codebooks, bias=False)

    def forward(self, x):
        """ (B, T, D) --> (B, T, N) --> (B, N, T) """
        x = self.projection(x)
        x = torch.sigmoid(x)  # Ensure that predicted codes take values in [0, 1] according to PLCModel.code_transform
        x = x.transpose(-2, -1)
        return x


class LayerNormalization(nn.Module):
    """ Add & Norm layer """

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
            LayerNormalization(d_model),
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

        self.norm = LayerNormalization(self.d_model)  # New: norm is now part of attention block

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
        # x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.n_heads * self.d_h)
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
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
            params:
            x: (B, T, D)
            mask: (T, T)

            returns:
            x: (B, T, D)
        """
        # New: Moved here as we could think of a scenario in which x.size() changes between TransformerDecoderBlocks
        mask = nn.Transformer.generate_square_subsequent_mask(sz=x.size(-2), device=self.device) # Generates an additive mask for the target sequence
        x = x + self.attention_block(x, x, x, mask)
        x = x + self.feed_forward_block(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_attn: int, n_heads: int, dropout: float, dropout_attn: float) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, d_attn, n_heads, dropout, dropout_attn) for _ in range(n_layers)
        ])

    def forward(self, x):
        """
            params:
            x: FloatTensor(B, T, D)

            returns:
            x: FloatTensor(B, T, D)
        """
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        return x


class Transformer(nn.Module):
    """ Transformer model """
    def __init__(
            self,
            n_codebooks: int = 9,
            codebook_size: int = 1024,
            d_model: int = 512,
            d_attn: Optional[int] = None,
            n_heads: int = 8,
            n_layers: int = 6,
            dropout: float = 0.1,
            dropout_attn: float = 0.0,
            input_length: int = 1024
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_length = input_length

        self.input_projection = InputProjections(n_codebooks, d_model)
        self.decoder = TransformerDecoder(n_layers, d_model, d_attn, n_heads, dropout, dropout_attn)
        self.output_projection = OutputProjections(d_model, n_codebooks) # (512, 1024) = 11.6 ms predetti per packet

    def forward(self, x):
        """ codes: (B, N, T) --> codes: (B, N, T) """
        x = self.input_projection(x)  # (B, N, T) --> (B, T, D)
        x = self.decoder(x)  # (B, T, D) --> (B, T, D)
        x = self.output_projection(x)  # (B, T, D) --> (B, N, T)
        return x
