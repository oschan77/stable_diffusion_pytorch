import torch
from attention import SelfAttention
from torch import nn
from torch.nn import functional as F


class CLIP_Embedding(nn.Module):
    def __init__(self, n_vocab=49408, n_embd=768, n_token=77):
        super(CLIP_Embedding, self).__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        # tokens: (Batch_Size, Seq_Len)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = x + self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super(CLIPLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(normalized_shape=n_embd)
        self.attention1 = SelfAttention(n_heads=n_head, d_embed=n_embd)
        self.layernorm2 = nn.LayerNorm(normalized_shape=n_embd)
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # x: (Batch_Size, Seq_Len, Dim)

        # Self-Attention
        # residue: (Batch_Size, Seq_Len, Dim)
        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm1(x)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention1(x, causal=True)
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = x + residue

        # Feed Forward
        # residue: (Batch_Size, Seq_Len, Dim)
        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm2(x)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear1(x)
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear2(x)
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = x + residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.embedding = CLIP_Embedding(n_vocab=49408, n_embd=768, n_token=77)
        self.layers = nn.ModuleList(
            [CLIPLayer(n_head=12, n_embd=768) for _ in range(12)]
        )
        self.layernorm = nn.LayerNorm(normalized_shape=768)

    def forward(self, tokens):
        # tokens: (Batch_Size, Seq_Len)
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        out = self.layernorm(state)

        return out
