import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- RMSNorm ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


# --- Rotary PE ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos()[None, None, :, :])
        self.register_buffer('sin', emb.sin()[None, None, :, :])

    def forward(self, seq_len: int):
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]


def apply_rotary(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)

    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        h_norm = self.attn_norm(x)
        b, n, _ = h_norm.shape
        q = self.to_q(h_norm).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h_norm).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(h_norm).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(b, n, -1)
        x = x + self.dropout(self.to_out(out))
        h2 = self.ffn_norm(x)
        x = x + self.dropout(self.ffn(h2))
        return x


# --- HierarchicalBert with ACT ---
class HierarchicalBert(nn.Module):
    def __init__(self, vocab_size, dim=256, layers=8, heads=4,
                 max_seq=512, local_win=128, enable_act=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.local_win = local_win
        self.enable_act = enable_act
        self.embed = nn.Embedding(vocab_size, dim)
        self.type_embed = nn.Embedding(2, dim)
        self.embed_norm = RMSNorm(dim)
        # RoPE is initialized once with the maximum possible sequence length
        self.rope_g = RotaryPositionalEncoding(dim // heads, max_seq, base=160000)
        self.rope_l = RotaryPositionalEncoding(dim // heads, max_seq, base=10000)
        half = layers // 2
        self.low = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(half)])
        self.high = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers - half)])
        self.z_L0 = nn.Parameter(torch.zeros(1, 1, dim))
        self.z_H0 = nn.Parameter(torch.zeros(1, 1, dim))
        self.final_norm = RMSNorm(dim)
        self.mlm_head = nn.Linear(dim, vocab_size)
        if self.enable_act:
            self.q_head = nn.Linear(dim, 2)

    def forward(self, input_ids, token_type_ids, attn_mask, N=2, T=4):
        b, n = input_ids.shape
        x = self.embed(input_ids) + self.type_embed(token_type_ids)
        x = self.embed_norm(x)
        pad = (1 - attn_mask)[:, None, None, :] * -1e9
        band = torch.ones((n, n), device=x.device).triu(-self.local_win // 2).tril(self.local_win // 2)
        local = torch.full((n, n), float('-inf'), device=x.device).masked_fill(band.bool(), 0.0)
        masks = {'g': pad, 'l': pad + local[None, None]}

        zL = self.z_L0.expand(b, n, -1)
        zH = self.z_H0.expand(b, 1, -1)

        logits_list = []
        q_list = []

        for seg in range(N):
            if seg == 0:
                h_low = x + zL + zH.expand(b, n, -1)
            else:
                h_low = zL + zH.expand(b, n, -1)

            for _ in range(T):
                for i, blk in enumerate(self.low):
                    is_g = (i + 1) % 3 == 0
                    cos, sin = self.rope_g(n) if is_g else self.rope_l(n)
                    mask_type = masks['g'] if is_g else masks['l']
                    h_low = blk(h_low, cos, sin, mask_type)

            zL = h_low

            agg = zL.mean(1, keepdim=True)
            hH = zH + agg
            cos1, sin1 = self.rope_g(1)
            for blk in self.high:
                hH = blk(hH, cos1, sin1, None)
            zH = hH

            out = self.final_norm(zL + zH.expand(b, n, -1))
            logits = self.mlm_head(out)
            logits_list.append(logits)

            if self.enable_act:
                qv = self.q_head(zH.squeeze(1))
                q_list.append(qv)

            if seg < N - 1:
                zL = zL.detach()
                zH = zH.detach()

        return logits_list, q_list
