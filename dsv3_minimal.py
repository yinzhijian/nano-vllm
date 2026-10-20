"""Minimal DeepSeek-V3 forward pass for learning: pure torch, CPU, no KV cache, no TP, random weights.

Run: python dsv3_minimal.py            (set DEBUG=1 to print every intermediate shape)
Debug: put a breakpoint anywhere; all sizes are tiny.

Shape letters: B=batch, T=seq len, D=hidden, H=heads, N=B*T tokens, E=routed experts, K=top-k experts,
  C=kv_lora_rank, R=qk_rope_head_dim, dn=qk_nope_head_dim, dv=v_head_dim, Q=q_lora_rank
"""
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

DEBUG = os.environ.get("DEBUG", "0") == "1"


def dbg(name: str, x: torch.Tensor):
    if DEBUG:
        print(f"  {name:<14} {tuple(x.shape)} {x.dtype}")


@dataclass
class Config:
    vocab_size: int = 1000
    hidden_size: int = 64                # D
    num_layers: int = 3
    first_k_dense: int = 1               # first layers use a dense MLP, the rest use MoE
    num_heads: int = 4                   # H
    q_lora_rank: int = 32                # Q
    kv_lora_rank: int = 16               # C
    qk_nope_head_dim: int = 16           # dn
    qk_rope_head_dim: int = 8            # R
    v_head_dim: int = 16                 # dv
    dense_inter: int = 128
    moe_inter: int = 32
    n_routed_experts: int = 8            # E
    n_shared_experts: int = 1
    top_k: int = 2                       # K
    n_group: int = 4
    topk_group: int = 2
    routed_scaling: float = 2.5
    rope_theta: float = 10000.0
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # [..., dim] -> [..., dim]
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


def rope_cos_sin(T: int, R: int, theta: float):
    inv_freq = 1.0 / theta ** (torch.arange(0, R, 2).float() / R)    # [R/2]
    freqs = torch.outer(torch.arange(T).float(), inv_freq)           # [T, R/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, h, R] where adjacent dims (2i, 2i+1) form one rotation pair; cos/sin: [T, R/2]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]    # [1, T, 1, R/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]                        # each [B, T, h, R/2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.stack([y1, y2], dim=-1).flatten(-2)           # [B, T, h, R]


class MLA(nn.Module):
    """Multi-head Latent Attention, naive form: K/V are fully re-materialised from the latent every call."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        H, D = c.num_heads, c.hidden_size
        self.q_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
        self.q_a_proj = nn.Linear(D, c.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(c.q_lora_rank, c.eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, H * self.q_head_dim, bias=False)
        self.kv_a_proj = nn.Linear(D, c.kv_lora_rank + c.qk_rope_head_dim, bias=False)    # -> [c_kv | k_pe]
        self.kv_a_norm = RMSNorm(c.kv_lora_rank, c.eps)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, H * (c.qk_nope_head_dim + c.v_head_dim), bias=False)
        self.o_proj = nn.Linear(H * c.v_head_dim, D, bias=False)
        self.scale = self.q_head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: [B, T, D]
        c = self.c                                                      # 取出配置
        B, T, _ = x.shape                                               # 读取 batch 大小和序列长度
        H = c.num_heads                                                 # 注意力头数
        # KV cache 总览：只缓存 c_kv(归一化后) 和 k_pe(RoPE 后)，每 token 共 C+R 个数；q、k_nope、v 都不缓存，每步由 c_kv 现算。
        cos, sin = rope_cos_sin(T, c.qk_rope_head_dim, c.rope_theta)    # [T, R/2] each  预计算每个位置的 RoPE cos/sin  [有 cache 时：位置要取 past_len..past_len+T-1]

        # ---- query: D -> Q -> H*(dn+R)
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))              # [B, T, H*(dn+R)]  先降维到 Q，归一化，再升维到所有头
        q = q.view(B, T, H, self.q_head_dim)                            # [B, T, H, dn+R]  按头拆开
        q_nope, q_pe = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)    # [B,T,H,dn], [B,T,H,R]  分成不带位置和带位置两部分
        q_pe = apply_rope(q_pe, cos, sin)                               # [B, T, H, R]  只对 rope 部分施加旋转位置编码  [q 不缓存：只用当前 T 个新 token]

        # ---- key/value: D -> (C latent + R rope key); the latent is what an MLA KV cache would store
        kv = self.kv_a_proj(x)                                          # [B, T, C+R]  一次投影同时得到 latent 和 rope key
        c_kv, k_pe = kv.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)    # [B,T,C], [B,T,R]  拆成 latent 和 rope key
        c_kv = self.kv_a_norm(c_kv)                                     # [B, T, C]  对 latent 做 RMSNorm  [写入 cache：新 token 的 c_kv]
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin)                  # [B, T, 1, R]  shared by all heads  加一个 head 维后做 RoPE，所有头共用  [写入 cache：新 token 的 k_pe，已带位置信息]
        # [拼接点] c_kv = cat([cache_c_kv, c_kv], dim=1) -> [B, S, C]；k_pe = cat([cache_k_pe, k_pe], dim=1) -> [B, S, 1, R]；S = past_len + T

        kv_up = self.kv_b_proj(c_kv).view(B, T, H, c.qk_nope_head_dim + c.v_head_dim)    # [B, T, H, dn+dv]  把 latent 升维还原出每个头的 K/V  [有 cache 时对拼接后的全部 S 个 c_kv 升维，T->S]
        k_nope, v = kv_up.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)    # [B,T,H,dn], [B,T,H,dv]  拆成不带位置的 K 和 V  [不缓存，每步重算；吸收权重后可省掉这步]

        q = torch.cat([q_nope, q_pe], dim=-1)                           # [B, T, H, dn+R]  拼回完整的 query
        k = torch.cat([k_nope, k_pe.expand(-1, -1, H, -1)], dim=-1)     # [B, T, H, dn+R]  rope key 复制到每个头后与 k_nope 拼接  [有 cache 时 T->S，k_pe 来自拼接后的 cache]

        # ---- causal attention
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale     # [B, H, T(query), T(key)]  每个头的 q·k 点积并缩放
        causal = torch.ones(T, T, dtype=torch.bool).triu(1)             # [T, T], True = future position  上三角为 True，表示未来位置  [有 cache 时 mask 为 [T, S]，用 triu(1 + past_len)；decode 时 T=1 无需 mask]
        scores = scores.masked_fill(causal, float("-inf"))              # 把未来位置的分数设为 -inf，防止看到后面的 token
        probs = scores.float().softmax(-1).to(x.dtype)                  # [B, H, T, T]  在 key 维度上做 softmax（用 float32 保证精度）
        o = torch.einsum("bhts,bshv->bthv", probs, v)                   # [B, T, H, dv]  用注意力权重对 V 加权求和
        dbg("q", q); dbg("k", k); dbg("v", v); dbg("c_kv(latent)", c_kv); dbg("scores", scores); dbg("o", o)    # DEBUG 模式下打印中间 shape
        return self.o_proj(o.flatten(2))                                # [B, T, D]  合并所有头的输出并投影回 hidden 维度


class MLP(nn.Module):
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # [..., dim] -> [..., dim]
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    """Sigmoid router + aux-loss-free selection bias + group-limited top-k."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.randn(c.n_routed_experts, c.hidden_size) * 0.02)    # [E, D]
        self.bias = nn.Parameter(torch.zeros(c.n_routed_experts), requires_grad=False)       # [E], only used for selection

    def forward(self, x: torch.Tensor):    # x: [N, D]
        c = self.c
        N, E = x.size(0), c.n_routed_experts
        scores = F.linear(x.float(), self.weight.float()).sigmoid()    # [N, E]
        choice = scores + self.bias                                    # [N, E]
        grouped = choice.view(N, c.n_group, E // c.n_group)            # [N, G, E/G]
        group_score = grouped.topk(2, dim=-1).values.sum(-1)           # [N, G]  top-2 sum per group
        top_groups = group_score.topk(c.topk_group, dim=-1).indices    # [N, topk_group]
        group_mask = torch.zeros_like(group_score).scatter_(1, top_groups, 1.0)    # [N, G]
        expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape(N, E)    # [N, E]
        topk_idx = choice.masked_fill(expert_mask == 0, float("-inf")).topk(c.top_k, dim=-1).indices    # [N, K]
        topk_w = scores.gather(1, topk_idx)                            # [N, K]  weights use unbiased scores
        topk_w = topk_w / topk_w.sum(-1, keepdim=True) * c.routed_scaling
        return topk_w, topk_idx


class MoE(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate = Gate(c)
        self.experts = nn.ModuleList([MLP(c.hidden_size, c.moe_inter) for _ in range(c.n_routed_experts)])
        self.shared_experts = MLP(c.hidden_size, c.moe_inter * c.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: [B, T, D]
        B, T, D = x.shape
        flat = x.reshape(-1, D)                            # [N, D]
        topk_w, topk_idx = self.gate(flat)                 # [N, K], [N, K]
        out = self.shared_experts(flat)                    # [N, D]  shared expert sees every token
        for e, expert in enumerate(self.experts):
            token_ids, slot = (topk_idx == e).nonzero(as_tuple=True)    # tokens routed to expert e, and which of their K slots
            if token_ids.numel() == 0:
                continue
            y = expert(flat[token_ids])                    # [n_e, D]
            out.index_add_(0, token_ids, y * topk_w[token_ids, slot, None].to(y.dtype))
        dbg("topk_idx", topk_idx)
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, c: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.eps)
        self.attn = MLA(c)
        self.mlp_norm = RMSNorm(c.hidden_size, c.eps)
        self.mlp = MoE(c) if layer_idx >= c.first_k_dense else MLP(c.hidden_size, c.dense_inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # [B, T, D]
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DeepseekV3(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Block(c, i) for i in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:    # [B, T] -> logits [B, T, vocab]
        x = self.embed(input_ids)                                  # [B, T, D]
        for i, layer in enumerate(self.layers):
            if DEBUG:
                print(f"layer {i} ({type(layer.mlp).__name__})")
            x = layer(x)                                           # [B, T, D]
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # No KV cache: each step re-runs the whole sequence, so cost per step grows with length.
        for _ in range(max_new_tokens):
            logits = self(input_ids)                               # [B, T, vocab]
            next_id = logits[:, -1].argmax(-1, keepdim=True)       # [B, 1]
            input_ids = torch.cat([input_ids, next_id], dim=1)     # [B, T+1]
        return input_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    model = DeepseekV3(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))    # [B=2, T=6]
    with torch.no_grad():
        logits = model(ids)
    print("logits", tuple(logits.shape))
    print("generated", model.generate(ids[:1], 5).tolist())
