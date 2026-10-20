"""DeepSeek-V4 最简前向实现（学习用）：纯 torch、CPU、无 KV cache、无 TP、随机权重。

结构依据官方 DeepSeek-V4-Flash 的 inference/model.py 与 kernel.py（hc_split_sinkhorn、sparse_attn）。
为了可读，做了这些简化：不做 FP8/FP4 量化、不做 Hadamard 旋转、不做 YaRN（只保留两套 rope_theta）、
不含 MTP 层、输出所有位置的 logits（官方只取最后一个位置）。配置缩小到 CPU 可调试的规模。

运行：python dsv4_minimal.py          （DEBUG=1 会打印每层中间 shape）
调试：任意位置打断点即可，所有张量都很小。

与 V3 相比的关键变化：
  1. 注意力只有 1 个 KV 头（MQA），且 K 与 V 是同一个张量 kv；输出还要做逆 RoPE，再走"分组低秩"的 o 投影。
  2. 三种注意力层（由 compress_ratios 决定）：
       ratio=0   纯滑窗注意力（SWA）
       ratio=4   CSA：滑窗 + 压缩 KV（Compressor）+ Indexer 选 top-k 压缩块
       ratio>4   HCA：滑窗 + 重度压缩 KV（无 Indexer，所有已完成的压缩块都可见）
  3. attn_sink：每个头一个可学习的"汇聚"logit，只加到 softmax 分母，不贡献 value。
  4. mHC 超连接：隐状态是 hc_mult 份拷贝 [B,T,hc,D]，用 Sinkhorn 约束的混合矩阵代替普通残差。
  5. MoE：sqrt(softplus) 打分；前 n_hash_layers 层按 token id 查表路由（hash routing）。

维度符号：B=batch, T=序列长, D=hidden, H=头数, hd=head_dim, rd=rope_head_dim, Q=q_lora_rank,
  G=o_groups, hc=hc_mult, win=滑窗大小, r=压缩比, Tc=T//r 压缩块数, S=T+Tc 注意力的 key 总数,
  Hi/di=indexer 头数/头维度, E=路由专家数, K=每 token 激活专家数, N=B*T
"""
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
    dim: int = 64                          # D
    n_layers: int = 4
    n_hash_layers: int = 1                 # 前几层 MoE 用 hash 路由
    n_heads: int = 4                       # H
    head_dim: int = 16                     # hd（最后 rd 维带 RoPE，其余不带）
    rope_head_dim: int = 8                 # rd
    q_lora_rank: int = 32                  # Q
    o_groups: int = 2                      # G
    o_lora_rank: int = 16
    window_size: int = 4                   # win（真实模型 128）
    compress_ratios: tuple = (0, 4, 8, 0)  # 每层的压缩比；0=纯滑窗，4=CSA(带 indexer)，其他=HCA（真实模型是 4 和 128）
    rope_theta: float = 10000.0            # 纯滑窗层的 rope base
    compress_rope_theta: float = 160000.0  # 带压缩层的 rope base
    index_n_heads: int = 2                 # Hi
    index_head_dim: int = 16               # di
    index_topk: int = 3                    # 每个 query 最多选几个压缩块
    hc_mult: int = 4                       # hc
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    n_routed_experts: int = 8              # E
    n_shared_experts: int = 1
    n_activated_experts: int = 2           # K
    moe_inter_dim: int = 32
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # [..., dim] -> [..., dim]
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


# ---------------------------------------------------------------- RoPE（相邻两维为一对，与官方复数乘法一致）
def rope_cos_sin(T: int, rd: int, theta: float):
    inv_freq = 1.0 / theta ** (torch.arange(0, rd, 2).float() / rd)    # [rd/2]
    freqs = torch.outer(torch.arange(T).float(), inv_freq)             # [T, rd/2]
    return freqs.cos(), freqs.sin()


def rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    # x: [B, t, h, hd]，只旋转最后 rd 维；cos/sin: [t, rd/2]；inverse=True 做逆旋转（用于注意力输出 o）
    if inverse:
        sin = -sin
    nope, pe = x[..., :-rd], x[..., -rd:]                              # [B,t,h,hd-rd], [B,t,h,rd]
    c, s = cos[None, :, None, :], sin[None, :, None, :]                # [1, t, 1, rd/2]
    x1, x2 = pe[..., 0::2], pe[..., 1::2]                              # 各 [B, t, h, rd/2]
    pe = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2)    # [B, t, h, rd]
    return torch.cat([nope, pe], dim=-1)                               # [B, t, h, hd]


# ---------------------------------------------------------------- Compressor：把每 r 个 token 门控池化成 1 个压缩 KV
class Compressor(nn.Module):
    def __init__(self, c: Config, ratio: int, head_dim: int):
        super().__init__()
        self.c, self.ratio, self.head_dim = c, ratio, head_dim
        self.overlap = ratio == 4                                      # ratio=4 时用重叠窗口，边界更平滑
        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.randn(ratio, coff * head_dim) * 0.02)    # [r, coff*hd] 窗口内位置偏置
        self.wkv = nn.Linear(c.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(c.dim, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, c.norm_eps)

    def overlap_transform(self, t: torch.Tensor, value: float) -> torch.Tensor:
        # t: [B, Tc, r, 2*hd] -> [B, Tc, 2r, hd]；前 r 个位置放"上一窗口"的前半维，后 r 个放"当前窗口"的后半维
        B, Tc, r, _ = t.shape
        d = self.head_dim
        out = t.new_full((B, Tc, 2 * r, d), value)
        out[:, :, r:] = t[..., d:]
        out[:, 1:, :r] = t[:, :-1, :, :d]
        return out

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):    # x: [B, T, D]
        B, T, _ = x.shape
        r = self.ratio
        Tc = T // r                                                    # 完整压缩块数；末尾不足 r 个的 token 暂不压缩
        if Tc == 0:
            return None
        cutoff = Tc * r
        kv = self.wkv(x[:, :cutoff]).unflatten(1, (Tc, r))             # [B, Tc, r, coff*hd]
        score = self.wgate(x[:, :cutoff]).unflatten(1, (Tc, r)) + self.ape    # [B, Tc, r, coff*hd]
        if self.overlap:
            kv = self.overlap_transform(kv, 0.0)                       # [B, Tc, 2r, hd]
            score = self.overlap_transform(score, float("-inf"))       # [B, Tc, 2r, hd]
        kv = (kv * score.softmax(dim=2)).sum(dim=2)                    # [B, Tc, hd]  窗口内按 softmax 权重池化
        kv = self.norm(kv)                                             # [B, Tc, hd]
        # 压缩块的 RoPE 位置取该块第一个 token 的位置：0, r, 2r, ...
        kv = rope_tail(kv.unsqueeze(2), cos[:cutoff:r], sin[:cutoff:r], self.c.rope_head_dim).squeeze(2)
        return kv                                                      # [B, Tc, hd]


# ---------------------------------------------------------------- Indexer：给每个 query 在压缩块里选 top-k（CSA 专用）
class Indexer(nn.Module):
    def __init__(self, c: Config, ratio: int):
        super().__init__()
        self.c, self.ratio = c, ratio
        self.wq_b = nn.Linear(c.q_lora_rank, c.index_n_heads * c.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(c.dim, c.index_n_heads, bias=False)
        self.compressor = Compressor(c, ratio, c.index_head_dim)       # indexer 自己的一套压缩 key（官方还会做 Hadamard 旋转，这里省略）

    def forward(self, x: torch.Tensor, qr: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, Tc: int):
        c, r = self.c, self.ratio
        B, T, _ = x.shape
        q = self.wq_b(qr).view(B, T, c.index_n_heads, c.index_head_dim)            # [B, T, Hi, di]
        q = rope_tail(q, cos, sin, c.rope_head_dim)                                # [B, T, Hi, di]
        idx_kv = self.compressor(x, cos, sin)                                      # [B, Tc, di]
        w = self.weights_proj(x) * (c.index_head_dim ** -0.5 * c.index_n_heads ** -0.5)    # [B, T, Hi]
        score = torch.einsum("bthd,bcd->bthc", q, idx_kv)                          # [B, T, Hi, Tc]
        score = (score.relu() * w.unsqueeze(-1)).sum(dim=2)                        # [B, T, Tc]
        visible = torch.arange(Tc)[None, :] < (torch.arange(T)[:, None] + 1) // r  # [T, Tc] 压缩块必须已完整结束才可见
        score = score.masked_fill(~visible, float("-inf"))
        topk = score.topk(min(c.index_topk, Tc), dim=-1).indices                   # [B, T, k]
        valid = torch.gather(visible.expand(B, T, Tc), 2, topk)                    # [B, T, k] 选中项是否真的可见
        sel = torch.zeros(B, T, Tc, dtype=torch.int32)
        sel.scatter_add_(2, topk, valid.int())                                     # [B, T, Tc]
        dbg("idx_kv", idx_kv); dbg("topk", topk)
        return sel > 0                                                             # [B, T, Tc] 每个 query 选中的压缩块


# ---------------------------------------------------------------- Attention
class Attention(nn.Module):
    def __init__(self, c: Config, layer_id: int):
        super().__init__()
        self.c = c
        self.ratio = c.compress_ratios[layer_id]
        H, hd = c.n_heads, c.head_dim
        self.attn_sink = nn.Parameter(torch.zeros(H))                          # [H]
        self.wq_a = nn.Linear(c.dim, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.norm_eps)
        self.wq_b = nn.Linear(c.q_lora_rank, H * hd, bias=False)
        self.wkv = nn.Linear(c.dim, hd, bias=False)                            # 只有 1 个 KV 头，K=V
        self.kv_norm = RMSNorm(hd, c.norm_eps)
        self.wo_a = nn.Linear(H * hd // c.o_groups, c.o_groups * c.o_lora_rank, bias=False)    # 分组低秩 o 投影
        self.wo_b = nn.Linear(c.o_groups * c.o_lora_rank, c.dim, bias=False)
        self.scale = hd ** -0.5
        if self.ratio:
            self.compressor = Compressor(c, self.ratio, hd)
            self.indexer = Indexer(c, self.ratio) if self.ratio == 4 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: [B, T, D]
        c = self.c
        B, T, _ = x.shape
        H, hd, rd, win, r = c.n_heads, c.head_dim, c.rope_head_dim, c.window_size, self.ratio
        theta = c.compress_rope_theta if r else c.rope_theta
        cos, sin = rope_cos_sin(T, rd, theta)                                  # [T, rd/2]  有 cache 时位置应取 past_len..past_len+T-1

        # ---- query（不缓存：只用当前 T 个新 token）
        qr = self.q_norm(self.wq_a(x))                                         # [B, T, Q]   官方还把 qr 送进 indexer，qr 不缓存
        q = self.wq_b(qr).view(B, T, H, hd)                                    # [B, T, H, hd]
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + c.norm_eps).to(q.dtype)    # 每个头做无权重 RMSNorm
        q = rope_tail(q, cos, sin, rd)                                         # [B, T, H, hd]

        # ---- 滑窗 KV：单头，K 与 V 共用
        kv = self.kv_norm(self.wkv(x))                                         # [B, T, hd]
        kv = rope_tail(kv.unsqueeze(2), cos, sin, rd).squeeze(2)               # [B, T, hd]
        # 【缓存A·滑窗KV】kv 最近 win 个 token，环形缓冲 [B, win, hd]；token t 写入槽位 t % win

        # ---- 滑窗可见性：query t 只看 [t-win+1, t] 的原始 token
        pos = torch.arange(T)
        window_mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - win)    # [T, T]
        mask = window_mask.expand(B, T, T)                                     # [B, T, T]

        # ---- 压缩 KV（CSA / HCA 层）
        K = kv                                                                 # [B, T, hd]
        if r:
            kv_c = self.compressor(x, cos, sin)                                # [B, Tc, hd]；T<r 时为 None
            # 【缓存C·Compressor 状态】kv_state / score_state：还没凑满 r 个 token 的 wkv、wgate 输出
            #   [B, coff*r, coff*hd]（overlap 时还要保留上一窗口的前半维），凑满 r 个才产出 1 个压缩块
            if kv_c is not None:
                Tc = kv_c.size(1)
                # 【缓存B·压缩KV】kv_c 每完成一个 r-token 块追加 1 条 [B, Tc, hd]
                if self.indexer is not None:
                    sel = self.indexer(x, qr, cos, sin, Tc)                    # [B, T, Tc]  CSA：只看 indexer 选中的块
                    # 【缓存D·Indexer 压缩 key】indexer.compressor 的输出 idx_kv [B, Tc, di] 也要缓存
                else:
                    sel = (torch.arange(Tc)[None, :] < (pos[:, None] + 1) // r).expand(B, T, Tc)    # [B, T, Tc] HCA：所有已完成块都可见
                # 【拼接点】K = cat([窗口 cache 里的 kv, 压缩 cache 里的 kv_c])；官方用 offset 把压缩块下标排在窗口之后
                K = torch.cat([kv, kv_c], dim=1)                               # [B, S, hd]  S = T + Tc
                mask = torch.cat([mask, sel], dim=2)                           # [B, T, S]

        # ---- 注意力：K 同时当 V，attn_sink 只进分母
        scores = torch.einsum("bthd,bsd->bhts", q, K) * self.scale             # [B, H, T, S]
        scores = scores.masked_fill(~mask[:, None], float("-inf"))
        sink = self.attn_sink.view(1, H, 1, 1).expand(B, H, T, 1)              # [B, H, T, 1]
        probs = torch.cat([scores, sink], dim=-1).float().softmax(-1)[..., :-1].to(x.dtype)    # [B, H, T, S]
        o = torch.einsum("bhts,bsd->bthd", probs, K)                           # [B, T, H, hd]
        o = rope_tail(o, cos, sin, rd, inverse=True)                           # 逆 RoPE：value 里带了位置旋转，输出时转回去

        # ---- 分组低秩输出投影
        G = c.o_groups
        o = o.reshape(B, T, G, H * hd // G)                                    # [B, T, G, H*hd/G]
        wo_a = self.wo_a.weight.view(G, c.o_lora_rank, -1)                     # [G, o_lora, H*hd/G]
        o = torch.einsum("btgd,grd->btgr", o, wo_a)                            # [B, T, G, o_lora]
        dbg("q", q); dbg("K", K); dbg("scores", scores); dbg("mask", mask)
        return self.wo_b(o.flatten(2))                                         # [B, T, D]


# ---------------------------------------------------------------- MoE
class Gate(nn.Module):
    def __init__(self, c: Config, layer_id: int):
        super().__init__()
        self.c = c
        self.hash = layer_id < c.n_hash_layers
        self.weight = nn.Parameter(torch.randn(c.n_routed_experts, c.dim) * 0.02)    # [E, D]
        if self.hash:   # hash 路由：每个 token id 对应固定的 K 个专家
            self.register_buffer("tid2eid", torch.rand(c.vocab_size, c.n_routed_experts).topk(c.n_activated_experts).indices)    # [vocab, K]
        else:
            self.bias = nn.Parameter(torch.zeros(c.n_routed_experts))          # [E] 只影响选择，不影响权重

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):    # x: [N, D], input_ids: [N]
        scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()    # [N, E]
        original = scores
        if not self.hash:
            scores = scores + self.bias
            indices = scores.topk(self.c.n_activated_experts, dim=-1).indices   # [N, K]
        else:
            indices = self.tid2eid[input_ids]                                   # [N, K]
        weights = original.gather(1, indices)                                   # [N, K] 权重用未加 bias 的分数
        weights = weights / weights.sum(-1, keepdim=True) * self.c.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter: int, swiglu_limit: float):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)
        self.limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:    # [n, D] -> [n, D]
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:                                                      # SwiGLU 截断，防止激活过大
            up = up.clamp(-self.limit, self.limit)
            gate = gate.clamp(max=self.limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = h * weights
        return self.w2(h.to(x.dtype))


class MoE(nn.Module):
    def __init__(self, c: Config, layer_id: int):
        super().__init__()
        self.c = c
        self.gate = Gate(c, layer_id)
        self.experts = nn.ModuleList([Expert(c.dim, c.moe_inter_dim, c.swiglu_limit) for _ in range(c.n_routed_experts)])
        self.shared_experts = Expert(c.dim, c.moe_inter_dim, c.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:    # x: [B, T, D]
        shape = x.shape
        x = x.reshape(-1, self.c.dim)                                           # [N, D]
        weights, indices = self.gate(x, input_ids.flatten())                    # [N, K], [N, K]
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            idx, slot = torch.where(indices == e)                               # 路由到专家 e 的 token 及其所在 slot
            if idx.numel():
                y[idx] += expert(x[idx], weights[idx, slot, None].to(x.dtype))
        y = y + self.shared_experts(x)
        dbg("indices", indices)
        return y.view(shape)


# ---------------------------------------------------------------- Block（mHC 超连接）
def hc_split_sinkhorn(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float):
    # mixes: [B, T, (2+hc)*hc] -> pre [B,T,hc], post [B,T,hc], comb [B,T,hc,hc]（comb 经 Sinkhorn 近似双随机）
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))    # [B, T, hc, hc]
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):                                                  # 行、列交替归一化
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class Block(nn.Module):
    def __init__(self, c: Config, layer_id: int):
        super().__init__()
        self.c = c
        self.attn = Attention(c, layer_id)
        self.ffn = MoE(c, layer_id)
        self.attn_norm = RMSNorm(c.dim, c.norm_eps)
        self.ffn_norm = RMSNorm(c.dim, c.norm_eps)
        mix_hc = (2 + c.hc_mult) * c.hc_mult
        self.hc_attn_fn = nn.Parameter(torch.randn(mix_hc, c.hc_mult * c.dim) * 0.02)    # [mix_hc, hc*D]
        self.hc_ffn_fn = nn.Parameter(torch.randn(mix_hc, c.hc_mult * c.dim) * 0.02)
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.ones(3))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3))

    def hc_pre(self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        # x: [B, T, hc, D]；把 hc 份拷贝按动态权重 pre 合成 1 份送进子层
        shape = x.shape
        flat = x.flatten(2).float()                                             # [B, T, hc*D]
        mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.c.norm_eps)    # [B, T, mix_hc]
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, self.c.hc_mult, self.c.hc_sinkhorn_iters, self.c.hc_eps)
        y = (pre.unsqueeze(-1) * flat.view(shape)).sum(dim=2)                   # [B, T, D]
        return y.to(x.dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [B,T,D]（子层输出）, residual: [B,T,hc,D]；新的 hc 份 = post*子层输出 + comb 混合后的旧 hc 份
        return post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)    # [B, T, hc, D]

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:    # x: [B, T, hc, D]
        residual = x
        h, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)    # [B, T, D]
        h = self.attn(self.attn_norm(h))                                        # [B, T, D]
        x = self.hc_post(h, residual, post, comb)                               # [B, T, hc, D]

        residual = x
        h, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(h), input_ids)                               # [B, T, D]
        return self.hc_post(h, residual, post, comb)                            # [B, T, hc, D]


class DeepseekV4(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList([Block(c, i) for i in range(c.n_layers)])
        self.norm = RMSNorm(c.dim, c.norm_eps)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.hc_head_fn = nn.Parameter(torch.randn(c.hc_mult, c.hc_mult * c.dim) * 0.02)    # [hc, hc*D]
        self.hc_head_base = nn.Parameter(torch.zeros(c.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.ones(1))

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:    # [B, T, hc, D] -> [B, T, D]，把 hc 份合并回 1 份
        flat = x.flatten(2).float()                                             # [B, T, hc*D]
        mixes = F.linear(flat, self.hc_head_fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.c.norm_eps)    # [B, T, hc]
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.c.hc_eps
        return (pre.unsqueeze(-1) * flat.view(x.shape)).sum(dim=2).to(x.dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:    # [B, T] -> logits [B, T, vocab]
        h = self.embed(input_ids)                                               # [B, T, D]
        h = h.unsqueeze(2).repeat(1, 1, self.c.hc_mult, 1)                      # [B, T, hc, D]  复制成 hc 份
        for i, layer in enumerate(self.layers):
            if DEBUG:
                print(f"layer {i} compress_ratio={layer.attn.ratio} hash_route={layer.ffn.gate.hash}")
            h = layer(h, input_ids)                                             # [B, T, hc, D]
        return self.head(self.norm(self.hc_head(h)))                            # [B, T, vocab]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # 无 KV cache：每步都重跑整段序列；有 cache 时每步只需喂 1 个新 token（见 Attention 里的【缓存】标注）
        for _ in range(max_new_tokens):
            next_id = self(input_ids)[:, -1].argmax(-1, keepdim=True)           # [B, 1]
            input_ids = torch.cat([input_ids, next_id], dim=1)                  # [B, T+1]
        return input_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    model = DeepseekV4(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))    # [B=2, T=32]
    with torch.no_grad():
        logits = model(ids)
        print("logits", tuple(logits.shape))
        # 因果性自检：改最后一个 token，之前位置的 logits 不应变化（用来验证 mask / 压缩块可见性写对了）
        ids2 = ids.clone()
        ids2[:, -1] = (ids2[:, -1] + 1) % cfg.vocab_size
        diff = (model(ids2)[:, :-1] - logits[:, :-1]).abs().max().item()
        print("causal check, max diff on earlier positions:", diff)
    print("generated", model.generate(ids[:1, :6], 5).tolist())
