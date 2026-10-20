import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear, ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.mla_attention import MLAttention
from nanovllm.layers.rotary_embedding import apply_rotary_emb


def yarn_mscale(scale: float, mscale: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


class YarnRotaryEmbedding(nn.Module):

    def __init__(self, rope_dim: int, max_position: int, base: float, rope_scaling: dict | None) -> None:
        super().__init__()
        freq_idx = torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim
        inv_freq = 1.0 / (base ** freq_idx)
        if rope_scaling:
            factor = rope_scaling["factor"]
            orig_max = rope_scaling["original_max_position_embeddings"]
            beta_fast, beta_slow = rope_scaling.get("beta_fast", 32), rope_scaling.get("beta_slow", 1)

            def correction_dim(num_rot):
                return rope_dim * math.log(orig_max / (num_rot * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
            ramp = ((torch.arange(rope_dim // 2, dtype=torch.float) - low) / max(high - low, 1e-3)).clamp(0, 1)
            extrapolation_mask = 1 - ramp    # high-frequency dims keep original freq, low-frequency dims are interpolated
            inv_freq = inv_freq / factor * (1 - extrapolation_mask) + inv_freq * extrapolation_mask
            attn_scale = yarn_mscale(factor, rope_scaling.get("mscale", 1)) / yarn_mscale(factor, rope_scaling.get("mscale_all_dim", 0))
        else:
            attn_scale = 1.0
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos() * attn_scale, freqs.sin() * attn_scale), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions: torch.Tensor, q_pe: torch.Tensor, k_pe: torch.Tensor):
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        return apply_rotary_emb(q_pe, cos, sin), apply_rotary_emb(k_pe, cos, sin)


def deinterleave(x: torch.Tensor) -> torch.Tensor:
    # DeepSeek checkpoints use interleaved (x0, y0, x1, y1, ...) rope dims; convert to the half-split layout.
    return x.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)


class DeepseekV3Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Low-rank q: hidden -> q_lora_rank -> heads * qk_head_dim (replicated down-proj, TP-sharded up-proj)
        self.q_a_proj = ReplicatedLinear(hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, self.total_num_heads * self.qk_head_dim, bias=False)
        # Joint kv: hidden -> [c_kv (kv_lora_rank) | k_pe (rope_dim)]; this is what gets cached
        self.kv_a_proj_with_mqa = ReplicatedLinear(hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank, self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.v_head_dim, hidden_size, bias=False)

        rope_scaling = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_scaling.get("rope_theta", getattr(config, "rope_theta", 10000))
        if rope_scaling.get("factor") is None:
            rope_scaling = None
        self.rotary_emb = YarnRotaryEmbedding(
            self.qk_rope_head_dim, config.max_position_embeddings, rope_theta, rope_scaling)
        scale = self.qk_head_dim ** -0.5
        if rope_scaling:
            m = yarn_mscale(rope_scaling["factor"], rope_scaling.get("mscale_all_dim", 0))
            scale *= m * m
        self.attn = MLAttention(self.num_heads, self.kv_lora_rank, self.qk_rope_head_dim, scale)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        c_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)
        q_pe, k_pe = self.rotary_emb(positions, deinterleave(q_pe), deinterleave(k_pe).unsqueeze(1))
        k_pe = k_pe.squeeze(1)
        # Weight absorption: fold W_UK into q so scores are computed directly against the cached latent c_kv.
        w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_uk, w_uv = w.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        q_latent = torch.einsum("nhd,hdc->nhc", q_nope, w_uk)
        o_latent = self.attn(q_latent, q_pe, c_kv, k_pe)
        o = torch.einsum("nhc,hvc->nhv", o_latent, w_uv)    # W_UV applied after attention
        return self.o_proj(o.flatten(1, -1))


class DeepseekV3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, reduce_results: bool = True) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, reduce_results=reduce_results)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class DeepseekV3Gate(nn.Module):
    """Sigmoid router with aux-loss-free bias and node-limited (group) top-k."""

    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float()).sigmoid()    # [N, E]
        # The bias only influences which experts get selected, not their mixing weights.
        choice = scores + self.e_score_correction_bias
        N, E = choice.shape
        group_scores = choice.view(N, self.n_group, E // self.n_group).topk(2, dim=-1).values.sum(-1)
        group_idx = group_scores.topk(self.topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        expert_mask = group_mask.unsqueeze(-1).expand(N, self.n_group, E // self.n_group).reshape(N, E)
        topk_idx = choice.masked_fill(expert_mask == 0, float("-inf")).topk(self.top_k, dim=-1).indices
        topk_w = scores.gather(1, topk_idx)
        if self.norm_topk_prob:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
        return topk_w * self.routed_scaling_factor, topk_idx


class DeepseekV3MoE(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.num_experts = config.n_routed_experts
        self.gate = DeepseekV3Gate(config)
        # Experts are TP-sharded along the intermediate dim (no expert parallelism); all-reduce is done once in forward.
        self.experts = nn.ModuleList([
            DeepseekV3MLP(config.hidden_size, config.moe_intermediate_size, reduce_results=False)
            for _ in range(self.num_experts)
        ])
        self.shared_experts = DeepseekV3MLP(
            config.hidden_size, config.moe_intermediate_size * config.n_shared_experts, reduce_results=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_w, topk_idx = self.gate(x)
        K = topk_idx.size(1)
        flat_idx = topk_idx.flatten()
        order = flat_idx.argsort()    # group (token, expert) pairs by expert
        token_ids = order // K
        weights = topk_w.flatten()[order].to(x.dtype)
        counts = torch.bincount(flat_idx, minlength=self.num_experts).tolist()
        out = self.shared_experts(x)
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            ids = token_ids[start:start + n]
            out.index_add_(0, ids, self.experts[e](x[ids]) * weights[start:start + n, None])
            start += n
        if self.tp_size > 1:
            dist.all_reduce(out)
        return out


class DeepseekV3DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV3Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = DeepseekV3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # The released checkpoint has an extra MTP layer at index num_hidden_layers, unused here.
        self.skipped_weight_prefixes = (f"model.layers.{config.num_hidden_layers}.",)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
