import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3_5Config
import math
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class GatedRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float() * F.silu(gate.float())).to(x.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        kernel_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = self.key_dim * 2 + self.value_dim
        self.kernel_size = kernel_size

        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, self.num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, self.kernel_size, groups=conv_dim, bias=False
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.norm = GatedRMSNorm(self.value_head_dim, rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        print(f"Forward pass with hidden shape: {hidden.shape}, positions shape: {positions.shape}")
        batch, seq_len = hidden.shape

        qkv = self.in_proj_qkv(hidden).transpose(1, 2)
        qkv = F.conv1d(
            qkv,
            self.conv1d.weight,
            padding=self.kernel_size - 1,
            groups=qkv.shape[1],
        )[..., :seq_len]
        qkv = F.silu(qkv.transpose(1, 2))
        query, key, value = torch.split(
            qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, seq_len, self.num_value_heads, self.value_head_dim)

        if self.num_key_heads != self.num_value_heads:
            repeats = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        query = query.float()
        key = key.float()
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        value = value.float()
        beta = self.in_proj_b(hidden).float().sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden).float() + self.dt_bias.float()
        )

        state = torch.zeros(
            batch,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            device=hidden.device,
        )
        outputs = []
        for index in range(seq_len):
            q = query[:, index] / math.sqrt(self.key_head_dim)
            k = key[:, index]
            v = value[:, index]
            state = state * decay[:, index, :, None, None].exp()
            old_value = (state * k.unsqueeze(-1)).sum(dim=-2)
            delta = (v - old_value) * beta[:, index, :, None]
            state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q.unsqueeze(-1)).sum(dim=-2))

        outputs = torch.stack(outputs, dim=1).to(hidden.dtype)
        gate = self.in_proj_z(hidden).reshape(
            batch, seq_len, self.num_value_heads, self.value_head_dim
        )
        outputs = self.norm(outputs, gate).reshape(batch, seq_len, self.value_dim)
        return self.out_proj(outputs)


class Qwen3_5Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        attn_output_gate: bool = True,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.attn_output_gate = attn_output_gate

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size],
                dim=-1,
            )
            q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = q_gate.chunk(2, dim=-1)
            gate = gate.flatten(1, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o.flatten(1, -1)
        if self.attn_output_gate:
            o = o * torch.sigmoid(gate)
        output = self.o_proj(o)
        return output


class Qwen3_5MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, 'attention_bias', True),
                head_dim=getattr(config, 'head_dim', None),
                rope_scaling=getattr(config, "rope_parameters", None),
                attn_output_gate=getattr(config, "attn_output_gate", True),
            )
        else:
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        self.mlp = Qwen3_5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
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
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    def __init__(
        self,
        config: Qwen3_5Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
class LanguageModel(nn.Module):
    def __init__(
        self,
        config: Qwen3_5Config,
    ) -> None:
        super().__init__()
        self.language_model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.language_model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

class Qwen3_5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3_5Config
    ) -> None:
        super().__init__()
        self.model = LanguageModel(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)
