import torch
from torch import nn

from nanovllm.utils.context import get_context


class MLAttention(nn.Module):
    """Absorbed MLA attention over a paged latent cache.

    The cache stores only [c_kv (kv_lora_rank) | k_pe (rope_dim)] per token, shared by all heads
    (MQA-like), instead of per-head K/V. W_UK is folded into q and W_UV is applied after attention,
    so attention itself runs entirely in the latent space. Pure torch, per-sequence loop: for learning, not speed.
    """

    Q_CHUNK = 1024

    def __init__(self, num_heads: int, kv_lora_rank: int, rope_dim: int, scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim
        self.scale = scale
        # [num_blocks, block_size, kv_lora_rank + rope_dim]; assigned by ModelRunner.allocate_kv_cache
        self.latent_cache = torch.tensor([])

    def forward(
        self,
        q_latent: torch.Tensor,    # [N, H, kv_lora_rank]  (q_nope @ W_UK)
        q_pe: torch.Tensor,        # [N, H, rope_dim]
        c_kv: torch.Tensor,        # [N, kv_lora_rank]
        k_pe: torch.Tensor,        # [N, rope_dim]
    ) -> torch.Tensor:             # [N, H, kv_lora_rank]  (still latent, W_UV applied by caller)
        ctx = get_context()
        cache = self.latent_cache
        has_cache = cache.numel() > 0
        if has_cache:
            D = cache.size(-1)
            new_kv = torch.cat([c_kv, k_pe], dim=-1)
            cache.view(-1, D)[ctx.slot_mapping.long()] = new_kv
        q = torch.cat([q_latent, q_pe], dim=-1)    # [N, H, C+R]
        outs = []
        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q.tolist()
            cu_k = ctx.cu_seqlens_k.tolist()
            for i in range(len(cu_q) - 1):
                qi = q[cu_q[i]:cu_q[i + 1]]
                seqlen_k = cu_k[i + 1] - cu_k[i]
                if ctx.block_tables is None:    # no prefix cache: all keys are in the current batch
                    kv = torch.cat([c_kv, k_pe], dim=-1)[cu_q[i]:cu_q[i + 1]]
                else:
                    kv = self._gather(cache, ctx.block_tables[i], seqlen_k)
                outs.append(self._attend(qi, kv))
        else:
            lens = ctx.context_lens.tolist()
            for i, seqlen_k in enumerate(lens):
                kv = self._gather(cache, ctx.block_tables[i], seqlen_k)
                outs.append(self._attend(q[i:i + 1], kv))
        return torch.cat(outs, dim=0)

    @staticmethod
    def _gather(cache: torch.Tensor, block_table: torch.Tensor, seqlen_k: int) -> torch.Tensor:
        block_size = cache.size(1)
        blocks = block_table[:(seqlen_k + block_size - 1) // block_size].long()
        return cache[blocks].reshape(-1, cache.size(-1))[:seqlen_k]

    def _attend(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: [T, H, C+R], kv: [S, C+R]; queries are the last T positions of the S keys
        T, S = q.size(0), kv.size(0)
        c = kv[:, :self.kv_lora_rank]
        outs = []
        for s in range(0, T, self.Q_CHUNK):
            qc = q[s:s + self.Q_CHUNK]    # [t, H, C+R]
            t = qc.size(0)
            # latent score and rope score are fused into one matmul by concatenating along the feature dim
            scores = torch.einsum("thd,sd->hts", qc, kv).float() * self.scale
            q_pos = torch.arange(s, s + t, device=q.device) + (S - T)
            masked = torch.arange(S, device=q.device)[None, :] > q_pos[:, None]
            scores.masked_fill_(masked[None], float("-inf"))
            probs = scores.softmax(dim=-1).to(q.dtype)
            outs.append(torch.einsum("hts,sc->thc", probs, c))
        return torch.cat(outs, dim=0)
