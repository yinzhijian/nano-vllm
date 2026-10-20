import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)  # 当前 Triton 程序处理第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)  # 查询该 token 在 KV cache 中对应的物理槽位
    if slot == -1: return  # -1 表示此 token 不写入缓存（例如 warmup），直接结束当前程序
    key_offsets = idx * key_stride + tl.arange(0, D)  # 计算该 token 的 K 向量在输入张量中的元素偏移
    value_offsets = idx * value_stride + tl.arange(0, D)  # 计算该 token 的 V 向量在输入张量中的元素偏移
    key = tl.load(key_ptr + key_offsets)  # 读取该 token 的全部 KV head 和 head_dim 对应的 K 值
    value = tl.load(value_ptr + value_offsets)  # 读取该 token 的全部 KV head 和 head_dim 对应的 V 值
    cache_offsets = slot * D + tl.arange(0, D)  # 计算目标缓存槽位中整条 K/V 向量的元素偏移
    tl.store(k_cache_ptr + cache_offsets, key)  # 将 K 向量写入对应的缓存槽位
    tl.store(v_cache_ptr + cache_offsets, value)  # 将 V 向量写入对应的缓存槽位


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape  # 输入形状为 [token数, KV头数, 每头维度]
    D = num_heads * head_dim  # 一个 token 的 K 或 V 向量展平后的元素数
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # 要求每个 head 内的最后一维连续存储
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 要求相邻 KV head 在内存中紧密排列
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # 要求缓存中相邻槽位的向量间隔为 D 个元素
    assert slot_mapping.numel() == N  # 每个输入 token 必须有一个对应的缓存槽位
    # 启动 N 个 Triton 程序，每个程序负责一个 token 的 K/V 写入。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
