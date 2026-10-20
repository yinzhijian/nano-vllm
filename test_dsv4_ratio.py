"""dsv4_minimal.py 中 compress_ratio = 0 / 4 / 8 三类注意力层的单测。

运行：python test_dsv4_ratio.py    或    python -m pytest test_dsv4_ratio.py -v

核心思路：不去窥探内部 mask，而是用"黑盒依赖矩阵"验证行为。
  依赖矩阵 dep[t, j] = 把输入位置 j 的 token 换掉后，输出位置 t 是否变化。
  每种 ratio 的"哪些 token 能被看见"是确定的，因此可以写出期望矩阵，逐项比对。
"""
import torch

from dsv4_minimal import Attention, Compressor, Config, rope_cos_sin, rope_tail

CFG = Config()    # compress_ratios=(0, 4, 8, 0)，window_size=4，index_topk=3
D, HD, RD, WIN = CFG.dim, CFG.head_dim, CFG.rope_head_dim, CFG.window_size


def make_attn(ratio: int) -> Attention:
    layer_id = CFG.compress_ratios.index(ratio)    # 0->层0，4->层1，8->层2
    torch.manual_seed(0)
    return Attention(CFG, layer_id).eval()


@torch.no_grad()
def dependency_matrix(attn: Attention, T: int) -> torch.Tensor:
    """dep[t, j] = 替换 x[:, j] 后输出位置 t 是否变化（阈值 1e-6）。"""
    torch.manual_seed(1)
    x = torch.randn(1, T, D)
    base = attn(x)                                                      # [1, T, D]
    dep = torch.zeros(T, T, dtype=torch.bool)
    for j in range(T):
        x2 = x.clone()
        x2[:, j] = torch.randn(D)                                       # 只改第 j 个 token
        diff = (attn(x2) - base).abs().amax(-1)[0]                      # [T] 每个输出位置的变化量
        dep[:, j] = diff > 1e-6
    return dep


def window_visible(T: int) -> torch.Tensor:
    t, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    return (j <= t) & (j > t - WIN)                                     # query t 只看 [t-WIN+1, t]


# ---------------------------------------------------------------- ratio = 0：纯滑窗
def test_ratio0_pure_sliding_window():
    """原理：ratio=0 的层没有 compressor / indexer，K 只有当前序列的原始 kv，mask 就是滑窗。
    因此 out[t] 只依赖 x[t-WIN+1 .. t]；窗口之外的 token 哪怕换掉，输出必须逐位不变。
    这也是纯滑窗层的 KV cache 只需保留最近 WIN 个 token（环形缓冲）的根本原因。"""
    attn = make_attn(0)
    assert not hasattr(attn, "compressor") and not hasattr(attn, "indexer")
    T = 16
    assert torch.equal(dependency_matrix(attn, T), window_visible(T))


# ---------------------------------------------------------------- ratio = 8：HCA（重度压缩，无 indexer）
def test_ratio8_hca_sees_window_and_all_completed_blocks():
    """原理：HCA 每 8 个 token 池化成 1 个压缩块（不重叠），没有 indexer，所以 query t 能看到
      (a) 滑窗内的原始 token，以及 (b) 所有"已完整结束"的压缩块 c，条件 (c+1)*8 <= t+1，即 c < (t+1)//8。
    压缩块 c 只由 token [8c, 8c+8) 池化而来，所以 token j 对 out[t] 的额外影响条件是 j//8 < (t+1)//8。
    关键边界：块 0 在 t=7 才完成，所以 t=4..6 时 token 0 已滑出窗口、块 0 又未完成 → 此时必须"看不到"token 0；
    t=7 起重新可见。这个边界保证了压缩块不会泄漏尚未到来的 token。"""
    attn = make_attn(8)
    assert attn.indexer is None and attn.compressor.overlap is False
    T = 16
    t, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    expected = window_visible(T) | ((j // 8) < (t + 1) // 8)
    dep = dependency_matrix(attn, T)
    assert torch.equal(dep, expected)
    assert not dep[4:7, 0].any() and dep[7, 0]                           # 边界：t=4..6 不可见，t=7 可见


# ---------------------------------------------------------------- ratio = 4：CSA（压缩 + indexer 选 top-k）
def test_ratio4_indexer_selects_topk_visible_blocks():
    """原理：CSA 不是看所有压缩块，而是由 Indexer 给每个 query 给每个压缩块打分，只选 top-k（这里 k=3）。
    对 query t，可见块数 = (t+1)//4（块必须完整结束）。因此被选中的块数应恰好是 min(index_topk, (t+1)//4)：
      t < 3 时没有任何完整块 → 选 0 个；块数不足 k 时全选；块数超过 k 时只选 k 个（稀疏性的来源）。
    同时选中的块下标必须都 < (t+1)//4，即不能选到未来的块。"""
    attn = make_attn(4)
    assert attn.indexer is not None and attn.compressor.overlap is True
    T = 32
    torch.manual_seed(1)
    x = torch.randn(1, T, D)
    with torch.no_grad():
        qr = attn.q_norm(attn.wq_a(x))
        cos, sin = rope_cos_sin(T, RD, CFG.compress_rope_theta)
        Tc = T // 4
        sel = attn.indexer(x, qr, cos, sin, Tc)                          # [1, T, Tc] bool
    assert sel.shape == (1, T, Tc)
    visible = (torch.arange(T) + 1) // 4                                 # [T] 每个 query 的可见块数
    assert torch.equal(sel[0].sum(-1), visible.clamp(max=CFG.index_topk))
    block = torch.arange(Tc)[None, :]
    assert not (sel[0] & (block >= visible[:, None])).any()              # 不会选到未来块
    assert sel[0, -1].sum() < visible[-1]                                # 最后一个 token：8 个可见块只选了 3 个


# ---------------------------------------------------------------- Compressor：压缩本身对不对
@torch.no_grad()
def reference_block(comp: Compressor, x: torch.Tensor, c: int) -> torch.Tensor:
    """按定义手算第 c 个压缩块：窗口内按 softmax(gate+ape) 对 wkv 做加权池化 → norm → 在位置 c*r 做 RoPE。"""
    r, d = comp.ratio, comp.head_dim
    cur = slice(c * r, (c + 1) * r)
    kv = comp.wkv(x[:, cur])                                             # [1, r, coff*hd]
    sc = comp.wgate(x[:, cur]) + comp.ape
    if comp.overlap:                                                     # 只取后半维作为"当前窗口"
        kv, sc = kv[..., d:], sc[..., d:]
        if c > 0:                                                        # 再拼上"上一窗口"的前半维
            prev = slice((c - 1) * r, c * r)
            kv = torch.cat([comp.wkv(x[:, prev])[..., :d], kv], dim=1)   # [1, 2r, hd]
            sc = torch.cat([(comp.wgate(x[:, prev]) + comp.ape)[..., :d], sc], dim=1)
    pooled = comp.norm((kv * sc.softmax(dim=1)).sum(dim=1))              # [1, hd]
    cos, sin = rope_cos_sin(x.size(1), RD, CFG.compress_rope_theta)
    p = c * r
    return rope_tail(pooled[:, None, None, :], cos[p:p + 1], sin[p:p + 1], RD)[:, 0, 0]


@torch.no_grad()
def check_compressor(ratio: int, T: int):
    torch.manual_seed(0)
    comp = Compressor(CFG, ratio, HD).eval()
    torch.manual_seed(1)
    x = torch.randn(1, T, D)
    cos, sin = rope_cos_sin(T, RD, CFG.compress_rope_theta)
    out = comp(x, cos, sin)
    assert out.shape == (1, T // ratio, HD)
    for c in range(T // ratio):
        torch.testing.assert_close(out[:, c], reference_block(comp, x, c), atol=1e-5, rtol=1e-5)


def test_compressor_ratio8_matches_manual():
    """原理：ratio=8 无重叠。块 c = norm(Σ_i softmax_i(gate_i+ape_i) · kv_i)，再对压缩块施加 RoPE，
    位置取块首 token 的位置 8c。用手写公式逐块比对（块 0 的 RoPE 位置为 0，是恒等旋转）。"""
    check_compressor(8, 16)


def test_compressor_ratio4_overlap_matches_manual():
    """原理：ratio=4 用重叠窗口。wkv/wgate 输出 2*hd 维：后半维属于"当前窗口"，前半维属于"下一个窗口"。
    所以块 c 的池化对象是 8 项：上一窗口(c-1)的 4 个 token 的前半维 + 当前窗口 c 的 4 个 token 的后半维，
    在这 8 项上统一做 softmax。块 0 没有上一窗口（对应分数 -inf），只剩当前窗口的 4 项。"""
    check_compressor(4, 16)


def test_compressor_drops_incomplete_tail_and_short_input():
    """原理：压缩块必须凑满 r 个 token。T=30、r=4 时只产出 7 个块，末尾 2 个 token 暂不压缩
    （增量解码时它们保存在 Compressor 的 kv_state/score_state 中，凑满再产出）；T<r 时没有任何块，返回 None。"""
    torch.manual_seed(0)
    comp = Compressor(CFG, 4, HD).eval()
    with torch.no_grad():
        cos, sin = rope_cos_sin(30, RD, CFG.compress_rope_theta)
        assert comp(torch.randn(1, 30, D), cos, sin).shape == (1, 7, HD)
        assert comp(torch.randn(1, 3, D), cos[:3], sin[:3]) is None


# ---------------------------------------------------------------- 三种 ratio 共有的性质
def test_all_ratios_are_causal():
    """原理：无论哪种 ratio，改变最后一个 token 都不能影响之前位置的输出（含压缩块、indexer 的选择）。"""
    T = 32
    for ratio in (0, 4, 8):
        attn = make_attn(ratio)
        torch.manual_seed(1)
        x = torch.randn(2, T, D)
        x2 = x.clone()
        x2[:, -1] = torch.randn(2, D)
        with torch.no_grad():
            diff = (attn(x2)[:, :-1] - attn(x)[:, :-1]).abs().max().item()
        assert diff < 1e-6, f"ratio={ratio} 泄漏了未来信息: {diff}"


def test_short_sequence_below_ratio_still_works():
    """原理：序列比压缩比还短时没有压缩块，CSA/HCA 层应自动退化成只看滑窗，不能报错或产生 NaN。"""
    for ratio in (0, 4, 8):
        attn = make_attn(ratio)
        with torch.no_grad():
            out = attn(torch.randn(1, 3, D))
        assert out.shape == (1, 3, D) and torch.isfinite(out).all()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
