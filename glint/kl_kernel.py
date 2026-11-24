import triton
import triton.language as tl

@triton.jit
def kl_kernel(
    s_ptr, t_ptr, # [N, V_padded] student and teacher logits
    kl_ptr, lse_s_ptr, lse_t_ptr, # [N] fp32 outputs
    stride_s, stride_t, # row strides (elements)
    V, # real vocab size (151_665)
    BLOCK_V: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    s_row = s_ptr + row * stride_s # student rows
    t_row = t_ptr + row * stride_t # teacher rows
    cols = tl.arange(0, BLOCK_V)

    NEG_INF = float("-inf")
    m_s = tl.full([], NEG_INF, tl.float32)
    m_t = tl.full([], NEG_INF, tl.float32)
    d_s = tl.zeros([], tl.float32)
    d_t = tl.zeros([], tl.float32)

    # pass 1
    for start in range(0, V, BLOCK_V):
        offs = start + cols
        mask = offs < V
        s = tl.load(s_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
        t = tl.load(t_row + offs, mask=mask, other=NEG_INF).to(tl.float32)

        m_s_new = tl.maximum(m_s, tl.max(s, axis=0))
        d_s = d_s * tl.exp(m_s - m_s_new) + tl.sum(tl.exp(s - m_s_new), axis=0)
        m_s = m_s_new

        m_t_new = tl.maximum(m_t, tl.max(t, axis=0))
        d_t = d_t * tl.exp(m_t - m_t_new) + tl.sum(tl.exp(t - m_t_new), axis=0)
        m_t = m_t_new

    lse_s = m_s + tl.log(d_s)
    lse_t = m_t + tl.log(d_t)
    tl.store(lse_s_ptr + row, lse_s)
    tl.store(lse_t_ptr + row, lse_t)

    # pass 2
    acc = tl.zeros([], tl.float32)
    for start in range(0, V, BLOCK_V):
        offs = start + cols
        mask = offs < V
        s = tl.load(s_row + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(t_row + offs, mask=mask, other=0.0).to(tl.float32)

        p = tl.exp(s - lse_s)
        acc += tl.sum(tl.where(mask, p * (s - t), 0), axis=0)

    kl = acc + (lse_t - lse_s)
    tl.store(kl_ptr + row, kl)
