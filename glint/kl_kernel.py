import torch
import triton

import triton.language as tl

@triton.jit
def kl_fwd_kernel(
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

def reverse_kl_chunked(h_s, w_s, h_t, w_t, V=151_665, chunk_rows=1024, BLOCK_V=4096):
    """
    h_s [N, d_s],
    w_s [V_s, d_s],
    h_t [N, d_t],
    w_t [V_t, d_t]  ->  kl [N] fp32
    """
    assert h_s.shape[0] == h_t.shape[0], "student/teacher row counts differ"
    assert V <= w_s.shape[0] and V <= w_t.shape[0], "V exceeds a padded vocab"
    assert h_s.shape[1] == w_s.shape[1]
    assert h_t.shape[1] == w_t.shape[1]
    assert h_s.is_cuda and h_t.is_cuda

    N = h_s.shape[0]
    kl = torch.empty(N, dtype=torch.float32, device=h_s.device)
    lse_s = torch.empty_like(kl)
    lse_t = torch.empty_like(kl)

    for i in range(0, N, chunk_rows):
        j = min(i + chunk_rows, N)
        n = j - i
        s = h_s[i:j] @ w_s.T
        t = h_t[i:j] @ w_t.T

        # launch kernel
        kl_fwd_kernel[(n,)](
            s, t, kl[i:j], lse_s[i:j], lse_t[i:j],
            s.stride(0), t.stride(0),
            V, BLOCK_V=BLOCK_V
        )
        del s, t

    return kl, lse_s, lse_t

@triton.jit
def kl_bwd_kernel(
    s_ptr, t_ptr, # [n, V_s_pad], [n, V_t_pad] logits (recomputed)
    lse_s_ptr, lse_t_ptr, kl_ptr, g_ptr, # [n] fp32 each; g = dL/dK
    grad_s_ptr, # [n, V_s_pad] output, same dtype as s
    stride_s, stride_t, stride_gs, # row strides
    V,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    s_row = s_ptr + row * stride_s
    t_row = t_ptr + row * stride_t
    grad_s_row = grad_s_ptr + row  * stride_gs

    lse_s = tl.load(row + lse_s_ptr)
    lse_t = tl.load(row + lse_t_ptr)
    kl = tl.load(kl_ptr + row)
    g = tl.load(row + g_ptr)

    cols = tl.arange(0, BLOCK_V)
    for start in range(0, V, BLOCK_V):
        offs = start + cols
        mask = offs < V

        s = tl.load(s_row + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(t_row + offs, mask=mask, other=0.0).to(tl.float32)

        logp = s - lse_s
        logq = t - lse_t
        p = tl.exp(logp)
        ell = logp - logq
        grad = g * p * (ell - kl)

        tl.store(
            grad_s_row + offs,
            grad.to(grad_s_ptr.dtype.element_ty),
            mask=mask
        )

def reverse_kl_bwd_chunked(h_s, w_s, h_t, w_t, lse_s, lse_t, kl, g,
                           V=151_665, chunk_rows=1024, BLOCK_V=4096):
    N = h_s.shape[0]
    grad_h_s = torch.empty_like(h_s)
    grad_w_s = torch.zeros_like(w_s, dtype=torch.float32) # accumulate in fp32

    for i in range(0, N, chunk_rows):
        j = min(i + chunk_rows, N)
        n = j - i
        s = h_s[i:j] @ w_s.T
        t = h_t[i:j] @ w_t.T
        grad_s = torch.zeros_like(s) # zeros: padding columns must be 0

        kl_bwd_kernel[(n,)](
            s, t, lse_s[i:j], lse_t[i:j], kl[i:j], g[i:j], grad_s,
            s.stride(0), t.stride(0), grad_s.stride(0),
            V, BLOCK_V=BLOCK_V,
        )

        grad_h_s[i:j] = grad_s @ w_s
        grad_w_s = torch.addmm(grad_w_s, grad_s.T, h_s[i:j], out=grad_w_s, out_dtype=torch.float32)
        del s, t, grad_s

    return grad_h_s, grad_w_s.to(w_s.dtype)

class ReverseKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_s, w_s, h_t, w_t, V, chunk_rows):
        kl, lse_s, lse_t = reverse_kl_chunked(
            h_s, w_s, h_t, w_t, V=V, chunk_rows=chunk_rows
        ) # run forward wrapper
        ctx.save_for_backward(kl, lse_s, lse_t, h_s, w_s, h_t, w_t) # whatever backward needs
        ctx.V, ctx.chunk_rows = V, chunk_rows # non-tensors go on ctx directly
        return kl

    @staticmethod
    def backward(ctx, g): # g = dL/dK, shape [N]
        g = g.contiguous()
        kl, lse_s, lse_t, h_s, w_s, h_t, w_t = ctx.saved_tensors
        grad_h_s, grad_w_s = reverse_kl_bwd_chunked(
            h_s, w_s,
            h_t, w_t,
            lse_s, lse_t, kl,
            g,
            V=ctx.V, chunk_rows=ctx.chunk_rows
        )
        return grad_h_s, grad_w_s, None, None, None, None # one per forward input

def reverse_kl_fused(h_s, w_s, h_t, w_t, V=151_665, chunk_rows=1024):
    return ReverseKL.apply(h_s, w_s, h_t, w_t, V, chunk_rows)
