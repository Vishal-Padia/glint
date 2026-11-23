import torch

import torch.nn.functional as F

def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Per-row reverse KL divergence KL(student || teacher), float64 reference.

    Args:
        student_logits: [N, V] raw logits from the student.
        teacher_logits: [N, V] raw logits from the teacher.

    Returns:
        [N] tensor, one KL value per token position (float64).
    """
    # convert student_logits and teacher_logits to float64
    s = student_logits.to(torch.float64)
    t = teacher_logits.to(torch.float64)

    # logits to log probabilities
    log_p_s = F.log_softmax(s, dim=-1)
    log_p_t = F.log_softmax(t, dim=-1)

    # student probabilities from it's log probs
    p_s = log_p_s.exp()

    # sum over the vocab dim
    return (p_s * (log_p_s - log_p_t)).sum(dim=-1)


if __name__ == "__main__":
    import math

    # 1. Hand-computed 3-token example: softmax(log p) == p, so log-probs work as logits.
    # Build in float64: a float32 input is already rounded before reverse_kl upcasts it.
    p = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64).log()
    q = torch.tensor([[0.5, 0.4, 0.1]], dtype=torch.float64).log()
    expected = 0.7 * math.log(0.7 / 0.5) + 0.2 * math.log(0.2 / 0.4)
    got = reverse_kl(p, q).item()
    assert abs(got - expected) < 1e-12, (got, expected)
    print(f"3-token example: {got:.6f} (expected {expected:.6f})")

    # 2. Identical distributions -> exactly 0.
    x = torch.randn(4, 1000)
    assert torch.all(reverse_kl(x, x) == 0), reverse_kl(x, x)
    print("same input: 0")

    # 3. Random logits -> never negative. Scale 10 makes some probabilities tiny.
    torch.manual_seed(0)
    k = reverse_kl(10 * torch.randn(64, 1000), 10 * torch.randn(64, 1000))
    assert torch.all(k >= 0) and torch.all(torch.isfinite(k)), k
    print(f"random logits: min {k.min():.4f}, max {k.max():.4f}, all >= 0 and finite")

    # 4. Not symmetric: KL(p||q) != KL(q||p).
    print(f"asymmetry: KL(p||q)={reverse_kl(p, q).item():.4f}, KL(q||p)={reverse_kl(q, p).item():.4f}")
