import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from glint.kl_reference import reverse_kl
from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT = "def reverse(s):\n    return s[::-1]\n\nThe capital of France is Paris. Distillation trains a student"


def prime_logits(model, input_ids):
    n = input_ids.shape[1]
    position_ids = torch.arange(n, device="cuda").unsqueeze(0)
    return model(input_ids, position_ids, seq_lens=torch.tensor([n], device="cuda"))["logits"]


def report(name, a, b):
    kl = reverse_kl(a[0], b[0])
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(
        f"{name}: max|logit diff|={(a - b).abs().max().item():.4f}  "
        f"mean KL={kl.mean().item():.2e}  max KL={kl.max().item():.2e}  argmax agree={agree:.1%}"
    )
    return kl.mean().item()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def main():
    log(f"loading tokenizer for {NAME}")
    tok = AutoTokenizer.from_pretrained(NAME)
    input_ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    log(f"prompt is {input_ids.shape[1]} tokens")

    log("loading HF reference model (downloads ~1 GB on first run)")
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    hf = hf.cuda().eval()

    log("loading the same weights into prime-rl's patched Qwen2 model")
    config = AutoConfig.from_pretrained(NAME, attn_implementation="flash_attention_2")
    config.use_cache = False
    prime = AutoModelForCausalLMPrimeRL.from_pretrained(NAME, config=config, dtype=torch.bfloat16).cuda().eval()
    inject_prime_lm_head(prime, chunk_size=None)
    log(f"prime-rl resolved the model class to: {type(prime).__name__}")

    log("forward pass on both, comparing logits (KL uses glint's reverse_kl)")
    hf_out = hf(input_ids).logits.float()
    good = report("patched vs HF      ", prime_logits(prime, input_ids).float(), hf_out)

    log("sanity check: zero the q/k/v biases (the attention_bias=None bug) and compare again")
    for layer in prime.model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            proj.bias.zero_()
    bad = report("biases zeroed vs HF", prime_logits(prime, input_ids).float(), hf_out)

    assert good < 1e-2, f"patched model diverges from HF: mean KL {good}"
    assert bad > 100 * good, "zeroing biases barely changed the output; the check is too weak"
    log("REAL WEIGHTS: PASS")


if __name__ == "__main__":
    main()
