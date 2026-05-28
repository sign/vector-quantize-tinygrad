#!/usr/bin/env python3
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_CUDA_BENCH = os.getenv("VQTG_CUDA_BENCH") == "1"
os.environ.setdefault("DEV", "CUDA" if RUN_CUDA_BENCH else "PYTHON")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from tinygrad import Tensor  # noqa: E402
from vector_quantize_pytorch import LFQ as TorchLFQ  # noqa: E402
from vector_quantize_tinygrad import LFQ as TinyLFQ  # noqa: E402


# Tinygrad may simplify some gradients to device-less constants; adding x*0
# pins them back to the same device so .numpy() does not hit the CPU compiler.
def tiny_grad(loss, x):
    (grad,) = loss.gradient(x, gradient=Tensor(1.0, device=x.device))
    return (grad + x * 0.0).numpy()


# Scalars and tensors both become numpy arrays for one comparison path.
def tiny_numpy(x):
    if x.device is None:
        x = x + Tensor.zeros(*x.shape) if x.shape else x + Tensor(0.0, device="PYTHON")
    return x.numpy()


# Torch tensors need detaching before numpy conversion during gradient tests.
def torch_numpy(x):
    return x.detach().cpu().numpy()


# Tinygrad and PyTorch Linear store weights as (out,in), so copying is direct.
def copy_linear(tiny_linear, torch_linear):
    tiny_linear.weight = Tensor(torch_numpy(torch_linear.weight).astype(np.float32))
    tiny_linear.bias = Tensor(torch_numpy(torch_linear.bias).astype(np.float32)) if torch_linear.bias is not None else None


# Compares the no-projection path, including entropy/commit losses and input
# gradients through the straight-through estimator. Shape: B,N,D -> B,N,D.
def test_basic_outputs_and_grads_match():
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=(2, 5, 4)).astype(np.float32)

    torch_lfq = TorchLFQ(dim=4, codebook_size=16, entropy_loss_weight=0.1, commitment_loss_weight=0.25)
    tiny_lfq = TinyLFQ(dim=4, codebook_size=16, entropy_loss_weight=0.1, commitment_loss_weight=0.25)

    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tiny = Tensor(x_np)

    torch_lfq.train()
    with Tensor.train():
        torch_ret, torch_breakdown = torch_lfq(x_torch, inv_temperature=5.0, return_loss_breakdown=True)
        tiny_ret, tiny_breakdown = tiny_lfq(x_tiny, inv_temperature=5.0, return_loss_breakdown=True)
        tiny_loss = tiny_ret.quantized.square().mean() + tiny_ret.entropy_aux_loss

    torch_loss = torch_ret.quantized.square().mean() + torch_ret.entropy_aux_loss
    torch_loss.backward()

    np.testing.assert_array_equal(tiny_numpy(tiny_ret.indices), torch_numpy(torch_ret.indices))
    np.testing.assert_allclose(tiny_numpy(tiny_ret.quantized), torch_numpy(torch_ret.quantized), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tiny_numpy(tiny_breakdown.per_sample_entropy), torch_numpy(torch_breakdown.per_sample_entropy), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_numpy(tiny_breakdown.batch_entropy), torch_numpy(torch_breakdown.batch_entropy), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_numpy(tiny_breakdown.commitment), torch_numpy(torch_breakdown.commitment), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tiny_grad(tiny_loss, x_tiny), torch_numpy(x_torch.grad), rtol=2e-4, atol=2e-4)


# Compares the projected path by copying PyTorch Linear weights into tinygrad.
# Shape: B,N,6 -> project to B,N,3 -> quantize -> project back to B,N,6.
def test_projected_outputs_and_grads_match():
    rng = np.random.default_rng(1)
    x_np = rng.normal(size=(2, 4, 6)).astype(np.float32)

    torch_lfq = TorchLFQ(dim=6, codebook_size=8, entropy_loss_weight=0.05, commitment_loss_weight=0.1, has_projections=True)
    tiny_lfq = TinyLFQ(dim=6, codebook_size=8, entropy_loss_weight=0.05, commitment_loss_weight=0.1, has_projections=True)
    copy_linear(tiny_lfq.project_in, torch_lfq.project_in)
    copy_linear(tiny_lfq.project_out, torch_lfq.project_out)

    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tiny = Tensor(x_np)

    torch_lfq.train()
    with Tensor.train():
        torch_ret = torch_lfq(x_torch, inv_temperature=4.0)
        tiny_ret = tiny_lfq(x_tiny, inv_temperature=4.0)
        tiny_loss = tiny_ret.quantized.square().mean() + tiny_ret.entropy_aux_loss

    torch_loss = torch_ret.quantized.square().mean() + torch_ret.entropy_aux_loss
    torch_loss.backward()

    np.testing.assert_array_equal(tiny_numpy(tiny_ret.indices), torch_numpy(torch_ret.indices))
    np.testing.assert_allclose(tiny_numpy(tiny_ret.quantized), torch_numpy(torch_ret.quantized), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_numpy(tiny_ret.entropy_aux_loss), torch_numpy(torch_ret.entropy_aux_loss), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_grad(tiny_loss, x_tiny), torch_numpy(x_torch.grad), rtol=3e-4, atol=3e-4)


# Verifies channel-first spatial packing and multi-codebook index restoration.
# Shape: B,D,H,W -> quantized B,D,H,W and indices B,H,W,C.
def test_channel_first_multi_codebook_indices_to_codes_match():
    rng = np.random.default_rng(2)
    x_np = rng.normal(size=(1, 6, 2, 3)).astype(np.float32)

    torch_lfq = TorchLFQ(dim=6, codebook_size=8, num_codebooks=2)
    tiny_lfq = TinyLFQ(dim=6, codebook_size=8, num_codebooks=2)

    x_torch = torch.tensor(x_np)
    x_tiny = Tensor(x_np)

    torch_lfq.eval()
    torch_ret = torch_lfq(x_torch)
    tiny_ret = tiny_lfq(x_tiny)

    np.testing.assert_array_equal(tiny_numpy(tiny_ret.indices), torch_numpy(torch_ret.indices))
    np.testing.assert_allclose(tiny_numpy(tiny_ret.quantized), torch_numpy(torch_ret.quantized), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tiny_numpy(tiny_lfq.indices_to_codes(tiny_ret.indices)), torch_numpy(torch_lfq.indices_to_codes(torch_ret.indices)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tiny_numpy(tiny_lfq.indices_to_codes(tiny_ret.indices)), tiny_numpy(tiny_ret.quantized), rtol=1e-6, atol=1e-6)


# Compares masked batches because padding/masking is common in token streams.
# Shape: B,N,D plus mask B -> masked entropy/commit losses and B,N,D grads.
def test_masked_losses_and_grads_match():
    rng = np.random.default_rng(3)
    x_np = rng.normal(size=(3, 4, 4)).astype(np.float32)
    mask_np = np.array([True, False, True])

    torch_lfq = TorchLFQ(dim=4, codebook_size=16, entropy_loss_weight=0.1, commitment_loss_weight=0.2)
    tiny_lfq = TinyLFQ(dim=4, codebook_size=16, entropy_loss_weight=0.1, commitment_loss_weight=0.2)

    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tiny = Tensor(x_np)

    torch_lfq.train()
    with Tensor.train():
        torch_ret, torch_breakdown = torch_lfq(x_torch, inv_temperature=3.0, return_loss_breakdown=True, mask=torch.tensor(mask_np))
        tiny_ret, tiny_breakdown = tiny_lfq(x_tiny, inv_temperature=3.0, return_loss_breakdown=True, mask=Tensor(mask_np))
        tiny_loss = tiny_ret.quantized.square().mean() + tiny_ret.entropy_aux_loss

    torch_loss = torch_ret.quantized.square().mean() + torch_ret.entropy_aux_loss
    torch_loss.backward()

    np.testing.assert_array_equal(tiny_numpy(tiny_ret.indices), torch_numpy(torch_ret.indices))
    np.testing.assert_allclose(tiny_numpy(tiny_breakdown.per_sample_entropy), torch_numpy(torch_breakdown.per_sample_entropy), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_numpy(tiny_breakdown.batch_entropy), torch_numpy(torch_breakdown.batch_entropy), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_numpy(tiny_breakdown.commitment), torch_numpy(torch_breakdown.commitment), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tiny_grad(tiny_loss, x_tiny), torch_numpy(x_torch.grad), rtol=3e-4, atol=3e-4)


# Covers the common BSQ-ish option and input soft clamping together.
# Shape stays B,N,D, but quantized components become normalized code values.
def test_spherical_soft_clamp_outputs_and_grads_match():
    rng = np.random.default_rng(4)
    x_np = (rng.normal(size=(2, 5, 4)) * 2.0).astype(np.float32)

    torch_lfq = TorchLFQ(
        dim=4,
        codebook_size=16,
        entropy_loss_weight=0.05,
        commitment_loss_weight=0.15,
        soft_clamp_input_value=1.25,
        spherical=True,
    )
    tiny_lfq = TinyLFQ(
        dim=4,
        codebook_size=16,
        entropy_loss_weight=0.05,
        commitment_loss_weight=0.15,
        soft_clamp_input_value=1.25,
        spherical=True,
    )

    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tiny = Tensor(x_np)

    torch_lfq.train()
    with Tensor.train():
        torch_ret = torch_lfq(x_torch, inv_temperature=2.5)
        tiny_ret = tiny_lfq(x_tiny, inv_temperature=2.5)
        tiny_loss = tiny_ret.quantized.square().mean() + tiny_ret.entropy_aux_loss

    torch_loss = torch_ret.quantized.square().mean() + torch_ret.entropy_aux_loss
    torch_loss.backward()

    np.testing.assert_array_equal(tiny_numpy(tiny_ret.indices), torch_numpy(torch_ret.indices))
    np.testing.assert_allclose(tiny_numpy(tiny_ret.quantized), torch_numpy(torch_ret.quantized), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_numpy(tiny_ret.entropy_aux_loss), torch_numpy(torch_ret.entropy_aux_loss), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tiny_grad(tiny_loss, x_tiny), torch_numpy(x_torch.grad), rtol=4e-4, atol=4e-4)


# These configs exercise large soft-assignment matrices without changing the
# public benchmark defaults: 128 inputs, bsz=16, and about 10 GiB VRAM.
def cuda_benchmark_configs():
    configs = [
        {"name": "single_codebook_1024", "dim": 10, "codebook_size": 1024, "num_codebooks": 1, "has_projections": False},
        {"name": "two_codebooks_512", "dim": 18, "codebook_size": 512, "num_codebooks": 2, "has_projections": False},
        {"name": "projected_2048", "dim": 64, "codebook_size": 2048, "num_codebooks": 1, "has_projections": True},
    ]
    if os.getenv("VQTG_CUDA_CONFIG_INDEX") is not None:
        return [configs[int(os.getenv("VQTG_CUDA_CONFIG_INDEX"))]]
    return configs[: int(os.getenv("VQTG_CUDA_CONFIG_LIMIT", str(len(configs))))]


# Token count is chosen from the dominant entropy logits/probs tensor.
# B,N,C,K with a multiplier approximates forward+backward saved activations.
def cuda_benchmark_tokens(cfg, batch_size):
    target_gb = float(os.getenv("VQTG_CUDA_TARGET_GB", "10"))
    activation_multiplier = float(os.getenv("VQTG_CUDA_ACTIVATION_MULTIPLIER", "8"))
    target_bytes = target_gb * (1024**3)
    tokens = int(target_bytes / (batch_size * cfg["num_codebooks"] * cfg["codebook_size"] * 4 * activation_multiplier))
    return max(64, tokens), target_gb


# Clean cached allocators between implementations/configurations so a 10 GiB
# run does not accidentally become cumulative across benchmark cases.
def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from tinygrad.device import Device

        Device["CUDA"].synchronize()
        Device["CUDA"].allocator.free_cache()
    except Exception:
        pass


# The benchmark should time real parameter gradients when projections exist,
# not only the input gradient.
def tiny_lfq_parameters(model):
    params = []
    for layer in (model.project_in, model.project_out):
        if layer is None or not hasattr(layer, "weight"):
            continue
        params.append(layer.weight)
        if getattr(layer, "bias", None) is not None:
            params.append(layer.bias)
    return params


# Same parameter ordering as tiny_lfq_parameters for sampled grad comparison.
def torch_lfq_parameters(model):
    params = []
    for layer in (model.project_in, model.project_out):
        if layer is None or not hasattr(layer, "weight"):
            continue
        params.append(layer.weight)
        if getattr(layer, "bias", None) is not None:
            params.append(layer.bias)
    return params


# Builds matching PyTorch/tinygrad LFQ modules; projected configs copy weights
# so output and gradient comparisons remain meaningful.
def make_benchmark_pair(cfg):
    kwargs = dict(
        dim=cfg["dim"],
        codebook_size=cfg["codebook_size"],
        num_codebooks=cfg["num_codebooks"],
        has_projections=cfg["has_projections"],
        entropy_loss_weight=0.02,
        commitment_loss_weight=0.05,
    )
    torch_lfq = TorchLFQ(**kwargs)
    tiny_lfq = TinyLFQ(**kwargs)
    if cfg["has_projections"]:
        copy_linear(tiny_lfq.project_in, torch_lfq.project_in)
        copy_linear(tiny_lfq.project_out, torch_lfq.project_out)
    return torch_lfq.cuda().train(), tiny_lfq


# Runs PyTorch eagerly on CUDA, timing forward and backward separately.
# Samples are small slices used later to compare tinygrad correctness.
def run_torch_cuda_batches(model, batches, inv_temperature=2.0):
    torch.cuda.reset_peak_memory_stats()
    forward_s = backward_s = 0.0
    samples = []
    params = torch_lfq_parameters(model)
    for x_np in batches:
        model.zero_grad(set_to_none=True)
        x = torch.tensor(x_np, device="cuda", requires_grad=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        ret = model(x, inv_temperature=inv_temperature)
        loss = ret.quantized.square().mean() + ret.entropy_aux_loss
        torch.cuda.synchronize()
        forward_s += time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_s += time.perf_counter() - start

        samples.append(
            {
                "loss": float(loss.detach().cpu()),
                "quantized": torch_numpy(ret.quantized[:, :8, :8]),
                "indices": torch_numpy(ret.indices[:, :8, ...]),
                "grad": torch_numpy(x.grad[:, :8, :8]),
                "param_grads": [torch_numpy(p.grad.flatten()[:16]) for p in params if p.grad is not None],
            }
        )
        del x, ret, loss
    return {"forward_s": forward_s, "backward_s": backward_s, "peak_gb": torch.cuda.max_memory_allocated() / (1024**3)}, samples


# Runs tinygrad on CUDA and compares sampled outputs/gradients against PyTorch.
# Backward uses a fresh unrealized graph; realizing a tinygrad loss severs grads.
def run_tiny_cuda_batches(model, batches, torch_samples, inv_temperature=2.0):
    from tinygrad.device import Device
    from tinygrad.helpers import GlobalCounters

    forward_s = backward_s = 0.0
    peak_bytes = 0
    params = tiny_lfq_parameters(model)
    for i, x_np in enumerate(batches):
        x_fwd = Tensor(x_np, device="CUDA")
        with Tensor.train():
            Device["CUDA"].synchronize()
            start = time.perf_counter()
            ret_fwd = model(x_fwd, inv_temperature=inv_temperature)
            loss_fwd = ret_fwd.quantized.square().mean() + ret_fwd.entropy_aux_loss
            loss_fwd.realize(ret_fwd.quantized, ret_fwd.indices)
            Device["CUDA"].synchronize()
            forward_s += time.perf_counter() - start

        tiny_loss = float(loss_fwd.numpy())
        tiny_quantized = ret_fwd.quantized[:, :8, :8].numpy()
        tiny_indices = ret_fwd.indices[:, :8, ...].numpy()
        del x_fwd, ret_fwd, loss_fwd

        x = Tensor(x_np, device="CUDA")
        with Tensor.train():
            ret_bwd = model(x, inv_temperature=inv_temperature)
            loss_bwd = ret_bwd.quantized.square().mean() + ret_bwd.entropy_aux_loss
            start = time.perf_counter()
            targets = [x, *params]
            grads = loss_bwd.gradient(*targets, gradient=Tensor(1.0, device="CUDA"))
            grads = [(grad + target * 0.0) for grad, target in zip(grads, targets)]
            grad_checksum = sum(grad.square().sum() for grad in grads)
            grad_checksum.realize(*grads)
            Device["CUDA"].synchronize()
            backward_s += time.perf_counter() - start
            grad = grads[0]

        peak_bytes = max(peak_bytes, GlobalCounters.mem_used_per_device["CUDA"])
        sample = torch_samples[i]
        np.testing.assert_allclose(tiny_loss, sample["loss"], rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(tiny_quantized, sample["quantized"], rtol=1e-2, atol=1e-2)
        np.testing.assert_array_equal(tiny_indices, sample["indices"])
        np.testing.assert_allclose(grad[:, :8, :8].numpy(), sample["grad"], rtol=2e-2, atol=2e-2)
        for tiny_param_grad, torch_param_grad in zip(grads[1:], sample["param_grads"]):
            np.testing.assert_allclose(tiny_param_grad.flatten()[:16].numpy(), torch_param_grad, rtol=3e-2, atol=3e-2)
        del x, ret_bwd, loss_bwd, grad, grads, grad_checksum
    return {"forward_s": forward_s, "backward_s": backward_s, "tracked_gb": peak_bytes / (1024**3)}


# Opt-in heavy benchmark: VQTG_CUDA_BENCH=1 video-tokenizer/.venv/bin/python ...
# Defaults process 128 dummy inputs with bsz=16 and ~10 GiB fwd/bwd activations.
def test_cuda_heavy_timing_benchmark():
    if not RUN_CUDA_BENCH:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("VQTG_CUDA_BENCH=1 requires torch CUDA")
    if os.getenv("VQTG_CUDA_CONFIG_INDEX") is None:
        for idx in range(len(cuda_benchmark_configs())):
            env = os.environ.copy()
            env["VQTG_CUDA_CONFIG_INDEX"] = str(idx)
            subprocess.run([sys.executable, __file__], check=True, env=env)
        return

    batch_size = 16
    total_inputs = int(os.getenv("VQTG_CUDA_INPUTS", "128"))
    if total_inputs % batch_size:
        raise ValueError(f"VQTG_CUDA_INPUTS must be divisible by {batch_size}")

    rows = []
    for cfg_idx, cfg in enumerate(cuda_benchmark_configs()):
        tokens, target_gb = cuda_benchmark_tokens(cfg, batch_size)
        approx_gb = batch_size * tokens * cfg["num_codebooks"] * cfg["codebook_size"] * 4 * float(os.getenv("VQTG_CUDA_ACTIVATION_MULTIPLIER", "8")) / (1024**3)
        rng = np.random.default_rng(10_000 + cfg_idx)
        warmup = rng.normal(size=(batch_size, tokens, cfg["dim"])).astype(np.float32)
        batches = [rng.normal(size=(batch_size, tokens, cfg["dim"])).astype(np.float32) for _ in range(total_inputs // batch_size)]

        cleanup_cuda()
        torch_lfq, tiny_lfq = make_benchmark_pair(cfg)
        _torch_warmup, warmup_samples = run_torch_cuda_batches(torch_lfq, [warmup])
        torch_stats, torch_samples = run_torch_cuda_batches(torch_lfq, batches)
        del torch_lfq
        cleanup_cuda()

        _tiny_warmup = run_tiny_cuda_batches(tiny_lfq, [warmup], warmup_samples)
        tiny_stats = run_tiny_cuda_batches(tiny_lfq, batches, torch_samples)
        del tiny_lfq, warmup, batches, warmup_samples, torch_samples
        cleanup_cuda()

        row = {
            "name": cfg["name"],
            "tokens": tokens,
            "approx_gb": approx_gb,
            "torch_forward_s": torch_stats["forward_s"],
            "torch_backward_s": torch_stats["backward_s"],
            "tiny_forward_s": tiny_stats["forward_s"],
            "tiny_backward_s": tiny_stats["backward_s"],
            "torch_peak_gb": torch_stats["peak_gb"],
            "tiny_tracked_gb": tiny_stats["tracked_gb"],
        }
        rows.append(row)

        fwd_ratio = row["tiny_forward_s"] / max(row["torch_forward_s"], 1e-9)
        bwd_ratio = row["tiny_backward_s"] / max(row["torch_backward_s"], 1e-9)
        print(
            f"CUDA LFQ {row['name']}: N={tokens}, target~{target_gb:.2f}GiB, approx={approx_gb:.2f}GiB | "
            f"torch fwd/bwd={row['torch_forward_s']:.3f}/{row['torch_backward_s']:.3f}s "
            f"tinygrad fwd/bwd={row['tiny_forward_s']:.3f}/{row['tiny_backward_s']:.3f}s "
            f"ratios={fwd_ratio:.2f}x/{bwd_ratio:.2f}x "
            f"torch_peak={row['torch_peak_gb']:.2f}GiB tiny_tracked={row['tiny_tracked_gb']:.2f}GiB"
        )

        max_slowdown = float(os.getenv("VQTG_CUDA_MAX_SLOWDOWN", "0"))
        if max_slowdown > 0:
            assert fwd_ratio <= max_slowdown and bwd_ratio <= max_slowdown

    assert rows and all(r["torch_forward_s"] > 0 and r["tiny_forward_s"] > 0 and r["torch_backward_s"] > 0 and r["tiny_backward_s"] > 0 for r in rows)


# Running the file directly is the lowest-friction path for local checks;
# pytest can still call the test_* functions if configured for it.
def main():
    test_basic_outputs_and_grads_match()
    test_projected_outputs_and_grads_match()
    test_channel_first_multi_codebook_indices_to_codes_match()
    test_masked_losses_and_grads_match()
    test_spherical_soft_clamp_outputs_and_grads_match()
    test_cuda_heavy_timing_benchmark()
    print("all vector-quantize-tinygrad tests passed")


if __name__ == "__main__":
    main()
