import argparse
import itertools
import random
import math
import statistics
import sys
import time
from pathlib import Path

import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import pattern_freq_cuda  # type: ignore

    return pattern_freq_cuda


def _pad_2d(tensor: torch.Tensor, target: int) -> torch.Tensor:
    cur = tensor.shape[0]
    if cur == target:
        return tensor
    if cur > target:
        return tensor[:target]
    pad = torch.zeros(target - cur, tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def _build_blocks(n_species: int, species_for_model: int, n_blocks: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    blocks = []
    for _ in range(n_blocks):
        block = sorted(rng.sample(range(n_species), species_for_model))
        blocks.append(block)
    return blocks


def _run_loop_impl(seq_cuda, blocks, target_len, mod, device):
    pattern_list = []
    for block in blocks:
        block_sorted = sorted(block)
        qs = list(itertools.combinations(block_sorted, 4))
        idx_cuda = torch.tensor(qs, dtype=torch.long, device=device)
        pattern_tensor = mod.compute_pattern_frequencies_cuda(seq_cuda, idx_cuda)
        pattern_tensor = _pad_2d(pattern_tensor, target_len)
        pattern_list.append(pattern_tensor)
    return torch.stack(pattern_list, dim=0)


def _run_grouped_uniform_impl(seq_cuda, blocks, local_quartets_cpu, target_len, mod, device):
    blocks_cpu = torch.tensor(blocks, dtype=torch.long)
    quartet_idx_cpu = blocks_cpu[:, local_quartets_cpu]
    quartet_idx_cuda = quartet_idx_cpu.to(device=device, dtype=torch.long)
    out = mod.compute_pattern_frequencies_cuda_grouped_uniform(seq_cuda, quartet_idx_cuda)
    cur = out.shape[1]
    if cur == target_len:
        return out
    if cur > target_len:
        return out[:, :target_len, :]
    pad = torch.zeros(out.shape[0], target_len - cur, out.shape[2], device=out.device, dtype=out.dtype)
    return torch.cat([out, pad], dim=1)


def _time_cuda_run(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-species", type=int, default=120)
    parser.add_argument("--seq-len", type=int, default=800)
    parser.add_argument("--species-for-model", type=int, default=24)
    parser.add_argument("--n-blocks", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("该脚本需要 CUDA 环境。")

    mod = _load_module()
    if not hasattr(mod, "compute_pattern_frequencies_cuda_grouped_uniform"):
        raise RuntimeError("当前 pattern_freq_cuda 不包含 uniform grouped 算子，请先重新编译。")

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    seq = torch.randint(
        low=0,
        high=4,
        size=(args.n_species, args.seq_len),
        dtype=torch.int8,
        generator=generator,
    )
    seq_cuda = seq.to(device)

    blocks = _build_blocks(args.n_species, args.species_for_model, args.n_blocks, args.seed)
    quartets_per_block = math.comb(args.species_for_model, 4)
    target_len = quartets_per_block
    local_quartets_cpu = torch.tensor(
        list(itertools.combinations(range(args.species_for_model), 4)),
        dtype=torch.long,
    )

    for _ in range(args.warmup):
        _run_loop_impl(seq_cuda, blocks, target_len, mod, device)
        _run_grouped_uniform_impl(seq_cuda, blocks, local_quartets_cpu, target_len, mod, device)

    loop_times = []
    grouped_times = []
    max_diffs = []

    for _ in range(args.repeat):
        out_loop, t_loop = _time_cuda_run(
            lambda: _run_loop_impl(seq_cuda, blocks, target_len, mod, device)
        )
        out_grouped, t_grouped = _time_cuda_run(
            lambda: _run_grouped_uniform_impl(seq_cuda, blocks, local_quartets_cpu, target_len, mod, device)
        )
        diff = (out_loop - out_grouped).abs().max().item()
        max_diffs.append(diff)
        loop_times.append(t_loop)
        grouped_times.append(t_grouped)

    loop_mean = statistics.mean(loop_times)
    grouped_mean = statistics.mean(grouped_times)
    speedup = loop_mean / grouped_mean if grouped_mean > 0 else float("inf")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"n_species={args.n_species}, seq_len={args.seq_len}, species_for_model={args.species_for_model}, n_blocks={args.n_blocks}")
    print(f"quartets_per_block={target_len}, total_quartets={target_len * args.n_blocks}")
    print(f"max_abs_diff={max(max_diffs):.6g}")
    print(f"loop_mean_sec={loop_mean:.6f}")
    print(f"grouped_uniform_mean_sec={grouped_mean:.6f}")
    print(f"speedup_x={speedup:.3f}")


if __name__ == "__main__":
    main()
