#!/usr/bin/env python3
"""Benchmark original vs experimental QuartFormer preprocessing chains."""

from __future__ import annotations

import argparse
import importlib
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
CPP_DIR = REPO_ROOT / "cpp_source"
CUDA_DIR = CPP_DIR / "cuda13_pattern_freq"


@dataclass
class LoadResult:
    payload: np.ndarray
    species_names: list[str]
    seq_length: int
    load_seconds: float
    h2d_seconds: float
    total_seconds: float
    host_bytes: int
    h2d_peak_mib: float


@dataclass
class PatternResult:
    seconds: float
    processed_tensors: int
    peak_mib: float


@dataclass
class ChainSpec:
    name: str
    sp_mod: Any
    pf_mod: Any
    kind: str


def _import_extension(module_name: str, parent_dir: Path):
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    return importlib.import_module(module_name)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _cuda_sync()


def _format_mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def _load_extensions(candidate: str) -> tuple[ChainSpec, ChainSpec]:
    original = ChainSpec(
        name="original",
        sp_mod=_import_extension("sequence_processor", CPP_DIR),
        pf_mod=_import_extension("pattern_freq_cuda", CUDA_DIR),
        kind="dense",
    )

    if candidate == "v3":
        experimental = ChainSpec(
            name="v3",
            sp_mod=_import_extension("sequence_processor_v3", CPP_DIR),
            pf_mod=_import_extension("pattern_freq_cuda_v3", CUDA_DIR),
            kind="packed",
        )
    else:
        raise ValueError(f"Unsupported candidate: {candidate}")

    return original, experimental


def _load_payload(spec: ChainSpec, phy_path: Path) -> tuple[np.ndarray, list[str], int]:
    if spec.kind == "packed":
        payload, species_names, seq_length = spec.sp_mod.load_phy_to_packed_tensor(
            str(phy_path), drop_conserved_sites=True
        )
        return payload, list(species_names), int(seq_length)
    if spec.kind == "bitplane":
        payload, species_names, seq_length = spec.sp_mod.load_phy_to_bitplanes(
            str(phy_path), drop_conserved_sites=True
        )
        return payload, list(species_names), int(seq_length)

    payload, species_names = spec.sp_mod.load_phy_to_tensor(str(phy_path), drop_conserved_sites=True)
    return payload, list(species_names), int(payload.shape[1])


def _payload_to_cuda(payload: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(payload).to(device)


def _time_loader(spec: ChainSpec, phy_path: Path, device: torch.device, repeats: int) -> LoadResult:
    best: LoadResult | None = None
    for _ in range(repeats):
        _empty_cuda_cache()
        t0 = time.perf_counter()
        payload, species_names, seq_length = _load_payload(spec, phy_path)
        t1 = time.perf_counter()

        torch.cuda.reset_peak_memory_stats(device)
        seq_cuda = _payload_to_cuda(payload, device)
        _cuda_sync()
        t2 = time.perf_counter()
        peak_mib = _format_mib(torch.cuda.max_memory_allocated(device))
        del seq_cuda
        _empty_cuda_cache()

        result = LoadResult(
            payload=payload,
            species_names=species_names,
            seq_length=seq_length,
            load_seconds=t1 - t0,
            h2d_seconds=t2 - t1,
            total_seconds=t2 - t0,
            host_bytes=int(payload.nbytes),
            h2d_peak_mib=peak_mib,
        )
        if best is None or result.total_seconds < best.total_seconds:
            best = result

    assert best is not None
    return best


def _build_blocks(num_species: int, k_param: float, model_size: int) -> list[list[int]]:
    if num_species == model_size:
        return [list(range(num_species))]
    batching_mod = _import_extension("batching_algorithms", CPP_DIR)
    return batching_mod.pair_balanced_block_design_v5(
        num_species=num_species,
        k_param=k_param,
        batch_size=model_size,
        seed=42,
        species_weight=2.0,
        threads=1,
    )


def _call_pattern_fn(
    spec: ChainSpec,
    seq_cuda: torch.Tensor,
    idx_cuda: torch.Tensor,
    seq_length: int,
) -> torch.Tensor:
    if spec.kind == "packed":
        return spec.pf_mod.compute_pattern_frequencies_cuda_packed(seq_cuda, idx_cuda, seq_length)
    if spec.kind == "bitplane":
        return spec.pf_mod.compute_pattern_frequencies_cuda_bitplanes(seq_cuda, idx_cuda, seq_length)
    return spec.pf_mod.compute_pattern_frequencies_cuda(seq_cuda, idx_cuda)


def _pattern_loop_once(
    spec: ChainSpec,
    payload: np.ndarray,
    seq_length: int,
    blocks: list[list[int]],
    infer_batch_size: int,
    model_size: int,
    max_batches: int | None,
    device: torch.device,
) -> PatternResult:
    local_quartet_template = np.asarray(
        list(itertools.combinations(range(model_size), 4)), dtype=np.int32
    )

    _empty_cuda_cache()
    seq_cuda = _payload_to_cuda(payload, device)
    _cuda_sync()

    batches_done = 0
    total_tensors = 0
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for i in range(0, len(blocks), infer_batch_size):
        if max_batches is not None and batches_done >= max_batches:
            break

        batch_blocks = blocks[i : i + infer_batch_size]
        batch_pattern_tensors = []

        for block in batch_blocks:
            block_sorted = np.asarray(sorted(block), dtype=np.int32)
            if block_sorted.size == model_size:
                quartets_np = block_sorted[local_quartet_template]
            else:
                quartets_np = np.asarray(
                    list(itertools.combinations(block_sorted.tolist(), 4)),
                    dtype=np.int32,
                )
            idx_cuda = torch.from_numpy(quartets_np).to(device=device, dtype=torch.long)
            batch_pattern_tensors.append(_call_pattern_fn(spec, seq_cuda, idx_cuda, seq_length))
            total_tensors += int(batch_pattern_tensors[-1].shape[0])
            del idx_cuda

        del batch_pattern_tensors
        batches_done += 1

    _cuda_sync()
    t1 = time.perf_counter()
    peak_mib = _format_mib(torch.cuda.max_memory_allocated(device))

    del seq_cuda
    _empty_cuda_cache()
    return PatternResult(seconds=t1 - t0, processed_tensors=total_tensors, peak_mib=peak_mib)


def _benchmark_pattern_loop(
    spec: ChainSpec,
    payload: np.ndarray,
    seq_length: int,
    blocks: list[list[int]],
    infer_batch_size: int,
    model_size: int,
    repeats: int,
    max_batches: int | None,
    device: torch.device,
) -> PatternResult:
    best_seconds = math.inf
    best_processed = 0
    best_peak_mib = 0.0
    for _ in range(repeats):
        result = _pattern_loop_once(
            spec,
            payload,
            seq_length,
            blocks,
            infer_batch_size,
            model_size,
            max_batches,
            device,
        )
        if result.seconds < best_seconds:
            best_seconds = result.seconds
            best_processed = result.processed_tensors
            best_peak_mib = result.peak_mib
    return PatternResult(best_seconds, best_processed, best_peak_mib)


def _verify_loader_outputs(
    original: ChainSpec,
    candidate: ChainSpec,
    old_load: LoadResult,
    new_load: LoadResult,
) -> None:
    if old_load.species_names != new_load.species_names:
        raise AssertionError("Species name mismatch between original and candidate loader")
    if old_load.seq_length != new_load.seq_length:
        raise AssertionError("Sequence length mismatch between original and candidate loader")

    if candidate.kind == "packed":
        unpacked = candidate.sp_mod.unpack_packed_tensor(new_load.payload, new_load.seq_length)
        if old_load.payload.shape != unpacked.shape:
            raise AssertionError("Loaded tensor shape mismatch between original and candidate loader")
        if not np.array_equal(old_load.payload, unpacked):
            mismatch_count = int(np.count_nonzero(old_load.payload != unpacked))
            raise AssertionError(f"Loaded tensor mismatch count={mismatch_count}")
    elif candidate.kind == "bitplane":
        unpacked = candidate.sp_mod.unpack_bitplanes_to_tensor(new_load.payload, new_load.seq_length)
        if old_load.payload.shape != unpacked.shape:
            raise AssertionError("Loaded tensor shape mismatch between original and candidate loader")
        if not np.array_equal(old_load.payload, unpacked):
            mismatch_count = int(np.count_nonzero(old_load.payload != unpacked))
            raise AssertionError(f"Loaded tensor mismatch count={mismatch_count}")
    else:
        if old_load.payload.shape != new_load.payload.shape:
            raise AssertionError("Loaded tensor shape mismatch between original and candidate loader")
        if not np.array_equal(old_load.payload, new_load.payload):
            mismatch_count = int(np.count_nonzero(old_load.payload != new_load.payload))
            raise AssertionError(f"Loaded tensor mismatch count={mismatch_count}")


def _verify_pattern_outputs(
    original: ChainSpec,
    candidate: ChainSpec,
    old_load: LoadResult,
    new_load: LoadResult,
    blocks: list[list[int]],
    model_size: int,
    verify_blocks: int,
    device: torch.device,
) -> None:
    local_quartet_template = np.asarray(
        list(itertools.combinations(range(model_size), 4)), dtype=np.int32
    )

    seq_cuda_old = _payload_to_cuda(old_load.payload, device)
    seq_cuda_new = _payload_to_cuda(new_load.payload, device)
    _cuda_sync()

    for block in blocks[:verify_blocks]:
        block_sorted = np.asarray(sorted(block), dtype=np.int32)
        if block_sorted.size == model_size:
            quartets_np = block_sorted[local_quartet_template]
        else:
            quartets_np = np.asarray(
                list(itertools.combinations(block_sorted.tolist(), 4)),
                dtype=np.int32,
            )
        idx_cuda = torch.from_numpy(quartets_np).to(device=device, dtype=torch.long)
        out_old = _call_pattern_fn(original, seq_cuda_old, idx_cuda, old_load.seq_length)
        out_new = _call_pattern_fn(candidate, seq_cuda_new, idx_cuda, new_load.seq_length)
        if out_old.shape != out_new.shape or not torch.equal(out_old, out_new):
            max_abs = float((out_old - out_new).abs().max().item())
            raise AssertionError(f"Pattern output mismatch detected; max_abs_diff={max_abs}")
        del idx_cuda, out_old, out_new

    del seq_cuda_old, seq_cuda_new
    _empty_cuda_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark original vs experimental preprocessing chain.")
    parser.add_argument("--phy", required=True, help="Input PHY path")
    parser.add_argument("--candidate", choices=["v3"], default="v3")
    parser.add_argument("--k-param", type=float, default=3.0)
    parser.add_argument("--infer-batch-size", type=int, default=1)
    parser.add_argument("--model-size", type=int, default=24)
    parser.add_argument("--loader-repeats", type=int, default=1)
    parser.add_argument("--pattern-repeats", type=int, default=1)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Limit batch iterations for controlled experiments",
    )
    parser.add_argument(
        "--verify-blocks",
        type=int,
        default=0,
        help="Number of blocks to compare exactly between original and candidate",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script")

    phy_path = Path(args.phy)
    if not phy_path.exists():
        raise FileNotFoundError(phy_path)

    device = torch.device("cuda")
    original, candidate = _load_extensions(args.candidate)

    print(f"[INFO] Benchmarking PHY: {phy_path}")
    print(f"[INFO] Candidate: {candidate.name}")
    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"[INFO] Settings: infer_batch_size={args.infer_batch_size}, "
        f"max_batches={args.max_batches}, verify_blocks={args.verify_blocks}"
    )

    print("[INFO] Timing loader (original/candidate)...")
    old_load = _time_loader(original, phy_path, device, args.loader_repeats)
    new_load = _time_loader(candidate, phy_path, device, args.loader_repeats)
    _verify_loader_outputs(original, candidate, old_load, new_load)

    num_species = len(old_load.species_names)
    blocks = _build_blocks(num_species, args.k_param, args.model_size)
    print(f"[INFO] num_species={num_species}, effective_seq_length={old_load.seq_length}, total_blocks={len(blocks)}")
    if num_species >= 256 and args.infer_batch_size >= 4 and args.max_batches == 1:
        print(
            "[WARN] Large species count with high infer_batch_size may take very long for pattern loop. "
            "Consider --infer-batch-size 1 for quick benchmarking."
        )

    print("[INFO] Benchmarking pattern loop (original)...")
    old_pattern = _benchmark_pattern_loop(
        original,
        old_load.payload,
        old_load.seq_length,
        blocks,
        args.infer_batch_size,
        args.model_size,
        args.pattern_repeats,
        args.max_batches,
        device,
    )
    print("[INFO] Benchmarking pattern loop (candidate)...")
    new_pattern = _benchmark_pattern_loop(
        candidate,
        new_load.payload,
        new_load.seq_length,
        blocks,
        args.infer_batch_size,
        args.model_size,
        args.pattern_repeats,
        args.max_batches,
        device,
    )

    if args.verify_blocks > 0:
        print(f"[INFO] Verifying exact pattern outputs (blocks={args.verify_blocks})...")
        _verify_pattern_outputs(
            original,
            candidate,
            old_load,
            new_load,
            blocks,
            args.model_size,
            args.verify_blocks,
            device,
        )
    else:
        print("[INFO] Skipping exact pattern output verification (--verify-blocks=0).")

    print("\n[RESULT] Load chain")
    print(
        f"  original: load={old_load.load_seconds:.6f}s h2d={old_load.h2d_seconds:.6f}s "
        f"total={old_load.total_seconds:.6f}s host={_format_mib(old_load.host_bytes):.2f}MiB "
        f"gpu_peak={old_load.h2d_peak_mib:.2f}MiB"
    )
    print(
        f"  {candidate.name}:       load={new_load.load_seconds:.6f}s h2d={new_load.h2d_seconds:.6f}s "
        f"total={new_load.total_seconds:.6f}s host={_format_mib(new_load.host_bytes):.2f}MiB "
        f"gpu_peak={new_load.h2d_peak_mib:.2f}MiB"
    )
    print(
        f"  speedup:  x{(old_load.total_seconds / new_load.total_seconds):.3f}"
        if new_load.total_seconds > 0
        else "  speedup:  n/a"
    )

    print("\n[RESULT] Pattern frequency chain")
    print(f"  processed_tensors(original)={old_pattern.processed_tensors}")
    print(f"  processed_tensors({candidate.name})={new_pattern.processed_tensors}")
    print(f"  original: {old_pattern.seconds:.6f}s peak={old_pattern.peak_mib:.2f}MiB")
    print(f"  {candidate.name}:       {new_pattern.seconds:.6f}s peak={new_pattern.peak_mib:.2f}MiB")
    print(
        f"  speedup:  x{(old_pattern.seconds / new_pattern.seconds):.3f}"
        if new_pattern.seconds > 0
        else "  speedup:  n/a"
    )
    print(f"\n[CHECK] Outputs match exactly between original and {candidate.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
