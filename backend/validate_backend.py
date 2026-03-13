#!/usr/bin/env python3
import argparse
import importlib.util
import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_BIN_DIR = REPO_ROOT / "server_bin"
BACKEND_DIR = REPO_ROOT / "backend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import QuartFormer


def _load_extension(module_name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_first_extension(module_name: str, directory: Path, stem: str):
    matches = sorted(directory.glob(f"{stem}*.so"))
    if not matches:
        raise FileNotFoundError(f"未找到扩展: {directory / (stem + '*.so')}")
    return _load_extension(module_name, matches[0])


def _build_blocks(n_species: int, k_param: float, model_size: int, num_blocks: int):
    if n_species == model_size:
        return [list(range(n_species))]

    so_path = REPO_ROOT / "cpp_source" / "batching_algorithms.cpython-310-x86_64-linux-gnu.so"
    spec = importlib.util.spec_from_file_location("batching_algorithms", so_path)
    batching_algorithms = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(batching_algorithms)
    blocks = batching_algorithms.pair_balanced_block_design_v5(
        num_species=n_species,
        k_param=k_param,
        batch_size=model_size,
        seed=42,
        species_weight=2.0,
        threads=1,
    )
    return blocks[:num_blocks]


def _default_phy_files():
    candidates = [
        REPO_ROOT / "data" / "24" / "0" / "GTR_10000000_MSA.phy",
        REPO_ROOT / "data" / "48" / "0" / "GTR_100000_MSA.phy",
        REPO_ROOT / "data" / "96" / "0" / "GTR_100000_MSA.phy",
        REPO_ROOT / "data" / "192" / "0" / "GTR_100000_MSA.phy",
        REPO_ROOT / "data" / "320" / "0" / "GTR_100000_MSA.phy",
    ]
    return [p for p in candidates if p.exists()]


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate backend packed pipeline against server_bin up to model logits.")
    parser.add_argument("--phy", action="append", help="PHY file path; can be specified multiple times")
    parser.add_argument("--task-type", default="homogeneous", choices=["homogeneous", "heterogeneous"])
    parser.add_argument("--drop-conserved-sites", action="store_true", default=True)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--k-param", type=float, default=3.0)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args(argv)

    phy_files = [Path(p) for p in args.phy] if args.phy else _default_phy_files()
    if not phy_files:
        print("[ERROR] 没有可验证的 PHY 文件")
        return 1

    if len(phy_files) > 1:
        failures = []
        for phy_path in phy_files:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phy",
                str(phy_path),
                "--task-type",
                args.task_type,
                "--num-blocks",
                str(args.num_blocks),
                "--k-param",
                str(args.k_param),
                "--atol",
                str(args.atol),
            ]
            if args.drop_conserved_sites:
                cmd.append("--drop-conserved-sites")

            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            if proc.returncode != 0:
                failures.append(str(phy_path))

        if failures:
            print("[RESULT] 验证失败:")
            for item in failures:
                print(f"  - {item}")
            return 1

        print("[RESULT] 所有对拍通过")
        return 0

    ref_sp = _load_extension(
        "sequence_processor_server",
        SERVER_BIN_DIR / "sequence_processor_server.cpython-310-x86_64-linux-gnu.so",
    )
    ref_pf = _load_extension(
        "pattern_freq_cuda_server",
        SERVER_BIN_DIR / "pattern_freq_cuda_server.cpython-310-x86_64-linux-gnu.so",
    )
    fast_sp = _load_first_extension("sequence_processor_backend", BACKEND_DIR, "sequence_processor_backend")
    fast_pf = _load_first_extension("pattern_freq_cuda_backend", BACKEND_DIR, "pattern_freq_cuda_backend")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[ERROR] validate_backend.py 需要 CUDA 环境")
        return 1

    model_dir = REPO_ROOT / "model" / args.task_type / "24"
    coeff_blocks = torch.load(model_dir / "coeff_blocks.pt", map_location="cpu", weights_only=False).to(device)
    quartet_matrix = torch.load(model_dir / "quartet_matrix.pt", map_location="cpu", weights_only=False).to(device)
    species_enc = torch.load(model_dir / "species_encoding.pt", map_location="cpu", weights_only=False).to(device)
    model_state = torch.load(model_dir / "qf1.pt", map_location="cpu", weights_only=False)

    expected_valid_len = math.comb(24, 4)
    expected_padded_len = 10640
    pad_len = expected_padded_len - expected_valid_len
    quartet_template = np.asarray(list(itertools.combinations(range(24), 4)), dtype=np.int32)

    failures = []
    for phy_path in phy_files:
        dense_ref, names_ref = ref_sp.load_phy_to_tensor(str(phy_path), args.drop_conserved_sites)
        packed_fast, names_fast, seq_length = fast_sp.load_phy_to_packed_tensor(str(phy_path), args.drop_conserved_sites)
        unpacked_fast = fast_sp.unpack_packed_tensor(packed_fast, int(seq_length))

        dense_ref_np = np.asarray(dense_ref)
        unpacked_fast_np = np.asarray(unpacked_fast)
        packed_fast_np = np.asarray(packed_fast)

        checks = []
        checks.append(("species_names", names_ref == names_fast))
        checks.append(("dense_equal", np.array_equal(dense_ref_np, unpacked_fast_np)))

        n_species = dense_ref_np.shape[0]
        blocks = _build_blocks(n_species, args.k_param, 24, args.num_blocks)
        batch_quartets = np.empty((len(blocks), expected_valid_len, 4), dtype=np.int32)
        for batch_idx, block in enumerate(blocks):
            block_sorted = np.asarray(sorted(block), dtype=np.int32)
            batch_quartets[batch_idx] = block_sorted[quartet_template]

        quartets_flat = batch_quartets.reshape(-1, 4)
        idx_cuda = torch.as_tensor(quartets_flat, device=device, dtype=torch.long)
        seq_ref_cuda = torch.from_numpy(dense_ref_np).to(device=device, dtype=torch.int8)
        seq_fast_cuda = torch.from_numpy(packed_fast_np).to(device=device, dtype=torch.uint8)

        with torch.no_grad():
            ref_pattern = ref_pf.compute_pattern_frequencies_cuda(seq_ref_cuda, idx_cuda).contiguous()
            fast_pattern = fast_pf.compute_pattern_frequencies_cuda_packed(
                seq_fast_cuda,
                idx_cuda,
                int(seq_length),
            ).contiguous()

        pattern_diff = _max_abs_diff(ref_pattern, fast_pattern)
        checks.append(("pattern_allclose", pattern_diff <= args.atol))

        batch_size = len(blocks)
        ref_pattern_batch = ref_pattern.view(batch_size, expected_valid_len, -1).contiguous()
        fast_pattern_batch = fast_pattern.view(batch_size, expected_valid_len, -1).contiguous()
        species_batch = species_enc.unsqueeze(0).expand(batch_size, -1, -1)

        input_ref = torch.cat([species_batch, ref_pattern_batch], dim=2).contiguous()
        input_fast = torch.cat([species_batch, fast_pattern_batch], dim=2).contiguous()
        input_diff = _max_abs_diff(input_ref, input_fast)
        checks.append(("input_allclose", input_diff <= args.atol))

        input_ref_padded = torch.cat(
            [input_ref, input_ref.new_zeros((batch_size, pad_len, input_ref.size(2)))],
            dim=1,
        ).contiguous()
        input_fast_padded = torch.cat(
            [input_fast, input_fast.new_zeros((batch_size, pad_len, input_fast.size(2)))],
            dim=1,
        ).contiguous()

        model = QuartFormer(species_num=24)
        model.load_state_dict(model_state)
        model = model.to(device).eval()
        model_input = torch.empty_like(input_ref_padded)

        with torch.no_grad():
            model_input.copy_(input_ref_padded)
            logits_ref = model(model_input, coeff_blocks, quartet_matrix).contiguous()
            model_input.copy_(input_fast_padded)
            logits_fast = model(model_input, coeff_blocks, quartet_matrix).contiguous()
            logits_ref = logits_ref[:, :expected_valid_len, :]
            logits_fast = logits_fast[:, :expected_valid_len, :]
            probs_ref = torch.softmax(logits_ref, dim=-1)
            probs_fast = torch.softmax(logits_fast, dim=-1)

        logits_diff = _max_abs_diff(logits_ref, logits_fast)
        probs_diff = _max_abs_diff(probs_ref, probs_fast)
        weights_ref = torch.round(probs_ref * 100.0).to(torch.int32)
        weights_fast = torch.round(probs_fast * 100.0).to(torch.int32)
        checks.append(("logits_allclose", logits_diff <= args.atol))
        checks.append(("probs_allclose", probs_diff <= args.atol))
        checks.append(("weights_equal", torch.equal(weights_ref, weights_fast)))

        ok = all(flag for _, flag in checks)
        print(f"[CHECK] {phy_path}")
        print(f"  species={n_species}, seq_length={int(seq_length)}, blocks={batch_size}")
        print(f"  pattern_diff={pattern_diff:.3e}, input_diff={input_diff:.3e}, logits_diff={logits_diff:.3e}, probs_diff={probs_diff:.3e}")
        for name, flag in checks:
            print(f"  {name}: {'OK' if flag else 'FAIL'}")

        if not ok:
            failures.append(str(phy_path))

        del model
        torch.cuda.synchronize()

    if failures:
        print("[RESULT] 验证失败:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("[RESULT] 所有对拍通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
