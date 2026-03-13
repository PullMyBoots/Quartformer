import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from model import QuartFormer


REPO_ROOT = Path(__file__).resolve().parent
SERVER_BIN_DIR = REPO_ROOT / "server_bin"


def _load_extension(module_name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _PackedSequenceProcessorAdapter:
    def __init__(self, impl):
        self._impl = impl
        self._last_seq_length = None

    def load_phy_to_tensor(self, phy_path, drop_conserved_sites=False):
        packed, species_names, seq_length = self._impl.load_phy_to_packed_tensor(
            phy_path,
            drop_conserved_sites,
        )
        self._last_seq_length = int(seq_length)
        return packed, species_names

    @property
    def seq_length(self) -> int:
        if self._last_seq_length is None:
            raise RuntimeError("Sequence length is not initialized")
        return self._last_seq_length


class _PackedPatternFreqAdapter:
    def __init__(self, impl, sp_adapter: _PackedSequenceProcessorAdapter):
        self._impl = impl
        self._sp_adapter = sp_adapter

    def compute_pattern_frequencies_cuda(self, sequences, quartet_indices):
        return self._impl.compute_pattern_frequencies_cuda_packed(
            sequences,
            quartet_indices,
            self._sp_adapter.seq_length,
        )


def _load_backend_modules(extension_backend: str):
    backend = extension_backend.strip().lower().replace("-", "_")
    if backend == "server_bin":
        sp_module = _load_extension(
            "sequence_processor_server",
            SERVER_BIN_DIR / "sequence_processor_server.cpython-310-x86_64-linux-gnu.so",
        )
        pattern_module = _load_extension(
            "pattern_freq_cuda_server",
            SERVER_BIN_DIR / "pattern_freq_cuda_server.cpython-310-x86_64-linux-gnu.so",
        )
        return sp_module, pattern_module
    if backend == "cpp_source":
        packed_sp = _load_extension(
            "sequence_processor",
            REPO_ROOT / "cpp_source" / "sequence_processor.cpython-310-x86_64-linux-gnu.so",
        )
        packed_pattern = _load_extension(
            "pattern_freq_cuda",
            REPO_ROOT / "cpp_source" / "cuda13_pattern_freq" / "pattern_freq_cuda.cpython-310-x86_64-linux-gnu.so",
        )
        sp_adapter = _PackedSequenceProcessorAdapter(packed_sp)
        pattern_adapter = _PackedPatternFreqAdapter(packed_pattern, sp_adapter)
        return sp_adapter, pattern_adapter
    raise ValueError(f"Unsupported backend: {extension_backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-batch model inputs/outputs between cpp_source and server_bin backends."
    )
    parser.add_argument(
        "--phy",
        default="data/ml_rf_real/plant1/MSA.phy",
        help="Input PHY file relative to repo root",
    )
    parser.add_argument(
        "--task-type",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
    )
    parser.add_argument("--k-param", type=float, default=3.0)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument(
        "--save-dir",
        default="output/backend_io_compare/plant1",
        help="Directory for summary and dumped tensors",
    )
    parser.add_argument(
        "--save-mode",
        choices=["mismatch", "all", "none"],
        default="mismatch",
        help="Dump tensors for mismatched batches or all batches",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional batch limit, 0 means all",
    )
    parser.add_argument(
        "--dump-logits",
        action="store_true",
        help="Also save full logits tensors for selected batches",
    )
    parser.add_argument(
        "--normalize-input-tensor",
        action="store_true",
        help="Force input_batch_for_model through contiguous().clone() before forward",
    )
    parser.add_argument(
        "--normalize-pattern-batch",
        action="store_true",
        help="Force pattern_batch through contiguous().clone() before concatenation",
    )
    parser.add_argument(
        "--sync-before-forward",
        action="store_true",
        help="Call torch.cuda.synchronize() before model forward",
    )
    return parser.parse_args()


def _build_blocks(num_species: int, k_param: float, model_size: int = 24):
    if num_species == model_size:
        return [list(range(num_species))]
    batching_algorithms = _load_extension(
        "batching_algorithms",
        REPO_ROOT / "cpp_source" / "batching_algorithms.cpython-310-x86_64-linux-gnu.so",
    )
    return batching_algorithms.pair_balanced_block_design_v5(
        num_species=num_species,
        k_param=k_param,
        batch_size=model_size,
        seed=42,
        species_weight=2.0,
        threads=1,
    )


def _prepare_backend_state(backend: str, phy_path: Path, device: torch.device):
    sp, pattern = _load_backend_modules(backend)
    seq_tensor, species_names = sp.load_phy_to_tensor(str(phy_path), drop_conserved_sites=True)
    seq_cuda = torch.from_numpy(np.asarray(seq_tensor)).to(device)
    return {
        "name": backend,
        "sp": sp,
        "pattern": pattern,
        "seq_tensor": seq_tensor,
        "species_names": list(species_names),
        "seq_cuda": seq_cuda,
        "num_species": int(seq_tensor.shape[0]),
        "effective_seq_length": int(seq_tensor.shape[1]),
    }


def _batch_quartets(batch_blocks, model_size: int, quartet_template: np.ndarray):
    batch_quartets_np = []
    for block in batch_blocks:
        block_sorted = np.asarray(sorted(block), dtype=np.int32)
        if block_sorted.size == model_size:
            quartets_np = block_sorted[quartet_template]
        else:
            quartets_np = np.asarray(list(itertools.combinations(block_sorted.tolist(), 4)), dtype=np.int32)
        batch_quartets_np.append(quartets_np)
    return batch_quartets_np


def _save_tensor(path: Path, tensor: torch.Tensor):
    torch.save(tensor.detach().cpu(), path)


def _tensor_layout_meta(tensor: torch.Tensor) -> dict:
    ptr = int(tensor.data_ptr())
    return {
        "shape": [int(x) for x in tensor.shape],
        "stride": [int(x) for x in tensor.stride()],
        "is_contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": ptr,
        "data_ptr_mod_16": ptr % 16,
        "data_ptr_mod_32": ptr % 32,
        "data_ptr_mod_64": ptr % 64,
        "data_ptr_mod_128": ptr % 128,
    }


def main() -> int:
    args = parse_args()
    phy_path = (REPO_ROOT / args.phy).resolve()
    save_dir = (REPO_ROOT / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this diagnostic")

    backend_a = _prepare_backend_state("cpp_source", phy_path, device)
    backend_b = _prepare_backend_state("server_bin", phy_path, device)
    if backend_a["species_names"] != backend_b["species_names"]:
        raise RuntimeError("Species names differ between backends")
    if backend_a["num_species"] != backend_b["num_species"]:
        raise RuntimeError("Species counts differ between backends")

    num_species = backend_a["num_species"]
    blocks = _build_blocks(num_species, args.k_param)
    total_batches = (len(blocks) + args.infer_batch_size - 1) // args.infer_batch_size
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    model_dir = REPO_ROOT / "model" / args.task_type / "24"
    species_enc = torch.load(model_dir / "species_encoding.pt", map_location="cpu", weights_only=False).to(device)
    coeff_blocks = torch.load(model_dir / "coeff_blocks.pt", map_location="cpu", weights_only=False).to(device)
    quartet_matrix = torch.load(model_dir / "quartet_matrix.pt", map_location="cpu", weights_only=False).to(device)
    attn_model = QuartFormer(species_num=24)
    attn_model.load_state_dict(torch.load(model_dir / "qf1.pt", map_location="cpu", weights_only=False))
    attn_model = attn_model.to(device).eval()

    quartet_template = np.asarray(list(itertools.combinations(range(24), 4)), dtype=np.int32)
    expected_valid_len = 10626
    expected_padded_len = 10640
    pad_len = expected_padded_len - expected_valid_len

    print(
        f"[INFO] phy={phy_path}, num_species={num_species}, total_blocks={len(blocks)}, "
        f"compare_batches={total_batches}, device={device}"
    )

    batch_rows = []
    mismatch_batches = []

    for batch_idx in range(total_batches):
        start = batch_idx * args.infer_batch_size
        end = min(start + args.infer_batch_size, len(blocks))
        batch_blocks = blocks[start:end]
        actual_bs = len(batch_blocks)
        batch_quartets = _batch_quartets(batch_blocks, 24, quartet_template)

        backend_results = {}
        for backend_state in (backend_a, backend_b):
            pattern_tensors = []
            for quartets_np in batch_quartets:
                idx_cuda = torch.from_numpy(quartets_np).to(device=device, dtype=torch.long)
                pattern_tensors.append(
                    backend_state["pattern"].compute_pattern_frequencies_cuda(
                        backend_state["seq_cuda"],
                        idx_cuda,
                    )
                )
            pattern_batch = torch.stack(pattern_tensors, dim=0)
            if args.normalize_pattern_batch:
                pattern_batch = pattern_batch.contiguous().clone()
            species_batch = species_enc.unsqueeze(0).expand(actual_bs, -1, -1)
            input_batch = torch.cat([species_batch, pattern_batch], dim=2)
            input_batch_for_model = torch.cat(
                [input_batch, input_batch.new_zeros((actual_bs, pad_len, input_batch.size(2)))],
                dim=1,
            )
            if args.normalize_input_tensor:
                input_batch_for_model = input_batch_for_model.contiguous().clone()
            if args.sync_before_forward:
                torch.cuda.synchronize(device)
            with torch.no_grad():
                logits = attn_model(input_batch_for_model, coeff_blocks, quartet_matrix)
                logits = logits[:, :expected_valid_len, :]
                probs = torch.softmax(logits, dim=-1)
                weights = probs * 100.0
            backend_results[backend_state["name"]] = {
                "pattern_batch": pattern_batch,
                "input_batch": input_batch_for_model,
                "logits": logits,
                "probs": probs,
                "weights": weights,
                "input_layout": _tensor_layout_meta(input_batch_for_model),
                "pattern_layout": _tensor_layout_meta(pattern_batch),
            }

        row = {
            "batch_idx": batch_idx,
            "block_start": start,
            "block_end": end,
            "batch_size": actual_bs,
            "cpp_source_input_layout": backend_results["cpp_source"]["input_layout"],
            "server_bin_input_layout": backend_results["server_bin"]["input_layout"],
            "cpp_source_pattern_layout": backend_results["cpp_source"]["pattern_layout"],
            "server_bin_pattern_layout": backend_results["server_bin"]["pattern_layout"],
        }
        mismatched = False
        for key in ("pattern_batch", "input_batch", "logits", "probs", "weights"):
            a_tensor = backend_results["cpp_source"][key]
            b_tensor = backend_results["server_bin"][key]
            diff = torch.abs(a_tensor - b_tensor)
            max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
            row[f"{key}_max_abs_diff"] = max_abs_diff
            row[f"{key}_equal"] = bool(max_abs_diff == 0.0)
            if max_abs_diff != 0.0:
                mismatched = True
                idx = np.unravel_index(int(diff.argmax().item()), diff.shape)
                row[f"{key}_first_bad_index"] = [int(i) for i in idx]
                row[f"{key}_cpp_source_value"] = float(a_tensor[idx].item())
                row[f"{key}_server_bin_value"] = float(b_tensor[idx].item())

        row["matched"] = not mismatched
        batch_rows.append(row)
        if mismatched:
            mismatch_batches.append(batch_idx)

        should_save = args.save_mode == "all" or (args.save_mode == "mismatch" and mismatched)
        if should_save:
            batch_dir = save_dir / f"batch_{batch_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            for backend_name, outputs in backend_results.items():
                _save_tensor(batch_dir / f"{backend_name}_pattern_batch.pt", outputs["pattern_batch"])
                _save_tensor(batch_dir / f"{backend_name}_input_batch.pt", outputs["input_batch"])
                _save_tensor(batch_dir / f"{backend_name}_weights.pt", outputs["weights"])
                _save_tensor(batch_dir / f"{backend_name}_probs.pt", outputs["probs"])
                if args.dump_logits:
                    _save_tensor(batch_dir / f"{backend_name}_logits.pt", outputs["logits"])

        print(
            f"[{'DIFF' if mismatched else 'match'}] batch={batch_idx:04d} blocks={start}:{end} "
            f"pattern={row['pattern_batch_max_abs_diff']:.3e} "
            f"input={row['input_batch_max_abs_diff']:.3e} "
            f"logits={row['logits_max_abs_diff']:.3e} "
            f"probs={row['probs_max_abs_diff']:.3e} "
            f"weights={row['weights_max_abs_diff']:.3e}"
        )

    summary = {
        "phy": str(phy_path),
        "num_species": num_species,
        "total_blocks": len(blocks),
        "compared_batches": total_batches,
        "mismatch_batches": mismatch_batches,
        "all_equal": len(mismatch_batches) == 0,
        "rows": batch_rows,
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Summary saved to {summary_path}")
    print(f"[FINAL] all_equal={summary['all_equal']}, mismatch_batches={mismatch_batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
