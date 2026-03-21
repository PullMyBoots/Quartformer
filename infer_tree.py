import argparse
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ete3 import Tree

REPO_ROOT = Path(__file__).resolve().parent
SERVER_BIN_DIR = REPO_ROOT / "server_bin"
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_CONFIG_PATH = REPO_ROOT / "infer_config.jsonc"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import QuartFormer, MLP
from utils import draw_newick_tree_figure, find_polytomy_representative_leaves


def _load_extension(module_name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_extension_by_stem(module_name: str, directory: Path, stem: str):
    matches = sorted(directory.glob(f"{stem}*.so"))
    if not matches:
        raise FileNotFoundError(f"Extension module not found: {directory / (stem + '*.so')}")
    return _load_extension(module_name, matches[0])


def _try_load_extension_by_stem(module_name: str, directory: Path, stem: str):
    matches = sorted(directory.glob(f"{stem}*.so"))
    if not matches:
        return None
    return _load_extension(module_name, matches[0])


class _PackedSequenceBackendAdapter:
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
    def sequence_length(self) -> int:
        if self._last_seq_length is None:
            raise RuntimeError("Sequence length not initialized before pattern frequency call")
        return self._last_seq_length


class _PackedPatternBackendAdapter:
    def __init__(self, impl, sp_adapter: _PackedSequenceBackendAdapter):
        self._impl = impl
        self._sp_adapter = sp_adapter

    def compute_pattern_frequencies_cuda(self, sequences, quartet_indices):
        return self._impl.compute_pattern_frequencies_cuda_packed(
            sequences,
            quartet_indices,
            self._sp_adapter.sequence_length,
        ).contiguous()


sp = _PackedSequenceBackendAdapter(
    _load_extension_by_stem("sequence_processor_backend", BACKEND_DIR, "sequence_processor_backend")
)
pattern_freq_cuda = _PackedPatternBackendAdapter(
    _load_extension_by_stem("pattern_freq_cuda_backend", BACKEND_DIR, "pattern_freq_cuda_backend"),
    sp,
)
quartet_aggregate_backend = _try_load_extension_by_stem(
    "quartet_aggregate_backend",
    BACKEND_DIR,
    "quartet_aggregate_backend",
)


def _load_runtime_config() -> dict:
    config_path = Path(os.environ.get("QF_INFER_CONFIG", str(DEFAULT_CONFIG_PATH)))
    cfg = _load_jsonc_config(config_path, section=None)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be an object/dict: {config_path}")
    # Preferred layout:
    # - advanced: advanced algorithm/perf knobs
    # - runtime: backward-compatible alias (older configs)
    # If neither exists, fall back to root-level keys.
    advanced = cfg.get("advanced")
    if isinstance(advanced, dict):
        return advanced
    runtime = cfg.get("runtime")
    if isinstance(runtime, dict):
        return runtime
    return cfg


def _load_jsonc_config(config_path: Path, section: str | None = None) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    raw = re.sub(r"^\s*//.*$", "", raw, flags=re.M)
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be an object/dict: {config_path}")
    if section is None:
        return cfg
    section_cfg = cfg.get(section)
    if section_cfg is None:
        return cfg
    if not isinstance(section_cfg, dict):
        raise ValueError(f"Config section '{section}' must be an object/dict: {config_path}")
    return section_cfg


def _parse_bool_config(value, key_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{key_name} only supports boolean values or 0/1")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"{key_name} only supports boolean values (true/false)")


def run_qf(
    phy_path,
    output_tree_path,
    task_type="homogeneous",
    infer_batch_size=32,
    ref_tree_path=None,
    metric: str = "rf",
    compute_branch_support: bool = False,
    keep_support_files: bool = False,
):
    """
    Enhanced standalone run_QF_framework:
    - Supports datasets with species count >= 24.
    - Uses batched inference.
    - Averages duplicated quartet weights.
    - Includes automatic polytomy resolution with MLP.
    - Model family is controlled by task_type (homogeneous / heterogeneous).
    - QuartFormer quartet weights can keep all 3 topologies or only top1.
    - MLP polytomy resolution always uses QFM-FI.
    - If compute_branch_support=True, keeps weighted quartets from QuartFormer+MLP
      and runs TREE-QMC --supportonly on the final fully-binary tree.
    - By default only tree files are kept; if keep_support_files=True, support
      artifacts (CSV/quartets) are preserved.
    - Runtime parameters are loaded from infer_config.jsonc.
    - Tree assembler is controlled by quartet_assembler (qfm / qmc).
    """
    phy_path = Path(phy_path)
    output_tree_path = _normalize_output_path(output_tree_path, phy_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = 24
    cfg = _load_runtime_config()
    k_param = float(cfg.get("k_param", 3.0))
    quartet_assembler = str(cfg.get("quartet_assembler", "")).strip().lower()
    if not quartet_assembler and "run_mode" in cfg:
        legacy_mode = str(cfg.get("run_mode", "regular")).strip().lower().replace("_", "-")
        quartet_assembler = "qfm" if legacy_mode == "slow" else "qmc"
    if not quartet_assembler:
        quartet_assembler = "qmc"
    qmc_iter_limit = int(cfg.get("qmc_iter_limit", 10))
    aggregate_mode = str(cfg.get("aggregate_mode", "off"))
    quartformer_top1_only = _parse_bool_config(
        cfg.get("quartformer_top1_only", False),
        "quartformer_top1_only",
    )
    aggregate_intermediate_flush = _parse_bool_config(
        cfg.get("aggregate_intermediate_flush", False),
        "aggregate_intermediate_flush",
    )
    aggregate_flush_threshold_cfg = int(cfg.get("aggregate_flush_threshold", 0))

    model_label = task_type

    def _model_dir(species_for_model: int) -> Path:
        return REPO_ROOT / "model" / model_label / str(species_for_model)

    def _encode_quartet_keys(quartets_np: np.ndarray) -> np.ndarray:
        q = quartets_np.astype(np.uint64, copy=False)
        return (q[:, 0] << 48) | (q[:, 1] << 32) | (q[:, 2] << 16) | q[:, 3]

    def _decode_quartet_keys(keys_np: np.ndarray) -> np.ndarray:
        quartets_np = np.empty((keys_np.shape[0], 4), dtype=np.int32)
        quartets_np[:, 0] = ((keys_np >> 48) & 0xFFFF).astype(np.int32)
        quartets_np[:, 1] = ((keys_np >> 32) & 0xFFFF).astype(np.int32)
        quartets_np[:, 2] = ((keys_np >> 16) & 0xFFFF).astype(np.int32)
        quartets_np[:, 3] = (keys_np & 0xFFFF).astype(np.int32)
        return quartets_np

    def _encode_split_keys(pair1: np.ndarray, pair2: np.ndarray) -> np.ndarray:
        p1 = pair1.astype(np.uint64, copy=False)
        p2 = pair2.astype(np.uint64, copy=False)
        return (p1[:, 0] << 48) | (p1[:, 1] << 32) | (p2[:, 0] << 16) | p2[:, 1]

    selected_assembler = None
    selected_tree_qmc_iter_limit = None
    if quartet_assembler not in ("qfm", "qmc"):
        raise ValueError("quartet_assembler only supports qfm or qmc")
    if qmc_iter_limit <= 0:
        raise ValueError("qmc_iter_limit must be > 0")

    aggregate_mode_key = aggregate_mode.strip().lower()
    if aggregate_mode_key not in ("full", "batch_only", "off"):
        raise ValueError("aggregate_mode only supports full, batch_only, or off")

    if quartet_aggregate_backend is None:
        raise FileNotFoundError(
            "backend/quartet_aggregate_backend extension was not found; NumPy aggregation fallback is no longer supported"
        )
    use_cpp_quartet_aggregate = True
    # Use a fixed conservative shard strategy to avoid exposing tuning knobs.
    quartet_aggregate_shard_count = max(1, (os.cpu_count() or 1) * 2)
    if not hasattr(quartet_aggregate_backend, "aggregates_to_splits"):
        raise RuntimeError(
            "quartet_aggregate_backend is missing aggregates_to_splits; NumPy split fallback is no longer supported"
        )
    use_cpp_split_convert = True
    last_assembler_error = ""

    def _set_assembler_error(cmd, returncode=None, stdout_text=None, stderr_text=None, exc: Exception | None = None):
        nonlocal last_assembler_error
        lines = [f"cmd={' '.join(str(x) for x in cmd)}"]
        if returncode is not None:
            lines.append(f"returncode={returncode}")
        if exc is not None:
            lines.append(f"exception={exc}")
        if stdout_text:
            lines.append(f"stdout={stdout_text.strip()[:2000]}")
        if stderr_text:
            lines.append(f"stderr={stderr_text.strip()[:4000]}")
        last_assembler_error = " | ".join(lines)

    def _aggregate_quartets(quartets_np: np.ndarray, weights_np: np.ndarray):
        cpp_result = quartet_aggregate_backend.aggregate_quartets(
            quartets_np,
            weights_np,
            num_shards=quartet_aggregate_shard_count,
        )
        return cpp_result

    def _reduce_aggregates(keys_np: np.ndarray, sums_np: np.ndarray, counts_np: np.ndarray):
        cpp_result = quartet_aggregate_backend.reduce_aggregates(
            keys_np,
            sums_np,
            counts_np,
            num_shards=quartet_aggregate_shard_count,
        )
        return cpp_result

    def _run_quartet_assembler(
        input_file: Path,
        output_file: Path,
        assembler_override: str | None = None,
        qmc_compact_format: bool = False,
    ) -> bool:
        nonlocal last_assembler_error
        last_assembler_error = ""
        assembler_to_use = assembler_override or selected_assembler
        if assembler_to_use is None:
            raise RuntimeError("Internal error: assembler is not initialized")
        if assembler_to_use == "qfm-fi":
            qfm_fast_jar = REPO_ROOT / "quartet_assemble_method" / "qfm_java_fast" / "QFM-FI_unzipped" / "QFM-FI-fast.jar"
            if not qfm_fast_jar.exists():
                raise FileNotFoundError(f"QFM-FI-fast jar not found: {qfm_fast_jar}")
            if input_file.suffix == ".bin":
                input_format = "bin"
            else:
                input_format = "newick"
            cmd = [
                "java",
                "-jar",
                str(qfm_fast_jar),
                "--input",
                str(input_file),
                "--input-format",
                input_format,
                "--output",
                str(output_file),
                "--taxon-prefix",
                taxon_prefix,
            ]
            result = subprocess.run(
                cmd,
                check=False, capture_output=True
            )
            if result.returncode != 0 or not output_file.exists():
                _set_assembler_error(cmd, result.returncode, result.stdout, result.stderr)
            return result.returncode == 0 and output_file.exists()
        
        # Default to TREE-QMC / QMC
        tree_qmc_bin_path = REPO_ROOT / "quartet_assemble_method" / "TREE-QMC_fast" / "build_local" / "tree-qmc"
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("tree-qmc executable not found")
        quartet_fmt = "___,___|___,___:___" if qmc_compact_format else "((___,___),(___,___));___"
        cmd = [
            str(tree_qmc_bin_path), "-i", str(input_file), "--quartets",
            "--quartetformat", quartet_fmt, "-o", str(output_file), "--override"
        ]
        if selected_tree_qmc_iter_limit is not None:
            cmd.extend(["--iterlimit", str(selected_tree_qmc_iter_limit)])
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not output_file.exists():
            _set_assembler_error(cmd, result.returncode, result.stdout, result.stderr)
        return result.returncode == 0 and output_file.exists()

    def _run_tree_qmc_from_memory(
        split_keys: np.ndarray,
        split_weights: np.ndarray,
        output_file: Path,
        qmc_compact_format: bool = True,
    ) -> bool:
        nonlocal last_assembler_error
        last_assembler_error = ""
        tree_qmc_bin_path = REPO_ROOT / "quartet_assemble_method" / "TREE-QMC_fast" / "build_local" / "tree-qmc"
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("tree-qmc executable not found")
        quartet_fmt = "___,___|___,___:___" if qmc_compact_format else "((___,___),(___,___));___"
        try:
            cmd = [
                str(tree_qmc_bin_path),
                "-i",
                "-",
                "--quartets",
                "--quartets-bin",
                "--quartetformat",
                quartet_fmt,
                "-o",
                str(output_file),
                "--override",
            ]
            if selected_tree_qmc_iter_limit is not None:
                cmd.extend(["--iterlimit", str(selected_tree_qmc_iter_limit)])

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.stdin is None:
                _set_assembler_error(cmd, exc=RuntimeError("Failed to open tree-qmc stdin"))
                return False
            try:
                _write_split_records_binary(
                    proc.stdin,
                    split_keys,
                    split_weights,
                )
                proc.stdin.close()
            except Exception as exc:
                proc.stdin.close()
                proc.stdin = None
                stdout_text, stderr_text = proc.communicate()
                _set_assembler_error(
                    cmd,
                    proc.returncode,
                    stdout_text.decode("utf-8", "ignore"),
                    stderr_text.decode("utf-8", "ignore"),
                    exc=exc,
                )
                return False
            proc.stdin = None
            stdout_text, stderr_text = proc.communicate()
            ok = proc.returncode == 0 and output_file.exists()
            if not ok:
                _set_assembler_error(
                    cmd,
                    proc.returncode,
                    stdout_text.decode("utf-8", "ignore"),
                    stderr_text.decode("utf-8", "ignore"),
                )
            return ok
        except Exception as exc:
            _set_assembler_error(cmd if "cmd" in locals() else ["tree-qmc"], exc=exc)
            return False

    def _run_tree_qmc_from_binary_file(
        binary_input_file: Path,
        output_file: Path,
        qmc_compact_format: bool = True,
    ) -> bool:
        nonlocal last_assembler_error
        last_assembler_error = ""
        tree_qmc_bin_path = REPO_ROOT / "quartet_assemble_method" / "TREE-QMC_fast" / "build_local" / "tree-qmc"
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("tree-qmc executable not found")
        if not binary_input_file.exists():
            _set_assembler_error(["tree-qmc"], exc=FileNotFoundError(f"Binary quartet file not found: {binary_input_file}"))
            return False
        quartet_fmt = "___,___|___,___:___" if qmc_compact_format else "((___,___),(___,___));___"
        cmd = [
            str(tree_qmc_bin_path),
            "-i",
            str(binary_input_file),
            "--quartets",
            "--quartets-bin",
            "--quartetformat",
            quartet_fmt,
            "-o",
            str(output_file),
            "--override",
        ]
        if selected_tree_qmc_iter_limit is not None:
            cmd.extend(["--iterlimit", str(selected_tree_qmc_iter_limit)])
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not output_file.exists():
            _set_assembler_error(cmd, result.returncode, result.stdout, result.stderr)
        return result.returncode == 0 and output_file.exists()

    def _run_qmc_support_annotation(
        quartet_file: Path,
        base_tree_file: Path,
        support_tree_file: Path,
        support_table_file: Path,
    ) -> bool:
        tree_qmc_bin_path = REPO_ROOT / "quartet_assemble_method" / "TREE-QMC_fast" / "build_local" / "tree-qmc"
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("tree-qmc executable not found")
        cmd = [
            str(tree_qmc_bin_path),
            "-i",
            str(quartet_file),
            "--quartets",
            "--quartetformat",
            "___,___|___,___:___",
            "--supportonly",
            str(base_tree_file),
            "-o",
            str(support_tree_file),
            "--override",
            "--writetable",
            str(support_table_file),
        ]
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.returncode == 0 and support_tree_file.exists()

    # --- 1. Load data and initialize ---
    seq_tensor, species_names = sp.load_phy_to_tensor(
        str(phy_path),
        drop_conserved_sites=True,
    )
    num_species = seq_tensor.shape[0]
    if num_species < 24:
        raise ValueError("run_qf only supports species count >= 24")
    if num_species >= 65536:
        raise ValueError("Current accelerated path requires species count < 65536")

    if quartet_assembler == "qfm":
        selected_assembler = "qfm-fi"
    else:
        selected_assembler = "tree-qmc"
        selected_tree_qmc_iter_limit = qmc_iter_limit
    print(
        f"[INFO] model_label={model_label}, quartet_assembler={quartet_assembler}, "
        f"selected_assembler={selected_assembler}, num_species={num_species}, "
        f"aggregate_mode={aggregate_mode_key}, "
        f"aggregate_intermediate_flush={aggregate_intermediate_flush}, "
        f"quartformer_top1_only={quartformer_top1_only}, "
        f"cpp_quartet_aggregate={use_cpp_quartet_aggregate}, "
        f"cpp_split_convert={use_cpp_split_convert}, "
        f"quartet_aggregate_shards={quartet_aggregate_shard_count}, "
        f"tree_qmc_iterlimit={selected_tree_qmc_iter_limit if selected_tree_qmc_iter_limit is not None else 'default'}"
    )

    species_name_to_idx = {name: idx for idx, name in enumerate(species_names)}
    taxon_prefix = "t"

    def _idx_to_label(idx: int) -> str:
        return f"{taxon_prefix}{idx}"

    def _label_to_idx(name) -> int | None:
        if name is None:
            return None
        s = str(name).strip()
        if not s:
            return None
        if s.startswith(taxon_prefix):
            tail = s[len(taxon_prefix):]
            if tail.isdigit():
                idx = int(tail)
                if 0 <= idx < num_species:
                    return idx
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < num_species:
                return idx
        return species_name_to_idx.get(s, None)

    def _normalize_rep_idx_list(names) -> list[int]:
        out = []
        seen = set()
        for n in names:
            idx = _label_to_idx(n)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    # Pre-build labels to avoid repeated f-string construction in hot paths.
    taxon_labels = [_idx_to_label(i) for i in range(num_species)]

    # --- 2. Build sampling blocks ---
    if num_species == 24:
        blocks = [list(range(num_species))]
    else:
        so_path = REPO_ROOT / "backend" / "batching_algorithms.cpython-310-x86_64-linux-gnu.so"
        spec = importlib.util.spec_from_file_location("batching_algorithms", so_path)
        batching_algorithms = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"Unable to load batching extension: {so_path}")
        spec.loader.exec_module(batching_algorithms)
        blocks = batching_algorithms.pair_balanced_block_design_v5(
            num_species=num_species, k_param=k_param, batch_size=model_size, seed=42, species_weight=2.0, threads=1
        )

    # --- 3. Batch inference for quartet weights ---
    model_dir = _model_dir(model_size)
    coeff_blocks = torch.load(model_dir / "coeff_blocks.pt", map_location="cpu", weights_only=False).to(device)
    quartet_matrix = torch.load(model_dir / "quartet_matrix.pt", map_location="cpu", weights_only=False).to(device)
    species_enc = torch.load(model_dir / "species_encoding.pt", map_location="cpu", weights_only=False).to(device)

    expected_valid_len = 10626
    expected_padded_len = 10640
    pad_len = expected_padded_len - expected_valid_len
    if species_enc.shape[0] != expected_valid_len:
        raise ValueError(
            f"species_encoding length mismatch: got {species_enc.shape[0]}, expected {expected_valid_len}"
        )
    if quartet_matrix.shape[0] != expected_padded_len:
        raise ValueError(
            f"quartet_matrix length mismatch: got {quartet_matrix.shape[0]}, expected {expected_padded_len}"
        )
    
    resolved_attn_weight = model_dir / "qf1.pt"
    if not resolved_attn_weight.exists():
        raise FileNotFoundError(f"Attention model checkpoint not found: {resolved_attn_weight}")
    attn_model = QuartFormer(species_num=model_size)
    attn_model.load_state_dict(torch.load(resolved_attn_weight, map_location="cpu", weights_only=False))
    attn_model = attn_model.to(device).eval()
    seq_cuda = torch.from_numpy(seq_tensor).to(device)

    local_quartet_template = np.asarray(
        list(itertools.combinations(range(model_size), 4)), dtype=np.int32
    )
    local_quartet_count = local_quartet_template.shape[0]

    agg_quartet_keys = np.empty(0, dtype=np.uint64)
    agg_weight_sums = np.empty((0, 3), dtype=np.float64)
    agg_counts = np.empty(0, dtype=np.int64)

    pending_keys = []
    pending_sums = []
    pending_counts = []
    pending_entries = 0
    flush_invocations = 0
    flush_time_s = 0.0
    stream_split_lines = 0
    # By default, avoid intermediate flushes so CPU aggregation does not interfere
    # with the GPU inference loop. If enabled via config, allow periodic flush.
    if aggregate_intermediate_flush:
        if aggregate_flush_threshold_cfg > 0:
            agg_flush_threshold = aggregate_flush_threshold_cfg
        else:
            agg_flush_threshold = max(2_000_000, local_quartet_count * max(int(infer_batch_size), 1) * 32)
    else:
        agg_flush_threshold = None
    qfm_fast_jar = REPO_ROOT / "quartet_assemble_method" / "qfm_java_fast" / "QFM-FI_unzipped" / "QFM-FI-fast.jar"
    stream_binary_mode = aggregate_mode_key in ("batch_only", "off") and (
        selected_assembler == "tree-qmc"
        or (selected_assembler == "qfm-fi" and qfm_fast_jar.exists())
    )

    shm_dir = Path("/dev/shm")
    if (selected_assembler == "tree-qmc" or stream_binary_mode) and shm_dir.exists() and shm_dir.is_dir():
        result_dir = shm_dir / "publish_code_qf2_result"
    else:
        result_dir = REPO_ROOT / "temp" / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    input_file = result_dir / f"qfm_input_{output_tree_path.stem}.txt"
    input_bin_file = result_dir / f"qfm_input_{output_tree_path.stem}.bin"
    output_file = result_dir / f"qfm_output_{output_tree_path.stem}.txt"
    qmc_support_base_tree_file = result_dir / f"qmc_support_base_{output_tree_path.stem}.nwk"
    qmc_support_annotated_file = result_dir / f"qmc_support_annotated_{output_tree_path.stem}.nwk"
    support_quartet_out = output_tree_path.with_name(f"{output_tree_path.stem}.support_quartets.txt")
    support_tree_out = output_tree_path.with_name(f"{output_tree_path.stem}.support.nwk")
    support_table_out = output_tree_path.with_name(f"{output_tree_path.stem}.support.csv")
    input_write_mode = "qmc" if selected_assembler == "tree-qmc" else "qfm"
    stream_handle = None
    support_handle = None
    if compute_branch_support:
        support_handle = open(support_quartet_out, "w", buffering=16 * 1024 * 1024)
    if aggregate_mode_key in ("batch_only", "off"):
        if stream_binary_mode:
            stream_handle = open(input_bin_file, "wb", buffering=16 * 1024 * 1024)
        else:
            stream_handle = open(input_file, "w", buffering=16 * 1024 * 1024)

    pair1_left = np.array([0, 0, 0], dtype=np.int32)
    pair1_right = np.array([1, 2, 3], dtype=np.int32)
    pair2_left = np.array([2, 1, 1], dtype=np.int32)
    pair2_right = np.array([3, 3, 2], dtype=np.int32)

    def _quartets_weights_to_splits(quartets_np: np.ndarray, weights_int: np.ndarray):
        split_key_chunks = []
        split_weight_chunks = []
        for cls_idx in range(3):
            cls_weights = weights_int[:, cls_idx]
            mask = cls_weights > 0
            if not np.any(mask):
                continue

            q_sel = quartets_np[mask]
            w_sel = cls_weights[mask].astype(np.int64, copy=False)

            p1a = q_sel[:, pair1_left[cls_idx]]
            p1b = q_sel[:, pair1_right[cls_idx]]
            p2a = q_sel[:, pair2_left[cls_idx]]
            p2b = q_sel[:, pair2_right[cls_idx]]

            pair1 = np.stack([np.minimum(p1a, p1b), np.maximum(p1a, p1b)], axis=1)
            pair2 = np.stack([np.minimum(p2a, p2b), np.maximum(p2a, p2b)], axis=1)

            swap = (pair2[:, 0] < pair1[:, 0]) | (
                (pair2[:, 0] == pair1[:, 0]) & (pair2[:, 1] < pair1[:, 1])
            )
            if swap.any():
                pair1_swap = pair1.copy()
                pair1[swap] = pair2[swap]
                pair2[swap] = pair1_swap[swap]

            split_key_chunks.append(_encode_split_keys(pair1, pair2))
            split_weight_chunks.append(w_sel)

        if not split_key_chunks:
            return (
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.int64),
            )

        return (
            np.concatenate(split_key_chunks),
            np.concatenate(split_weight_chunks).astype(np.int64, copy=False),
        )

    def _aggregates_to_splits(keys_np: np.ndarray, sums_np: np.ndarray, counts_np: np.ndarray):
        cpp_result = quartet_aggregate_backend.aggregates_to_splits(
            keys_np,
            sums_np,
            counts_np,
        )
        return cpp_result

    def _append_split_lines(output_handle, split_keys, split_weights, output_mode: str):
        if split_keys.size == 0:
            return
        if output_mode not in ("qmc", "qfm"):
            raise ValueError(f"Unknown output format: {output_mode}")
        write_chunk_size = 500_000
        for start in range(0, split_keys.size, write_chunk_size):
            end = min(start + write_chunk_size, split_keys.size)
            keys_chunk = split_keys[start:end]
            weights_chunk = split_weights[start:end]

            p1a = ((keys_chunk >> 48) & 0xFFFF).astype(np.int32)
            p1b = ((keys_chunk >> 32) & 0xFFFF).astype(np.int32)
            p2a = ((keys_chunk >> 16) & 0xFFFF).astype(np.int32)
            p2b = (keys_chunk & 0xFFFF).astype(np.int32)

            if output_mode == "qmc":
                chunk_text = "".join(
                    f"{taxon_labels[a]},{taxon_labels[b]}|{taxon_labels[c]},{taxon_labels[d]}:{w}\n"
                    for a, b, c, d, w in zip(p1a, p1b, p2a, p2b, weights_chunk)
                )
            else:
                chunk_text = "".join(
                    f"(({taxon_labels[a]},{taxon_labels[b]}),({taxon_labels[c]},{taxon_labels[d]})); {w}\n"
                    for a, b, c, d, w in zip(p1a, p1b, p2a, p2b, weights_chunk)
                )
            output_handle.write(chunk_text)

    def _write_split_records_binary(output_handle, split_keys: np.ndarray, split_weights: np.ndarray):
        if split_keys.size == 0:
            return
        keys = split_keys.astype(np.uint64, copy=False)
        weights = split_weights.astype(np.int64, copy=False)
        write_chunk_size = 2_000_000
        rec_dtype = np.dtype([("k", "<u8"), ("w", "<i8")])
        for start in range(0, keys.size, write_chunk_size):
            end = min(start + write_chunk_size, keys.size)
            rec = np.empty(end - start, dtype=rec_dtype)
            rec["k"] = keys[start:end]
            rec["w"] = weights[start:end]
            output_handle.write(rec.tobytes())

    def _flush_quartet_aggregates():
        nonlocal agg_quartet_keys, agg_weight_sums, agg_counts
        nonlocal pending_keys, pending_sums, pending_counts, pending_entries
        if pending_entries == 0:
            return

        key_parts = []
        sum_parts = []
        count_parts = []
        if agg_quartet_keys.size > 0:
            key_parts.append(agg_quartet_keys)
            sum_parts.append(agg_weight_sums)
            count_parts.append(agg_counts)
        key_parts.extend(pending_keys)
        sum_parts.extend(pending_sums)
        count_parts.extend(pending_counts)

        merged_keys = np.concatenate(key_parts)
        merged_sums = np.concatenate(sum_parts, axis=0)
        merged_counts = np.concatenate(count_parts)

        agg_quartet_keys, agg_weight_sums, agg_counts = _reduce_aggregates(
            merged_keys,
            merged_sums,
            merged_counts,
        )

        pending_keys.clear()
        pending_sums.clear()
        pending_counts.clear()
        pending_entries = 0

    print(f"[INFO] Starting batch inference ({len(blocks)} blocks, BatchSize={infer_batch_size})...")
    
    from tqdm import tqdm
    
    pbar = tqdm(total=len(blocks), desc="[INFO] Batch inference progress")
    for i in range(0, len(blocks), infer_batch_size):
        batch_blocks = blocks[i : i + infer_batch_size]
        actual_bs = len(batch_blocks)
        batch_quartets_np = np.empty((actual_bs, local_quartet_count, 4), dtype=np.int32)

        for batch_idx, block in enumerate(batch_blocks):
            block_sorted = np.asarray(sorted(block), dtype=np.int32)
            if block_sorted.size != model_size:
                raise ValueError(
                    f"Block size mismatch: got {block_sorted.size}, expected fixed size {model_size}"
                )
            batch_quartets_np[batch_idx] = block_sorted[local_quartet_template]

        quartets_flat = batch_quartets_np.reshape(-1, 4)
        idx_cuda = torch.as_tensor(quartets_flat, device=device, dtype=torch.long)

        pattern_batch = pattern_freq_cuda.compute_pattern_frequencies_cuda(seq_cuda, idx_cuda).view(
            actual_bs, local_quartet_count, -1
        )

        species_batch = species_enc.unsqueeze(0).expand(actual_bs, -1, -1)
        input_batch = torch.cat([species_batch, pattern_batch], dim=2)
        if input_batch.size(1) != expected_valid_len:
            raise ValueError(
                f"input_batch length mismatch: got {input_batch.size(1)}, expected {expected_valid_len}"
            )
        input_batch_for_model = torch.cat(
            [input_batch, input_batch.new_zeros((actual_bs, pad_len, input_batch.size(2)))],
            dim=1,
        )
        canonical_input_batch = torch.empty_like(input_batch_for_model)
        canonical_input_batch.copy_(input_batch_for_model)
        input_batch_for_model = canonical_input_batch

        with torch.no_grad():
            logits = attn_model(input_batch_for_model, coeff_blocks, quartet_matrix)

            logits = logits[:, :expected_valid_len, :]
            probs = torch.softmax(logits, dim=-1)
            weights_batch = (probs * 100.0).cpu().numpy()
            if quartformer_top1_only:
                top1_idx = np.argmax(weights_batch, axis=-1, keepdims=True)
                top1_mask = np.zeros_like(weights_batch)
                np.put_along_axis(top1_mask, top1_idx, 1.0, axis=-1)
                weights_batch *= top1_mask

        weights_flat = weights_batch.reshape(-1, 3).astype(np.float64, copy=False)
        if aggregate_mode_key == "full":
            uniq_keys, batch_sums, batch_counts = _aggregate_quartets(quartets_flat, weights_flat)

            pending_keys.append(uniq_keys)
            pending_sums.append(batch_sums)
            pending_counts.append(batch_counts)
            pending_entries += uniq_keys.size
            if agg_flush_threshold is not None and pending_entries >= agg_flush_threshold:
                flush_start = time.perf_counter()
                _flush_quartet_aggregates()
                flush_elapsed = time.perf_counter() - flush_start
                flush_invocations += 1
                flush_time_s += flush_elapsed
                pbar.write(
                    f"[INFO] Intermediate aggregation done #{flush_invocations}: unique_quartets={agg_quartet_keys.size}, "
                    f"flush_time={flush_elapsed:.2f}s"
                )
        elif aggregate_mode_key == "batch_only":
            uniq_keys, batch_sums, batch_counts = _aggregate_quartets(quartets_flat, weights_flat)
            split_keys_batch, split_weights_batch = _aggregates_to_splits(
                uniq_keys,
                batch_sums,
                batch_counts,
            )
            if split_keys_batch.size > 0:
                if stream_binary_mode:
                    _write_split_records_binary(stream_handle, split_keys_batch, split_weights_batch)
                else:
                    _append_split_lines(stream_handle, split_keys_batch, split_weights_batch, output_mode=input_write_mode)
                stream_split_lines += int(split_keys_batch.size)
                if support_handle is not None:
                    _append_split_lines(support_handle, split_keys_batch, split_weights_batch, output_mode="qmc")
        else:
            weights_int = np.rint(weights_flat).astype(np.int32)
            split_keys_batch, split_weights_batch = _quartets_weights_to_splits(quartets_flat, weights_int)
            if split_keys_batch.size > 0:
                if stream_binary_mode:
                    _write_split_records_binary(stream_handle, split_keys_batch, split_weights_batch)
                else:
                    _append_split_lines(stream_handle, split_keys_batch, split_weights_batch, output_mode=input_write_mode)
                stream_split_lines += int(split_keys_batch.size)
                if support_handle is not None:
                    _append_split_lines(support_handle, split_keys_batch, split_weights_batch, output_mode="qmc")
        pbar.update(actual_bs)
    pbar.close()
    # --- 4. Prepare assembly input ---
    if aggregate_mode_key == "full":
        if pending_entries > 0:
            print(
                f"[INFO] Running final aggregation: pending_entries={pending_entries}, "
                f"current_unique_quartets={agg_quartet_keys.size}"
            )
            flush_start = time.perf_counter()
            _flush_quartet_aggregates()
            flush_elapsed = time.perf_counter() - flush_start
            flush_invocations += 1
            flush_time_s += flush_elapsed
            print(
                f"[INFO] Final aggregation done: unique_quartets={agg_quartet_keys.size}, "
                f"flush_time={flush_elapsed:.2f}s"
            )

        if agg_quartet_keys.size == 0:
            print("[ERROR] No quartet weights were generated")
            return ""

        split_start = time.perf_counter()
        split_keys_unique, split_weight_sums = _aggregates_to_splits(
            agg_quartet_keys,
            agg_weight_sums,
            agg_counts,
        )
        split_elapsed = time.perf_counter() - split_start
        if split_keys_unique.size == 0:
            print("[ERROR] Averaged quartet weights are all zero; assembly cannot proceed")
            return ""

        print(
            f"[INFO] Aggregation stats: unique_quartets={agg_quartet_keys.size}, split_lines={split_keys_unique.size}, "
            f"flush_calls={flush_invocations}, flush_total_time={flush_time_s:.2f}s, "
            f"split_convert_time={split_elapsed:.2f}s"
        )

        use_qfm_fast_binary_in_full = selected_assembler == "qfm-fi" and qfm_fast_jar.exists()
        if selected_assembler != "tree-qmc" and not use_qfm_fast_binary_in_full:
            with open(input_file, "w", buffering=16 * 1024 * 1024) as f:
                _append_split_lines(f, split_keys_unique, split_weight_sums, output_mode=input_write_mode)
        if use_qfm_fast_binary_in_full:
            with open(input_bin_file, "wb", buffering=16 * 1024 * 1024) as f:
                _write_split_records_binary(f, split_keys_unique, split_weight_sums)
        if support_handle is not None:
            _append_split_lines(support_handle, split_keys_unique, split_weight_sums, output_mode="qmc")

        print(f"[INFO] Running initial assembler: assembler={selected_assembler}")
        assembler_start = time.perf_counter()
        ok = _run_tree_qmc_from_memory(
            split_keys_unique,
            split_weight_sums,
            output_file,
            qmc_compact_format=True,
        ) if selected_assembler == "tree-qmc" else _run_quartet_assembler(
            input_bin_file if use_qfm_fast_binary_in_full else input_file,
            output_file,
            qmc_compact_format=False,
        )
        print(f"[INFO] Initial assembler finished: assembler={selected_assembler}, elapsed={time.perf_counter() - assembler_start:.2f}s")
    else:
        if stream_handle is not None:
            stream_handle.close()
            stream_handle = None
        if stream_split_lines == 0:
            print("[ERROR] No split records were generated; assembly cannot proceed")
            return ""
        print(
            f"[INFO] Streaming stats: aggregate_mode={aggregate_mode_key}, split_lines={stream_split_lines}, "
            f"binary_io={stream_binary_mode}"
        )
        print(f"[INFO] Running initial assembler: assembler={selected_assembler}")
        assembler_start = time.perf_counter()
        if selected_assembler == "tree-qmc" and stream_binary_mode:
            ok = _run_tree_qmc_from_binary_file(
                input_bin_file,
                output_file,
                qmc_compact_format=True,
            )
        elif selected_assembler == "qfm-fi" and stream_binary_mode:
            ok = _run_quartet_assembler(
                input_bin_file,
                output_file,
                qmc_compact_format=False,
            )
        else:
            ok = _run_quartet_assembler(
                input_file,
                output_file,
                qmc_compact_format=(selected_assembler == "tree-qmc"),
            )
        print(f"[INFO] Initial assembler finished: assembler={selected_assembler}, elapsed={time.perf_counter() - assembler_start:.2f}s")
    if not ok:
        print("[ERROR] Initial assembly failed")
        if last_assembler_error:
            print(f"[ERROR] Assembler details: {last_assembler_error}")
        return ""

    # --- 5. Polytomy resolution (MLP) ---
    print("[INFO] Checking and resolving polytomies...")
    tree = Tree(str(output_file))
    tree.unroot()

    # Load MLP model for polytomy resolution.
    default_mlp_weight = REPO_ROOT / "model" / model_label / "best_mlp_model.pth"
    mlp_weight = default_mlp_weight
    if not mlp_weight.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {mlp_weight}")
    mlp_model = MLP().to(device)
    mlp_model.load_state_dict(torch.load(mlp_weight, map_location=device, weights_only=False))
    mlp_model.eval()

    def _infer_subtree_with_mlp(all_rep_ids, child_rep_ids):
        outgroup = sorted(set(all_rep_ids) - set(child_rep_ids))[0] if len(all_rep_ids) > len(child_rep_ids) else None
        quartets = list(itertools.combinations(all_rep_ids, 4))
        if not quartets:
            t = Tree()
            for sp_idx in child_rep_ids:
                t.add_child(name=_idx_to_label(sp_idx))
            t.unroot()
            return t

        idx_c = torch.tensor(quartets, dtype=torch.long, device=device)
        p_tensor = pattern_freq_cuda.compute_pattern_frequencies_cuda(seq_cuda, idx_c)
        
        with torch.no_grad():
            logits = mlp_model(p_tensor)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()

        quartets_np = np.asarray(quartets, dtype=np.int32)
        weights_int = np.rint(probs * 100.0).astype(np.int32)
        zero_rows = np.max(weights_int, axis=1) < 1
        if np.any(zero_rows):
            zero_argmax = np.argmax(probs[zero_rows], axis=1)
            weights_int[zero_rows] = 0
            weights_int[zero_rows, zero_argmax] = 1

        split_keys_local, split_weights_local = _quartets_weights_to_splits(quartets_np, weights_int)
        if split_keys_local.size == 0:
            t = Tree()
            for sp_idx in child_rep_ids:
                t.add_child(name=_idx_to_label(sp_idx))
            t.unroot()
            return t

        local_uniq_keys, local_inv = np.unique(split_keys_local, return_inverse=True)
        local_sum_weights = np.bincount(
            local_inv,
            weights=split_weights_local.astype(np.float64),
            minlength=local_uniq_keys.size,
        )
        local_sum_weights = np.rint(local_sum_weights).astype(np.int64)

        with open(input_file, "w", buffering=8 * 1024 * 1024) as f:
            _append_split_lines(f, local_uniq_keys, local_sum_weights, output_mode="qfm")
        if support_handle is not None:
            _append_split_lines(support_handle, local_uniq_keys, local_sum_weights, output_mode="qmc")

        _run_quartet_assembler(
            input_file,
            output_file,
            assembler_override="qfm-fi",
            qmc_compact_format=False,
        )
        
        sub_tree = Tree(str(output_file))
        sub_tree.unroot()
        if outgroup is not None:
            out_label = _idx_to_label(outgroup)
            if out_label in sub_tree.get_leaf_names():
                sub_tree.set_outgroup(out_label)
        sub_tree.prune([_idx_to_label(i) for i in child_rep_ids], preserve_branch_length=False)
        return sub_tree

    iteration = 0
    while True:
        polytomies = find_polytomy_representative_leaves(tree)
        if not polytomies:
            break
        iteration += 1
        poly = polytomies[0]
        target_node, all_reps_raw = poly["node"], poly["representative_leaves"]
        num_children = len(target_node.children)
        child_reps_raw = all_reps_raw[:num_children]
        all_rep_ids = _normalize_rep_idx_list(all_reps_raw)
        child_rep_ids = _normalize_rep_idx_list(child_reps_raw)
        if len(all_rep_ids) < 4 or len(child_rep_ids) < 2:
            print(f"[WARN] Invalid representative labels for polytomy; skipping this resolution pass: all={all_reps_raw}, child={child_reps_raw}")
            break
        
        rep_to_clade = {}
        for c in target_node.children:
            parsed_ids = _normalize_rep_idx_list(sorted(c.get_leaf_names()))
            if parsed_ids:
                rep_to_clade[_idx_to_label(parsed_ids[0])] = c
        guide = _infer_subtree_with_mlp(all_rep_ids, child_rep_ids)

        def build_resolved(g_node):
            if g_node.is_leaf():
                rep_idx = _label_to_idx(g_node.name)
                if rep_idx is None:
                    raise ValueError(f"Cannot parse guide-tree leaf label: {g_node.name!r}")
                key = _idx_to_label(rep_idx)
                if key not in rep_to_clade:
                    raise KeyError(f"Guide-tree leaf {key!r} is missing in target polytomy mapping")
                return rep_to_clade[key]
            new_n = Tree()
            for ch in g_node.children:
                new_n.add_child(build_resolved(ch))
            return new_n

        for ch in list(target_node.children):
            ch.detach()
        for guide_child in guide.children:
            target_node.add_child(build_resolved(guide_child))

    # --- 6. Finalize outputs ---
    if support_handle is not None:
        support_handle.close()
        if keep_support_files:
            print(f"[INFO] Support quartet file saved: {support_quartet_out}")
    if compute_branch_support:
        print("[INFO] Running TREE-QMC branch support annotation...")
        tree.write(outfile=str(qmc_support_base_tree_file), format=1)
        ok = _run_qmc_support_annotation(
            quartet_file=support_quartet_out,
            base_tree_file=qmc_support_base_tree_file,
            support_tree_file=qmc_support_annotated_file,
            support_table_file=support_table_out,
        )
        if ok:
            support_tree = Tree(str(qmc_support_annotated_file), format=1, quoted_node_names=True)
            support_tree.unroot()
            for node in support_tree.traverse("postorder"):
                if node.is_leaf():
                    continue
                q1 = _extract_q1_support(node.name)
                if q1 is not None:
                    node.support = q1
                node.name = ""
                node.dist = 0.0
            for leaf in support_tree.iter_leaves():
                idx = _label_to_idx(leaf.name)
                if idx is not None:
                    leaf.name = species_names[idx]
                leaf.dist = 0.0
            support_tree_out.write_text(_to_newick_support_only(support_tree, is_root=True) + "\n")
            print(f"[SUCCESS] Branch-support tree saved: {support_tree_out}")
            if keep_support_files:
                print(f"[SUCCESS] Branch-support table saved: {support_table_out}")
        else:
            print("[WARN] TREE-QMC support annotation failed; skipped support outputs")

    tree.unroot()
    for leaf in tree.iter_leaves():
        idx = _label_to_idx(leaf.name)
        if idx is not None:
            leaf.name = species_names[idx]
    tree.write(outfile=str(output_tree_path), format=1)
    print(f"[SUCCESS] Final binary tree saved: {output_tree_path}")
    input_file.unlink(missing_ok=True)
    input_bin_file.unlink(missing_ok=True)
    output_file.unlink(missing_ok=True)
    qmc_support_base_tree_file.unlink(missing_ok=True)
    qmc_support_annotated_file.unlink(missing_ok=True)
    if compute_branch_support and not keep_support_files:
        support_quartet_out.unlink(missing_ok=True)
        support_table_out.unlink(missing_ok=True)
    if ref_tree_path:
        from utils import compute_tree_difference

        metric_value = compute_tree_difference(
            output_tree_path,
            ref_tree_path,
            mode=metric,
            max_quartets=40000,
        )
        metric_name = "RF distance" if metric == "rf" else "Quartet concordance"
        print(f"[EVAL] {metric_name}: {metric_value:.6f}")
        return str(output_tree_path), metric_value
    return str(output_tree_path)


def _normalize_output_path(output_tree_path, phy_path):
    output_tree_path = Path(output_tree_path)
    if output_tree_path.exists() and output_tree_path.is_dir():
        output_tree_path = output_tree_path / f"{phy_path.stem}.qf2.nwk"
    elif output_tree_path.suffix == "":
        output_tree_path.mkdir(parents=True, exist_ok=True)
        output_tree_path = output_tree_path / f"{phy_path.stem}.qf2.nwk"
    else:
        output_tree_path.parent.mkdir(parents=True, exist_ok=True)
    return output_tree_path


def _extract_q1_support(node_name: str | None) -> float | None:
    if not node_name:
        return None
    m = re.search(r"(?:q1=|_q1_)([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", node_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _to_newick_support_only(node, is_root: bool = False) -> str:
    if node.is_leaf():
        return node.name
    children = ",".join(_to_newick_support_only(ch, False) for ch in node.children)
    if is_root:
        return f"({children});"
    support = getattr(node, "support", None)
    if support is None:
        return f"({children})"
    return f"({children}){float(support):.6f}"


def _render_tree_figure(tree_path: Path, show_support: bool = False) -> None:
    if not tree_path.exists():
        print(f"[WARN] Tree file does not exist; skipping render: {tree_path}")
        return
    fig_path = tree_path.with_suffix(".png")
    try:
        draw_newick_tree_figure(
            tree_path=tree_path,
            out_path=fig_path,
            title=tree_path.stem,
            show_support=show_support,
        )
        print(f"[SUCCESS] Tree figure saved: {fig_path}")
    except Exception as exc:
        print(f"[WARN] Tree rendering failed: {exc}")


def main(argv=None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON/JSONC")
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = Path(pre_args.config)
    try:
        full_cfg = _load_jsonc_config(config_path, section=None)
    except Exception:
        full_cfg = {}

    def _pick_section_dict(cfg: dict, *keys: str) -> dict:
        for k in keys:
            v = cfg.get(k)
            if isinstance(v, dict):
                return v
        return {}

    # Config sections:
    # - basic: common user-facing parameters (usually passed via CLI)
    # - advanced: advanced/perf/algorithm knobs (usually configured via JSONC)
    # Backward compatibility:
    # - cli -> basic
    # - runtime -> advanced
    # - root-level keys still work for both if sections are missing
    basic_cfg: dict = {}
    advanced_cfg: dict = {}
    if isinstance(full_cfg, dict):
        basic_cfg = _pick_section_dict(full_cfg, "basic", "cli")
        advanced_cfg = _pick_section_dict(full_cfg, "advanced", "runtime")
        if not basic_cfg:
            basic_cfg = full_cfg
        if not advanced_cfg:
            advanced_cfg = full_cfg

    normalized_basic_cfg: dict[str, object] = {}
    for k, v in basic_cfg.items():
        if not isinstance(k, str):
            continue
        normalized_basic_cfg[k.strip().lower().replace("-", "_")] = v
    normalized_advanced_cfg: dict[str, object] = {}
    for k, v in advanced_cfg.items():
        if not isinstance(k, str):
            continue
        normalized_advanced_cfg[k.strip().lower().replace("-", "_")] = v

    parser = argparse.ArgumentParser(
        description=(
            "QuartFormer inference driver. Parameters can be provided via CLI and/or a JSON/JSONC config file.\n"
            "Precedence: basic params use CLI > config > defaults; advanced params use config > CLI > defaults."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON/JSONC")
    parser.add_argument("--phy", required=False, help="Input alignment file path (PHY format)")
    parser.add_argument("--out", default="output_tree.nwk", help="Output tree path")
    parser.add_argument(
        "--task-type",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Model family",
    )
    parser.add_argument(
        "--compute-branch-support",
        action="store_true",
        help="Compute and write a support-annotated tree via TREE-QMC support-only mode.",
    )
    parser.add_argument(
        "--plot-tree",
        action="store_true",
        help="Render output tree(s) to PNG right after writing NWK files.",
    )
    parser.add_argument("--ref-tree", default="", help="Optional reference tree path")
    parser.add_argument("--metric", choices=["rf", "quartet"], default="rf")
    parser.add_argument(
        "--keep-support-files",
        action="store_true",
        help="Keep support artifacts (.support.csv / .support_quartets.txt).",
    )
    parser.add_argument("--infer-batch-size", type=int, default=32)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)

    def _cli_has_flag(*flags: str) -> bool:
        return any(flag in raw_argv for flag in flags)

    def _basic_config_get(*keys: str):
        for key in keys:
            if key in normalized_basic_cfg:
                return normalized_basic_cfg[key]
        return None

    def _advanced_config_get(*keys: str):
        for key in keys:
            if key in normalized_advanced_cfg:
                return normalized_advanced_cfg[key]
        return None

    def _normalize_choice(value, name: str, allowed: set[str]) -> str:
        v = str(value).strip()
        if v not in allowed:
            raise SystemExit(f"Invalid config value for {name}: {v}. Allowed: {sorted(allowed)}")
        return v

    # Basic parameters: CLI > config > defaults.
    # Only fill from config when the corresponding CLI flag is absent.
    cfg_val = _basic_config_get("phy")
    if (not _cli_has_flag("--phy")) and cfg_val is not None and str(cfg_val).strip():
        args.phy = str(cfg_val)
    cfg_val = _basic_config_get("out")
    if (not _cli_has_flag("--out")) and cfg_val is not None and str(cfg_val).strip():
        args.out = str(cfg_val)
    cfg_val = _basic_config_get("task_type", "task-type")
    if (not _cli_has_flag("--task-type")) and cfg_val is not None:
        args.task_type = _normalize_choice(cfg_val, "task_type", {"homogeneous", "heterogeneous"})
    cfg_val = _basic_config_get("compute_branch_support", "compute-branch-support")
    if (not _cli_has_flag("--compute-branch-support")) and cfg_val is not None:
        args.compute_branch_support = _parse_bool_config(cfg_val, "compute_branch_support")
    cfg_val = _basic_config_get("plot_tree", "plot-tree")
    if (not _cli_has_flag("--plot-tree")) and cfg_val is not None:
        args.plot_tree = _parse_bool_config(cfg_val, "plot_tree")
    cfg_val = _basic_config_get("ref_tree", "ref-tree")
    if (not _cli_has_flag("--ref-tree")) and cfg_val is not None:
        args.ref_tree = str(cfg_val)
    cfg_val = _basic_config_get("metric")
    if (not _cli_has_flag("--metric")) and cfg_val is not None:
        args.metric = _normalize_choice(cfg_val, "metric", {"rf", "quartet"})

    # Advanced parameters: config > CLI > defaults.
    cfg_val = _advanced_config_get("keep_support_files", "keep-support-files")
    if cfg_val is not None:
        args.keep_support_files = _parse_bool_config(cfg_val, "keep_support_files")
    cfg_val = _advanced_config_get("infer_batch_size", "infer-batch-size")
    if cfg_val is not None:
        args.infer_batch_size = int(cfg_val)

    if args.phy is None or not str(args.phy).strip():
        raise SystemExit("Missing required parameter: phy (provide --phy or set 'phy' in config)")
    if args.infer_batch_size <= 0:
        raise SystemExit("infer_batch_size must be > 0")

    os.environ["QF_INFER_CONFIG"] = args.config

    ref_tree_path = args.ref_tree if args.ref_tree else None
    result = run_qf(
        phy_path=args.phy,
        output_tree_path=args.out,
        task_type=args.task_type,
        infer_batch_size=args.infer_batch_size,
        ref_tree_path=ref_tree_path,
        metric=args.metric,
        compute_branch_support=args.compute_branch_support,
        keep_support_files=args.keep_support_files,
    )
    if ref_tree_path:
        tree_path, metric_value = result
        metric_name = "RF distance" if args.metric == "rf" else "Quartet concordance"
        print(f"\n[FINAL] Inferred tree: {tree_path}")
        print(f"[FINAL] {metric_name}: {metric_value:.6f}")
    else:
        tree_path = result
        print(f"\n[FINAL] Inferred tree: {tree_path}")
    if args.plot_tree:
        inferred_tree_path = Path(tree_path)
        _render_tree_figure(inferred_tree_path, show_support=False)
        if args.compute_branch_support:
            support_tree_path = inferred_tree_path.with_name(f"{inferred_tree_path.stem}.support.nwk")
            _render_tree_figure(support_tree_path, show_support=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
