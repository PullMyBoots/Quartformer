import argparse
import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ete3 import Tree

REPO_ROOT = Path(__file__).resolve().parent
SERVER_BIN_DIR = REPO_ROOT / "server_bin"
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import QuartFormer, MLP
from utils import find_polytomy_representative_leaves


def _load_extension(module_name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_extension_by_stem(module_name: str, directory: Path, stem: str):
    matches = sorted(directory.glob(f"{stem}*.so"))
    if not matches:
        raise FileNotFoundError(f"未找到扩展模块: {directory / (stem + '*.so')}")
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

def run_qf(
    phy_path,
    output_tree_path,
    task_type="homogeneous",
    k_param=3.0,
    cleanup_temp_files: bool = True,
    run_mode: str = "regular",
    infer_batch_size=32,
    attn_weight_path=None,
    mlp_weight_path=None,
    ref_tree_path=None,
    metric: str = "rf",
    compute_branch_support: bool = False,
    enable_confidence_amplitude: float = 0.0,
):
    """
    独立脚本增强版 run_QF_framework:
    - 仅支持物种数 >= 24。
    - 引入推断 Batching。
    - 对重复四元组权重进行均值化处理。
    - 新增：自动化多分叉修复流程（使用 MLP）。
    - 模型类型（gene / multilocus）由 attn_weight_path 或 mlp_weight_path 自动推断。
    - attn_weight_path: 注意力模型权重路径。
    - mlp_weight_path: MLP 权重路径；若未提供，则根据推断出的模型类型选择默认路径。
    - run_mode:
        regular: 物种数 <=96 使用 QFM-FI；>96 使用 TREE-QMC
        fast: 强制使用 TREE-QMC 组装
        very_fast: 强制使用 TREE-QMC 组装，并设置 --iterlimit 3
        extra_fast: 强制使用 TREE-QMC 组装，并设置 --iterlimit 1
        slow: 强制使用 QFM-FI 组装
    - 四元组预测权重统一保留 3 个拓扑（不使用 top1 裁剪）。
    - MLP 多分叉修复阶段固定使用 QFM-FI。
    - 若 compute_branch_support=True：保留 QuartFormer+MLP 两阶段加权四元组，并用 TREE-QMC --supportonly
      对最终完全二叉树计算支持度。
    """
    phy_path = Path(phy_path)
    output_tree_path = _normalize_output_path(output_tree_path, phy_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = 24

    def _infer_model_label(default_task_type: str, *weight_paths) -> str:
        families = []
        for raw_path in weight_paths:
            if not raw_path:
                continue
            parts = {part.lower() for part in Path(raw_path).expanduser().parts}
            if "gene" in parts:
                families.append("gene")
            if "multilocus" in parts:
                families.append("multilocus")
            if "homogeneous" in parts:
                families.append("homogeneous")
            if "heterogeneous" in parts:
                families.append("heterogeneous")
        families = sorted(set(families))
        if not families:
            return default_task_type
        if len(families) > 1:
            raise ValueError(f"attn_weight_path 与 mlp_weight_path 指向了不同模型类型: {families}")
        return families[0]

    model_label = _infer_model_label(task_type, attn_weight_path, mlp_weight_path)

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

    def _apply_confidence_amplitude(logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        if enable_confidence_amplitude <= 0:
            return probs
        top2_logits = torch.topk(logits, k=2, dim=-1).values
        margin = top2_logits[..., 0] - top2_logits[..., 1]
        amplitude = torch.clamp(1.0 + enable_confidence_amplitude * margin, min=1.0, max=3.0)
        return probs * amplitude.unsqueeze(-1)

    selected_assembler = None
    selected_tree_qmc_iter_limit = None
    run_mode_key = run_mode.strip().lower().replace("_", "-")
    if run_mode_key not in ("fast", "very-fast", "extra-fast", "regular", "slow"):
        raise ValueError("run_mode 仅支持 fast、very_fast、extra_fast、regular、slow")
    if enable_confidence_amplitude < 0:
        raise ValueError("enable_confidence_amplitude 必须 >= 0")

    def _run_quartet_assembler(
        input_file: Path,
        output_file: Path,
        assembler_override: str | None = None,
        qmc_compact_format: bool = False,
    ) -> bool:
        assembler_to_use = assembler_override or selected_assembler
        if assembler_to_use is None:
            raise RuntimeError("内部错误：assembler 未初始化")
        if assembler_to_use == "qfm-fi":
            qfm_jar = REPO_ROOT / "quartet_assemble_method" / "qfm_java" / "QFM-FI_unzipped" / "QFM-FI.jar"
            if not qfm_jar.exists():
                raise FileNotFoundError(f"未找到 QFM-FI jar: {qfm_jar}")
            result = subprocess.run(
                ["java", "-jar", str(qfm_jar), str(input_file), str(output_file), "4"],
                check=False, capture_output=True
            )
            return result.returncode == 0 and output_file.exists()
        
        # Default to TREE-QMC / QMC
        tree_qmc_bin_path = REPO_ROOT / "quartet_assemble_method" / "TREE-QMC" / "build" / "tree-qmc"
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("未找到 tree-qmc 可执行文件")
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.returncode == 0 and output_file.exists()

    def _run_qmc_support_annotation(
        quartet_file: Path,
        base_tree_file: Path,
        support_tree_file: Path,
        support_table_file: Path,
    ) -> bool:
        tree_qmc_bin_path = REPO_ROOT / "quartet_assemble_method" / "TREE-QMC" / "build" / "tree-qmc"
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("未找到 tree-qmc 可执行文件")
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

    # --- 1. 加载数据与初始化 ---
    seq_tensor, species_names = sp.load_phy_to_tensor(
        str(phy_path),
        drop_conserved_sites=True,
    )
    num_species = seq_tensor.shape[0]
    if num_species < 24:
        raise ValueError("run_qf 仅支持物种数 >= 24")
    if num_species >= 65536:
        raise ValueError("当前加速实现要求物种数 < 65536")

    if run_mode_key == "fast":
        selected_assembler = "tree-qmc"
    elif run_mode_key == "very-fast":
        selected_assembler = "tree-qmc"
        selected_tree_qmc_iter_limit = 3
    elif run_mode_key == "extra-fast":
        selected_assembler = "tree-qmc"
        selected_tree_qmc_iter_limit = 1
    elif run_mode_key == "slow":
        selected_assembler = "qfm-fi"
    else:
        selected_assembler = "qfm-fi" if num_species <= 96 else "tree-qmc"
    print(
        f"[INFO] model_label={model_label}, run_mode={run_mode_key}, "
        f"selected_assembler={selected_assembler}, num_species={num_species}, "
        f"confidence_amplitude={enable_confidence_amplitude}, "
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

    # 预生成标签，避免热点路径反复 f-string 构造
    taxon_labels = [_idx_to_label(i) for i in range(num_species)]

    # --- 2. 确定采样组合 (Blocks) ---
    if num_species == 24:
        blocks = [list(range(num_species))]
    else:
        so_path = REPO_ROOT / "cpp_source" / "batching_algorithms.cpython-310-x86_64-linux-gnu.so"
        spec = importlib.util.spec_from_file_location("batching_algorithms", so_path)
        batching_algorithms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batching_algorithms)
        blocks = batching_algorithms.pair_balanced_block_design_v5(
            num_species=num_species, k_param=k_param, batch_size=model_size, seed=42, species_weight=2.0, threads=1
        )

    # --- 3. 批量推断四元组权重 ---
    model_dir = _model_dir(model_size)
    coeff_blocks = torch.load(model_dir / "coeff_blocks.pt", map_location="cpu", weights_only=False).to(device)
    quartet_matrix = torch.load(model_dir / "quartet_matrix.pt", map_location="cpu", weights_only=False).to(device)
    species_enc = torch.load(model_dir / "species_encoding.pt", map_location="cpu", weights_only=False).to(device)

    expected_valid_len = 10626
    expected_padded_len = 10640
    pad_len = expected_padded_len - expected_valid_len
    if species_enc.shape[0] != expected_valid_len:
        raise ValueError(
            f"species_encoding 长度异常: {species_enc.shape[0]}，预期 {expected_valid_len}"
        )
    if quartet_matrix.shape[0] != expected_padded_len:
        raise ValueError(
            f"quartet_matrix 长度异常: {quartet_matrix.shape[0]}，预期 {expected_padded_len}"
        )
    
    resolved_attn_weight = Path(attn_weight_path).expanduser() if attn_weight_path else (model_dir / "qf1.pt")
    if not resolved_attn_weight.exists():
        raise FileNotFoundError(f"未找到注意力模型权重: {resolved_attn_weight}")
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
    shm_dir = Path("/dev/shm")
    if selected_assembler == "tree-qmc" and shm_dir.exists() and shm_dir.is_dir():
        result_dir = shm_dir / "publish_code_qf2_result"
    else:
        result_dir = REPO_ROOT / "temp" / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    input_file = result_dir / f"qfm_input_{output_tree_path.stem}.txt"
    output_file = result_dir / f"qfm_output_{output_tree_path.stem}.txt"
    qmc_support_base_tree_file = result_dir / f"qmc_support_base_{output_tree_path.stem}.nwk"
    qmc_support_annotated_file = result_dir / f"qmc_support_annotated_{output_tree_path.stem}.nwk"
    support_quartet_out = output_tree_path.with_name(f"{output_tree_path.stem}.support_quartets.txt")
    support_tree_out = output_tree_path.with_name(f"{output_tree_path.stem}.support.nwk")
    support_table_out = output_tree_path.with_name(f"{output_tree_path.stem}.support.csv")
    input_write_mode = "qmc" if selected_assembler == "tree-qmc" else "qfm"
    support_handle = None
    if compute_branch_support:
        support_handle = open(support_quartet_out, "w", buffering=16 * 1024 * 1024)

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

    def _append_split_lines(output_handle, split_keys, split_weights, output_mode: str):
        if split_keys.size == 0:
            return
        if output_mode not in ("qmc", "qfm"):
            raise ValueError(f"未知输出格式: {output_mode}")
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

        uniq_keys, inv = np.unique(merged_keys, return_inverse=True)
        reduced_sums = np.zeros((uniq_keys.size, 3), dtype=np.float64)
        for cls_idx in range(3):
            reduced_sums[:, cls_idx] = np.bincount(
                inv, weights=merged_sums[:, cls_idx], minlength=uniq_keys.size
            )
        reduced_counts = np.bincount(
            inv, weights=merged_counts, minlength=uniq_keys.size
        ).astype(np.int64)

        agg_quartet_keys = uniq_keys
        agg_weight_sums = reduced_sums
        agg_counts = reduced_counts

        pending_keys.clear()
        pending_sums.clear()
        pending_counts.clear()
        pending_entries = 0

    print(f"[INFO] 开始批量推断 ({len(blocks)} blocks, BatchSize={infer_batch_size})...")
    
    from tqdm import tqdm
    
    pbar = tqdm(total=len(blocks), desc="[INFO] 批量推断进度")
    for i in range(0, len(blocks), infer_batch_size):
        batch_blocks = blocks[i : i + infer_batch_size]
        actual_bs = len(batch_blocks)
        batch_quartets_np = np.empty((actual_bs, local_quartet_count, 4), dtype=np.int32)

        for batch_idx, block in enumerate(batch_blocks):
            block_sorted = np.asarray(sorted(block), dtype=np.int32)
            if block_sorted.size != model_size:
                raise ValueError(
                    f"block 大小异常: {block_sorted.size}，预期固定为 {model_size}"
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
                f"input_batch 长度异常: {input_batch.size(1)}，预期 {expected_valid_len}"
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
            probs = _apply_confidence_amplitude(logits, probs)
            weights_batch = (probs * 100.0).cpu().numpy()

        weights_flat = weights_batch.reshape(-1, 3).astype(np.float64, copy=False)
        quartet_keys_flat = _encode_quartet_keys(quartets_flat)

        uniq_keys, inv = np.unique(quartet_keys_flat, return_inverse=True)
        batch_sums = np.zeros((uniq_keys.size, 3), dtype=np.float64)
        for cls_idx in range(3):
            batch_sums[:, cls_idx] = np.bincount(
                inv, weights=weights_flat[:, cls_idx], minlength=uniq_keys.size
            )
        batch_counts = np.bincount(inv, minlength=uniq_keys.size).astype(np.int64)

        pending_keys.append(uniq_keys)
        pending_sums.append(batch_sums)
        pending_counts.append(batch_counts)
        pending_entries += uniq_keys.size
        pbar.update(actual_bs)
    pbar.close()
    _flush_quartet_aggregates()

    # --- 4. 均值化权重并组装初始树 ---
    if agg_quartet_keys.size == 0:
        print("[ERROR] 未生成任何四元组权重")
        return ""

    avg_weights_int = np.rint(
        agg_weight_sums / agg_counts[:, None]
    ).astype(np.int32)
    quartets_unique = _decode_quartet_keys(agg_quartet_keys)

    split_keys_unique, split_weight_sums = _quartets_weights_to_splits(quartets_unique, avg_weights_int)
    if split_keys_unique.size == 0:
        print("[ERROR] 四元组均值权重全为 0，无法组装")
        return ""

    print(f"[INFO] 聚合统计: unique_quartets={agg_quartet_keys.size}, split_lines={split_keys_unique.size}")

    with open(input_file, "w", buffering=16 * 1024 * 1024) as f:
        _append_split_lines(f, split_keys_unique, split_weight_sums, output_mode=input_write_mode)
    if support_handle is not None:
        _append_split_lines(support_handle, split_keys_unique, split_weight_sums, output_mode="qmc")

    if not _run_quartet_assembler(
        input_file,
        output_file,
        qmc_compact_format=(selected_assembler == "tree-qmc"),
    ):
        print("[ERROR] 初始组装失败")
        return ""

    # --- 5. 多分叉修复流程 (MLP) ---
    print("[INFO] 正在检查并修复多分叉节点...")
    tree = Tree(str(output_file))
    tree.unroot()

    # 加载 MLP 模型用于修复
    default_mlp_weight = REPO_ROOT / "model" / model_label / "best_mlp_model.pth"
    mlp_weight = Path(mlp_weight_path).expanduser() if mlp_weight_path else default_mlp_weight
    if not mlp_weight.exists():
        raise FileNotFoundError(f"未找到 MLP 权重: {mlp_weight}")
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
            probs = _apply_confidence_amplitude(logits, probs).cpu().numpy()

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
        if not polytomies: break
        iteration += 1
        poly = polytomies[0]
        target_node, all_reps_raw = poly["node"], poly["representative_leaves"]
        num_children = len(target_node.children)
        child_reps_raw = all_reps_raw[:num_children]
        all_rep_ids = _normalize_rep_idx_list(all_reps_raw)
        child_rep_ids = _normalize_rep_idx_list(child_reps_raw)
        if len(all_rep_ids) < 4 or len(child_rep_ids) < 2:
            print(f"[WARN] 多分叉代表叶子存在非法标签，跳过本轮修复: all={all_reps_raw}, child={child_reps_raw}")
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
                    raise ValueError(f"无法解析引导树叶子标签: {g_node.name!r}")
                key = _idx_to_label(rep_idx)
                if key not in rep_to_clade:
                    raise KeyError(f"引导树叶子 {key!r} 未在目标多分叉子树映射中找到")
                return rep_to_clade[key]
            new_n = Tree()
            for ch in g_node.children: new_n.add_child(build_resolved(ch))
            return new_n

        for ch in list(target_node.children):
            ch.detach()
        for guide_child in guide.children:
            target_node.add_child(build_resolved(guide_child))

    # --- 6. 完成 ---
    if support_handle is not None:
        support_handle.close()
        print(f"[INFO] 支持度四元组文件已保存至 {support_quartet_out}")
    if compute_branch_support:
        print("[INFO] 正在使用 TREE-QMC 输出分枝支持度...")
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
            for leaf in support_tree.iter_leaves():
                idx = _label_to_idx(leaf.name)
                if idx is not None:
                    leaf.name = species_names[idx]
            support_tree.write(outfile=str(support_tree_out), format=1)
            print(f"[SUCCESS] 分枝支持度树已保存至 {support_tree_out}")
            print(f"[SUCCESS] 分枝支持度表已保存至 {support_table_out}")
        else:
            print("[WARN] TREE-QMC 支持度输出失败，已跳过支持度文件")

    tree.unroot()
    for leaf in tree.iter_leaves():
        idx = _label_to_idx(leaf.name)
        if idx is not None:
            leaf.name = species_names[idx]
    tree.write(outfile=str(output_tree_path), format=1)
    print(f"[SUCCESS] 最终二叉树已保存至 {output_tree_path}")
    if cleanup_temp_files:
        input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)
        qmc_support_base_tree_file.unlink(missing_ok=True)
        qmc_support_annotated_file.unlink(missing_ok=True)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run server_bin dense pipeline in current repository.")
    parser.add_argument("--phy", required=True, help="Input alignment file path (PHY format)")
    parser.add_argument("--out", help="Output tree path", default="output_tree.nwk")
    parser.add_argument(
        "--task-type",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Default model family when weight paths do not imply gene/multilocus",
    )
    parser.add_argument("--k-param", type=float, default=3.0, help="Sampling parameter")
    parser.add_argument(
        "--run-mode",
        choices=["fast", "very_fast", "extra_fast", "regular", "slow"],
        default="regular",
    )
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--attn-weight-path", default=None, help="Optional attention model weight path")
    parser.add_argument("--mlp-weight-path", default=None, help="Optional MLP weight path")
    parser.add_argument("--ref-tree", default="", help="Optional reference tree path")
    parser.add_argument("--metric", choices=["rf", "quartet"], default="rf")
    parser.add_argument("--no-cleanup-temp-files", action="store_true")
    parser.add_argument(
        "--compute-branch-support",
        action="store_true",
        help="Retain weighted quartets from QuartFormer+MLP phases and compute support for final binary tree with TREE-QMC",
    )
    parser.add_argument(
        "--enable-confidence-amplitude",
        type=float,
        nargs="?",
        const=1.0,
        default=0.0,
        help="Confidence amplification strength (>=0). 0 disables; if flag is provided without value, defaults to 1.0",
    )
    args = parser.parse_args(argv)

    ref_tree_path = args.ref_tree if args.ref_tree else None
    result = run_qf(
        phy_path=args.phy,
        output_tree_path=args.out,
        task_type=args.task_type,
        k_param=args.k_param,
        cleanup_temp_files=not args.no_cleanup_temp_files,
        run_mode=args.run_mode,
        infer_batch_size=args.infer_batch_size,
        attn_weight_path=args.attn_weight_path,
        mlp_weight_path=args.mlp_weight_path,
        ref_tree_path=ref_tree_path,
        metric=args.metric,
        compute_branch_support=args.compute_branch_support,
        enable_confidence_amplitude=args.enable_confidence_amplitude,
    )
    if ref_tree_path:
        tree_path, metric_value = result
        metric_name = "RF distance" if args.metric == "rf" else "Quartet concordance"
        print(f"\n[FINAL] Inferred tree: {tree_path}")
        print(f"[FINAL] {metric_name}: {metric_value:.6f}")
    else:
        print(f"\n[FINAL] Inferred tree: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
