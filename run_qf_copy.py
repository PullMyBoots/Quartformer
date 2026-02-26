"""
QuartFormer: Deep learning-based phylogenetic tree inference tool.

This module provides the main inference pipeline for building phylogenetic trees
using deep learning models (QuartFormer and MLP) with quartet-based assembly.
"""

import torch
import subprocess
import argparse
import itertools
import importlib.util
import os
import tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from ete3 import Tree

from model import QuartFormer, MLP
from utils import _canonical_split_from_class, find_polytomy_representative_leaves


###################################################################################################
# C++ Extension Loading
###################################################################################################

def _load_cpp_extensions():
    """Load C++ extension modules for sequence processing and CUDA acceleration."""
    # Load sequence processor
    spec = importlib.util.spec_from_file_location(
        "sequence_processor",
        "cpp_source/sequence_processor.cpython-310-x86_64-linux-gnu.so"
    )
    sequence_processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence_processor)

    # Load CUDA pattern frequency processor
    spec = importlib.util.spec_from_file_location(
        "pattern_freq_cuda",
        "cpp_source/cuda13_pattern_freq/pattern_freq_cuda.cpython-310-x86_64-linux-gnu.so"
    )
    pattern_freq_cuda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pattern_freq_cuda)

    return sequence_processor, pattern_freq_cuda


# Load C++ extensions at module level
sp, pattern_freq_cuda = _load_cpp_extensions()


###################################################################################################
# Main Inference Pipeline
###################################################################################################

def run_qf(
    phy_path,
    output_tree_path,
    task_type="homogeneous",
    k_param=3.0,
    cleanup_temp_files: bool = True,
    run_mode: str = "regular",
    infer_batch_size=32,
    ref_tree_path: str = None,
    metric: str = "rf",
):
    """
    Run QuartFormer phylogenetic tree inference.

    Features:
        - Supports species count >= 24
        - Batched inference for efficiency
        - Averaged weight handling for duplicate quartets
        - Automated polytomy resolution using MLP

    Args:
        phy_path: Input alignment file path (PHY format)
        output_tree_path: Output tree path (file or directory)
        task_type: Task type
            - "homogeneous": Homogeneous mode, assumes all partitions consistent (default)
            - "heterogeneous": Heterogeneous mode, accounts for gene flow/conflicts
        k_param: Block design parameter (default: 3.0)
        cleanup_temp_files: Whether to clean up temporary files (default: True)
        run_mode: Assembly algorithm
            - "fast": Always use TREE-QMC
            - "regular": Use QFM-FI for <=96 species, TREE-QMC for >96
        infer_batch_size: Batch size for inference (default: 32)
        ref_tree_path: Reference tree path for evaluation (optional)
        metric: Evaluation metric when ref_tree_path is provided
            - "rf": Robinson-Foulds distance (0-1, lower is better)
            - "quartet": Quartet concordance (0-1, higher is better)

    Returns:
        str: Path to output tree file, or tuple of (tree_path, metric_value) if ref_tree_path provided
    """
    # ================================================================================
    # Initialization
    # ================================================================================
    phy_path = Path(phy_path)
    output_tree_path = _normalize_output_path(output_tree_path, phy_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_size = 24

    # Helper functions
    def _model_dir(species_for_model: int) -> Path:
        """Get model directory based on task type."""
        base_dir = Path("model/homogeneous") if task_type == "homogeneous" else Path("model/heterogeneous")
        return base_dir / str(species_for_model)

    def _encode_quartet_keys(quartets_np: np.ndarray) -> np.ndarray:
        """Encode quartet indices to single uint64 keys."""
        q = quartets_np.astype(np.uint64, copy=False)
        return (q[:, 0] << 48) | (q[:, 1] << 32) | (q[:, 2] << 16) | q[:, 3]

    def _decode_quartet_keys(keys_np: np.ndarray) -> np.ndarray:
        """Decode uint64 keys back to quartet indices."""
        quartets_np = np.empty((keys_np.shape[0], 4), dtype=np.int32)
        quartets_np[:, 0] = ((keys_np >> 48) & 0xFFFF).astype(np.int32)
        quartets_np[:, 1] = ((keys_np >> 32) & 0xFFFF).astype(np.int32)
        quartets_np[:, 2] = ((keys_np >> 16) & 0xFFFF).astype(np.int32)
        quartets_np[:, 3] = (keys_np & 0xFFFF).astype(np.int32)
        return quartets_np

    def _encode_split_keys(pair1: np.ndarray, pair2: np.ndarray) -> np.ndarray:
        """Encode split pairs to single uint64 keys."""
        p1 = pair1.astype(np.uint64, copy=False)
        p2 = pair2.astype(np.uint64, copy=False)
        return (p1[:, 0] << 48) | (p1[:, 1] << 32) | (p2[:, 0] << 16) | p2[:, 1]

    # ================================================================================
    # Load Input Data
    # ================================================================================
    seq_tensor, species_names = sp.load_phy_to_tensor(str(phy_path))
    num_species = seq_tensor.shape[0]

    if num_species < 24:
        raise ValueError("run_qf requires species count >= 24")
    if num_species >= 65536:
        raise ValueError("Current implementation requires species count < 65536")

    # Select quartet assembler based on run mode
    run_mode_key = run_mode.strip().lower().replace("_", "-")
    if run_mode_key not in ("fast", "regular"):
        raise ValueError("run_mode must be 'fast' or 'regular'")

    selected_assembler = "tree-qmc" if run_mode_key == "fast" else ("qfm-fi" if num_species <= 96 else "tree-qmc")
    print(f"[INFO] run_mode={run_mode_key}, assembler={selected_assembler}, species={num_species}")

    # Taxon label conversion
    taxon_prefix = "t"
    species_name_to_idx = {name: idx for idx, name in enumerate(species_names)}
    taxon_labels = [f"{taxon_prefix}{i}" for i in range(num_species)]

    def _label_to_idx(name) -> int | None:
        """Convert taxon label back to index."""
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
        """Normalize and deduplicate list of representative indices."""
        out = []
        seen = set()
        for n in names:
            idx = _label_to_idx(n)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    # ================================================================================
    # Generate Sampling Blocks
    # ================================================================================
    if num_species == 24:
        blocks = [list(range(num_species))]
    else:
        batching_so = "cpp_source/batching_algorithms.cpython-310-x86_64-linux-gnu.so"
        spec = importlib.util.spec_from_file_location("batching_algorithms", batching_so)
        batching_algorithms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batching_algorithms)
        blocks = batching_algorithms.pair_balanced_block_design_v5(
            num_species=num_species,
            k_param=k_param,
            batch_size=model_size,
            seed=42,
            species_weight=2.0,
            threads=1
        )

    # ================================================================================
    # Batch Inference of Quartet Weights
    # ================================================================================
    # Load model
    model_dir = _model_dir(model_size)
    coeff_blocks = torch.load(model_dir / "coeff_blocks.pt", map_location="cpu", weights_only=False).to(device)
    quartet_matrix = torch.load(model_dir / "quartet_matrix.pt", map_location="cpu", weights_only=False).to(device)
    species_enc = torch.load(model_dir / "species_encoding.pt", map_location="cpu", weights_only=False).to(device)

    attn_model = QuartFormer(species_num=model_size)
    attn_model.load_state_dict(torch.load(model_dir / "qf1.pt", map_location="cpu", weights_only=False))
    attn_model = attn_model.to(device).eval()
    seq_cuda = torch.from_numpy(seq_tensor).to(device)

    local_quartet_template = np.asarray(
        list(itertools.combinations(range(model_size), 4)), dtype=np.int32
    )

    # Aggregation buffers
    agg_quartet_keys = np.empty(0, dtype=np.uint64)
    agg_weight_sums = np.empty((0, 3), dtype=np.float64)
    agg_counts = np.empty(0, dtype=np.int64)

    pending_keys = []
    pending_sums = []
    pending_counts = []
    pending_entries = 0
    flush_threshold = 3_000_000 if num_species >= 160 else 1_000_000

    def _flush_quartet_aggregates():
        """Flush pending quartet aggregates to main aggregation buffers."""
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

    # Batch inference loop
    print(f"[INFO] Starting batch inference ({len(blocks)} blocks, batch_size={infer_batch_size})...")

    from tqdm import tqdm

    pbar = tqdm(total=len(blocks), desc="[INFO] Batch inference progress")
    for i in range(0, len(blocks), infer_batch_size):
        batch_blocks = blocks[i : i + infer_batch_size]
        actual_bs = len(batch_blocks)
        batch_pattern_tensors = []
        batch_quartets_np = []

        # Prepare batch data
        for block in batch_blocks:
            block_sorted = np.asarray(sorted(block), dtype=np.int32)
            if block_sorted.size == model_size:
                quartets_np = block_sorted[local_quartet_template]
            else:
                quartets_np = np.asarray(
                    list(itertools.combinations(block_sorted.tolist(), 4)),
                    dtype=np.int32,
                )
            batch_quartets_np.append(quartets_np)
            idx_cuda = torch.from_numpy(quartets_np).to(device=device, dtype=torch.long)
            batch_pattern_tensors.append(pattern_freq_cuda.compute_pattern_frequencies_cuda(seq_cuda, idx_cuda))

        # Run inference
        pattern_batch = torch.stack(batch_pattern_tensors, dim=0)
        species_batch = species_enc.unsqueeze(0).expand(actual_bs, -1, -1)
        input_batch = torch.cat([species_batch, pattern_batch], dim=2)

        # Padding到固定长度（如10640），在末尾补0
        target_len = 10640  # 目标长度，可根据需要调整
        current_len = 10626
        pad_len = 14

       
        input_batch_padded = torch.cat(
            [input_batch, input_batch.new_zeros((actual_bs, pad_len, input_batch.size(2)))],
            dim=1,
        )
    

        with torch.no_grad():
            logits = attn_model(input_batch_padded, coeff_blocks, quartet_matrix)
            # 截取回有效长度
            logits = logits[:, :current_len, :]
            probs = torch.softmax(logits, dim=-1)
            weights_batch = (probs * 100.0).cpu().numpy()

        # Aggregate results
        quartets_flat = np.concatenate(batch_quartets_np, axis=0)
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
        if pending_entries >= flush_threshold:
            _flush_quartet_aggregates()
        pbar.update(actual_bs)

    pbar.close()
    _flush_quartet_aggregates()

    # ================================================================================
    # Average Weights and Assemble Initial Tree
    # ================================================================================
    if agg_quartet_keys.size == 0:
        print("[ERROR] No quartet weights generated")
        return ""

    # Compute averaged weights
    avg_weights_int = np.rint(
        agg_weight_sums / agg_counts[:, None]
    ).astype(np.int32)
    quartets_unique = _decode_quartet_keys(agg_quartet_keys)

    # Convert to split format
    pair_indices = [
        ([0, 0, 0], [1, 2, 3], [2, 1, 1], [3, 3, 2])  # (p1L, p1R, p2L, p2R) for each class
    ]

    split_key_chunks = []
    split_weight_chunks = []
    for cls_idx in range(3):
        cls_weights = avg_weights_int[:, cls_idx]
        mask = cls_weights > 0
        if not np.any(mask):
            continue

        q_sel = quartets_unique[mask]
        w_sel = cls_weights[mask].astype(np.int64, copy=False)

        p1L = pair_indices[0][0][cls_idx]
        p1R = pair_indices[0][1][cls_idx]
        p2L = pair_indices[0][2][cls_idx]
        p2R = pair_indices[0][3][cls_idx]

        p1a = q_sel[:, p1L]
        p1b = q_sel[:, p1R]
        p2a = q_sel[:, p2L]
        p2b = q_sel[:, p2R]

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
        print("[ERROR] All averaged quartet weights are zero, cannot assemble")
        return ""

    split_keys_unique = np.concatenate(split_key_chunks)
    split_weight_sums = np.concatenate(split_weight_chunks).astype(np.int64, copy=False)
    print(f"[INFO] Aggregation stats: unique_quartets={agg_quartet_keys.size}, split_lines={split_keys_unique.size}")

    # Write quartet file and run assembler
    result_dir = _get_result_dir(selected_assembler)
    result_dir.mkdir(parents=True, exist_ok=True)
    input_file = result_dir / f"qfm_input_{output_tree_path.stem}.txt"
    output_file = result_dir / f"qfm_output_{output_tree_path.stem}.txt"

    if selected_assembler == "tree-qmc":
        assembled = _run_tree_qmc_with_ram_temp(
            split_keys=split_keys_unique,
            split_weights=split_weight_sums,
            taxon_labels=taxon_labels,
            output_file=output_file,
        )
    else:
        _write_quartet_file(
            split_keys_unique,
            split_weight_sums,
            taxon_labels,
            input_file,
            selected_assembler,
        )
        assembled = _run_quartet_assembler(
            input_file,
            output_file,
            selected_assembler,
        )

    if not assembled:
        print("[ERROR] Initial assembly failed")
        return ""

    # ================================================================================
    # Polytomy Resolution (MLP)
    # ================================================================================
    print("[INFO] Checking and resolving polytomies...")
    tree = Tree(str(output_file))
    tree.unroot()

    # Load MLP model for resolution
    mlp_model_dir = Path("model/homogeneous") if task_type == "homogeneous" else Path("model/heterogeneous")
    mlp_weight = mlp_model_dir / "best_mlp_model.pth"
    mlp_model = MLP().to(device)
    mlp_model.load_state_dict(torch.load(mlp_weight, map_location=device, weights_only=False))
    mlp_model.eval()

    def _infer_subtree_with_mlp(all_rep_ids, child_rep_ids):
        """Infer subtree topology using MLP model."""
        outgroup = sorted(set(all_rep_ids) - set(child_rep_ids))[0] if len(all_rep_ids) > len(child_rep_ids) else None
        quartets = list(itertools.combinations(all_rep_ids, 4))
        if not quartets:
            t = Tree()
            for sp_idx in child_rep_ids:
                t.add_child(name=f"{taxon_prefix}{sp_idx}")
            t.unroot()
            return t

        idx_c = torch.tensor(quartets, dtype=torch.long, device=device)
        p_tensor = pattern_freq_cuda.compute_pattern_frequencies_cuda(seq_cuda, idx_c)

        with torch.no_grad():
            logits = mlp_model(p_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        w_map = defaultdict(int)
        for idx, q in enumerate(quartets):
            ws = [int(round(probs[idx][c]*100)) for c in range(3)]
            if max(ws) < 1:
                ws[np.argmax(probs[idx])] = 1
            q_labels = tuple(f"{taxon_prefix}{sp_idx}" for sp_idx in q)
            for c, w in enumerate(ws):
                if w > 0:
                    w_map[_canonical_split_from_class(q_labels, c)] += w

        with open(input_file, 'w') as f:
            for s, w in w_map.items():
                f.write(f"{s} {w}\n")
        _run_quartet_assembler(input_file, output_file, selected_assembler)

        sub_tree = Tree(str(output_file))
        sub_tree.unroot()
        if outgroup is not None:
            out_label = f"{taxon_prefix}{outgroup}"
            if out_label in sub_tree.get_leaf_names():
                sub_tree.set_outgroup(out_label)
        sub_tree.prune([f"{taxon_prefix}{i}" for i in child_rep_ids], preserve_branch_length=False)
        return sub_tree

    # Iterative polytomy resolution
    iteration = 0
    while True:
        polytomies = find_polytomy_representative_leaves(tree)
        if not polytomies or iteration > 20:
            break
        iteration += 1
        poly = polytomies[0]
        target_node = poly["node"]
        all_reps_raw = poly["representative_leaves"]
        num_children = len(target_node.children)
        child_reps_raw = all_reps_raw[:num_children]
        all_rep_ids = _normalize_rep_idx_list(all_reps_raw)
        child_rep_ids = _normalize_rep_idx_list(child_reps_raw)

        if len(all_rep_ids) < 4 or len(child_rep_ids) < 2:
            print(f"[WARN] Invalid representative leaves in polytomy, skipping: all={all_reps_raw}, child={child_reps_raw}")
            break

        # Build mapping from representative to clade
        rep_to_clade = {}
        for c in target_node.children:
            parsed_ids = _normalize_rep_idx_list(sorted(c.get_leaf_names()))
            if parsed_ids:
                rep_to_clade[f"{taxon_prefix}{parsed_ids[0]}"] = c

        # Infer resolved subtree
        guide = _infer_subtree_with_mlp(all_rep_ids, child_rep_ids)

        def build_resolved(g_node):
            """Build resolved tree from guide tree."""
            if g_node.is_leaf():
                rep_idx = _label_to_idx(g_node.name)
                if rep_idx is None:
                    raise ValueError(f"Cannot parse guide tree leaf label: {g_node.name!r}")
                key = f"{taxon_prefix}{rep_idx}"
                if key not in rep_to_clade:
                    raise KeyError(f"Guide tree leaf {key!r} not found in clade mapping")
                return rep_to_clade[key]
            new_n = Tree()
            for ch in g_node.children:
                new_n.add_child(build_resolved(ch))
            return new_n

        # Replace polytomy with resolved topology
        for ch in list(target_node.children):
            ch.detach()
        for guide_child in guide.children:
            target_node.add_child(build_resolved(guide_child))

    # ================================================================================
    # Finalize Output
    # ================================================================================
    tree.unroot()
    for leaf in tree.iter_leaves():
        idx = _label_to_idx(leaf.name)
        if idx is not None:
            leaf.name = species_names[idx]
    tree.write(outfile=str(output_tree_path), format=1)
    print(f"[SUCCESS] Final binary tree saved to {output_tree_path}")

    if cleanup_temp_files:
        input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)

    # ================================================================================
    # Optional: Compare with Reference Tree
    # ================================================================================
    if ref_tree_path:
        from utils import compute_tree_difference
        metric_value = compute_tree_difference(
            output_tree_path,
            ref_tree_path,
            mode=metric,
            max_quartets=40000
        )
        metric_name = "RF distance" if metric == "rf" else "Quartet concordance"
        print(f"[EVAL] {metric_name}: {metric_value:.6f}")
        return str(output_tree_path), metric_value

    return str(output_tree_path)


###################################################################################################
# Helper Functions
###################################################################################################

def _normalize_output_path(output_tree_path, phy_path):
    """Normalize output tree path, handling directory and file inputs."""
    output_tree_path = Path(output_tree_path)
    if output_tree_path.exists() and output_tree_path.is_dir():
        output_tree_path = output_tree_path / f"{phy_path.stem}.qf2.nwk"
    elif output_tree_path.suffix == "":
        output_tree_path.mkdir(parents=True, exist_ok=True)
        output_tree_path = output_tree_path / f"{phy_path.stem}.qf2.nwk"
    else:
        output_tree_path.parent.mkdir(parents=True, exist_ok=True)
    return output_tree_path


def _get_result_dir(selected_assembler):
    """Get result directory based on selected assembler."""
    shm_dir = Path("temp/shm")
    if selected_assembler == "tree-qmc" and shm_dir.exists() and shm_dir.is_dir():
        return shm_dir / "publish_code_qf2_result"
    else:
        return Path("temp/result")


def _run_quartet_assembler(
    input_file: Path,
    output_file: Path,
    selected_assembler: str,
) -> bool:
    """Run quartet assembler (QFM-FI or TREE-QMC) on input quartet file."""
    if selected_assembler == "qfm-fi":
        qfm_jar = "quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar"
        if not Path(qfm_jar).exists():
            raise FileNotFoundError(f"QFM-FI jar not found: {qfm_jar}")
        result = subprocess.run(
            ["java", "-jar", qfm_jar, str(input_file), str(output_file), "4"],
            check=False,
            capture_output=True
        )
        return result.returncode == 0 and output_file.exists()

    # Default to TREE-QMC
    tree_qmc_bin_path = Path("quartet_assemble_method/TREE-QMC/build/tree-qmc")
    if not tree_qmc_bin_path.exists():
        raise FileNotFoundError("tree-qmc executable not found")
    cmd = [
        str(tree_qmc_bin_path),
        "-i", str(input_file),
        "--quartets",
        "--quartetformat", "___,___|___,___:___",
        "-o", str(output_file),
        "--override"
    ]
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0 and output_file.exists()


def _write_quartet_file(split_keys, split_weights, taxon_labels, output_file, selected_assembler):
    """Write quartet weights to file in format suitable for selected assembler."""
    write_chunk_size = 500_000
    with open(output_file, "w", buffering=16 * 1024 * 1024) as f:
        for start in range(0, split_keys.size, write_chunk_size):
            end = min(start + write_chunk_size, split_keys.size)
            keys_chunk = split_keys[start:end]
            weights_chunk = split_weights[start:end]

            p1a = ((keys_chunk >> 48) & 0xFFFF).astype(np.int32)
            p1b = ((keys_chunk >> 32) & 0xFFFF).astype(np.int32)
            p2a = ((keys_chunk >> 16) & 0xFFFF).astype(np.int32)
            p2b = (keys_chunk & 0xFFFF).astype(np.int32)

            if selected_assembler == "tree-qmc":
                chunk_text = "".join(
                    f"{taxon_labels[a]},{taxon_labels[b]}|{taxon_labels[c]},{taxon_labels[d]}:{w}\n"
                    for a, b, c, d, w in zip(p1a, p1b, p2a, p2b, weights_chunk)
                )
            else:
                chunk_text = "".join(
                    f"(({taxon_labels[a]},{taxon_labels[b]}),({taxon_labels[c]},{taxon_labels[d]})); {w}\n"
                    for a, b, c, d, w in zip(p1a, p1b, p2a, p2b, weights_chunk)
                )
            f.write(chunk_text)


def _preferred_temp_dir() -> Path:
    """Prefer RAM-backed temp dirs for large temporary quartet files."""
    for p in (Path("/dev/shm"), Path("/tmp")):
        if p.exists() and p.is_dir():
            return p
    return Path(tempfile.gettempdir())


def _run_tree_qmc_with_ram_temp(split_keys, split_weights, taxon_labels, output_file) -> bool:
    """
    Run TREE-QMC using a temporary quartet input file on RAM-backed filesystem.

    This avoids heavy write/read overhead on mounted disks like `/mnt/c` while
    remaining compatible with TREE-QMC's file-based parser.
    """
    temp_dir = _preferred_temp_dir()
    temp_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(prefix="qf_quartets_", suffix=".txt", dir=str(temp_dir))
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        _write_quartet_file(
            split_keys=split_keys,
            split_weights=split_weights,
            taxon_labels=taxon_labels,
            output_file=tmp_path,
            selected_assembler="tree-qmc",
        )
        return _run_quartet_assembler(
            input_file=tmp_path,
            output_file=output_file,
            selected_assembler="tree-qmc",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


###################################################################################################
# Command Line Interface
###################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QF: Deep learning-based phylogenetic tree inference tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_qf.py --phy data.phy --out output.nwk

  # With reference tree evaluation
  python run_qf.py --phy data.phy --out output.nwk --ref-tree reference.nwk --metric rf

  # Fast mode
  python run_qf.py --phy data.phy --out output.nwk --run-mode fast

  # Homogeneous mode
  python run_qf.py --phy data.phy --out output.nwk --task-type homogeneous
        """
    )

    # Required arguments
    parser.add_argument("--phy", required=True, help="Input alignment file path (PHY format)")
    parser.add_argument("--out", default="output_qf.nwk", help="Output tree path (file or directory)")

    # Task type
    parser.add_argument(
        "--task-type",
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="Task type: homogeneous (single tree) or heterogeneous (multi-partition conflicts)"
    )

    # Run mode
    parser.add_argument(
        "--run-mode",
        choices=["fast", "regular"],
        default="regular",
        help="Run mode: fast (always TREE-QMC) or regular (QFM-FI for <=96, TREE-QMC for >96)"
    )

    # Reference tree evaluation
    parser.add_argument(
        "--ref-tree",
        default="",
        help="Reference tree path (optional). If provided, computes difference with reference tree"
    )
    parser.add_argument(
        "--metric",
        choices=["rf", "quartet"],
        default="rf",
        help="Evaluation metric: rf (Robinson-Foulds distance) or quartet (quartet concordance)"
    )

    # Advanced parameters
    parser.add_argument("--k-param", type=float, default=3.0, help="Sampling parameter: number of sampled quartets = species_count^k (default: 3.0)")
    parser.add_argument("--infer-batch-size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--no-cleanup-temp-files", action="store_true", help="Keep temporary files")

    args = parser.parse_args()

    # Process ref_tree parameter
    ref_tree_path = args.ref_tree if args.ref_tree else None

    # Call run_qf
    result = run_qf(
        phy_path=args.phy,
        output_tree_path=args.out,
        task_type=args.task_type,
        k_param=args.k_param,
        cleanup_temp_files=not args.no_cleanup_temp_files,
        run_mode=args.run_mode,
        infer_batch_size=args.infer_batch_size,
        ref_tree_path=ref_tree_path,
        metric=args.metric,
    )

    # Handle return value
    if ref_tree_path:
        tree_path, metric_value = result
        print(f"\n[FINAL] Inferred tree: {tree_path}")
        metric_name = "RF distance" if args.metric == "rf" else "Quartet concordance"
        print(f"[FINAL] {metric_name}: {metric_value:.6f}")
    else:
        print(f"\n[FINAL] Inferred tree: {result}")
