import torch
import pickle
import subprocess
import shutil
import itertools
import importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
from ete3 import Tree

# 从 model.py 导入必要的组件
import model
from model import (
    QuartFormer, 
    MLP, 
    _canonical_split_from_class, 
    find_polytomy_representative_leaves
)
import sequence_processor as sp


def run_QF_framework_2(
    phy_path,
    output_tree_path,
    task_type="multilocus",
    k_param=3.0,
    cleanup_temp_files: bool = True,
    run_mode: str = "regular",
    infer_batch_size=32,
):
    """
    独立脚本增强版 run_QF_framework:
    - 仅支持物种数 >= 24。
    - 引入推断 Batching。
    - 对重复四元组权重进行均值化处理。
    - 新增：自动化多分叉修复流程（使用 MLP）。
    - run_mode:
        fast: 固定使用 QMC(TREE-QMC) + 当前极致优化流程
        regular: 物种数 <=96 用 QFM-FI；>96 用 QMC(TREE-QMC)
    """
    phy_path = Path(phy_path)
    output_tree_path = Path(output_tree_path)
    if output_tree_path.exists() and output_tree_path.is_dir():
        output_tree_path = output_tree_path / f"{phy_path.stem}.qf2.nwk"
    elif output_tree_path.suffix == "":
        output_tree_path.mkdir(parents=True, exist_ok=True)
        output_tree_path = output_tree_path / f"{phy_path.stem}.qf2.nwk"
    else:
        output_tree_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    model_size = 24

    def _model_dir(species_for_model: int) -> Path:
        return Path("model/gene") / str(species_for_model) if task_type == "gene" else Path("model/multilocus") / str(species_for_model)

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
    run_mode_key = run_mode.strip().lower().replace("_", "-")
    if run_mode_key not in ("fast", "regular"):
        raise ValueError("run_mode 仅支持 fast、regular")

    def _run_quartet_assembler(
        input_file: Path,
        output_file: Path,
        qmc_compact_format: bool = False,
    ) -> bool:
        if selected_assembler is None:
            raise RuntimeError("内部错误：selected_assembler 未初始化")
        if selected_assembler == "qfm-fi":
            qfm_jar = "quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar"
            if not Path(qfm_jar).exists():
                raise FileNotFoundError(f"未找到 QFM-FI jar: {qfm_jar}")
            result = subprocess.run(
                ["java", "-jar", qfm_jar, str(input_file), str(output_file), "4"],
                check=False, capture_output=True
            )
            return result.returncode == 0 and output_file.exists()
        
        # Default to TREE-QMC / QMC
        tree_qmc_bin_path = Path("quartet_assemble_method/TREE-QMC/build/tree-qmc")
        if not tree_qmc_bin_path.exists():
            raise FileNotFoundError("未找到 tree-qmc 可执行文件")
        quartet_fmt = "___,___|___,___:___" if qmc_compact_format else "((___,___),(___,___));___"
        cmd = [
            str(tree_qmc_bin_path), "-i", str(input_file), "--quartets",
            "--quartetformat", quartet_fmt, "-o", str(output_file), "--override"
        ]
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.returncode == 0 and output_file.exists()

    # --- 1. 加载数据与初始化 ---
    seq_tensor, species_names = sp.load_phy_to_tensor(str(phy_path))
    num_species = seq_tensor.shape[0]
    if num_species < 24:
        raise ValueError("run_QF_framework_2 仅支持物种数 >= 24")
    if num_species >= 65536:
        raise ValueError("当前加速实现要求物种数 < 65536")

    if run_mode_key == "fast":
        selected_assembler = "tree-qmc"
    else:
        selected_assembler = "qfm-fi" if num_species <= 96 else "tree-qmc"
    print(f"[INFO] run_mode={run_mode_key}, selected_assembler={selected_assembler}, num_species={num_species}")

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
        so_path = "cpp_source/batching_algorithms.cpython-310-x86_64-linux-gnu.so"
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
    
    attn_model = QuartFormer(species_num=model_size)
    attn_model.load_state_dict(torch.load(model_dir / "qf1.pt", map_location="cpu", weights_only=False))
    attn_model = attn_model.to(device).eval()
    seq_cuda = torch.from_numpy(seq_tensor).to(device)

    local_quartet_template = np.asarray(
        list(itertools.combinations(range(model_size), 4)), dtype=np.int32
    )

    agg_quartet_keys = np.empty(0, dtype=np.uint64)
    agg_weight_sums = np.empty((0, 3), dtype=np.float64)
    agg_counts = np.empty(0, dtype=np.int64)

    pending_keys = []
    pending_sums = []
    pending_counts = []
    pending_entries = 0
    flush_threshold = 3_000_000 if num_species >= 160 else 1_000_000

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
    
    import pattern_freq_cuda
    from tqdm import tqdm
    
    pbar = tqdm(total=len(blocks), desc="[INFO] 批量推断进度")
    for i in range(0, len(blocks), infer_batch_size):
        batch_blocks = blocks[i : i + infer_batch_size]
        actual_bs = len(batch_blocks)
        batch_pattern_tensors = []
        batch_quartets_np = []
        
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
            
        pattern_batch = torch.stack(batch_pattern_tensors, dim=0)
        species_batch = species_enc.unsqueeze(0).expand(actual_bs, -1, -1)
        input_batch = torch.cat([species_batch, pattern_batch], dim=2)
        
        with torch.no_grad():
            logits = attn_model(input_batch, coeff_blocks, quartet_matrix)
            probs = torch.softmax(logits, dim=-1)
            weights_batch = (probs * 100.0).cpu().numpy()

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

    # --- 4. 均值化权重并组装初始树 ---
    if agg_quartet_keys.size == 0:
        print("[ERROR] 未生成任何四元组权重")
        return ""

    avg_weights_int = np.rint(
        agg_weight_sums / agg_counts[:, None]
    ).astype(np.int32)
    quartets_unique = _decode_quartet_keys(agg_quartet_keys)

    pair1_left = np.array([0, 0, 0], dtype=np.int32)
    pair1_right = np.array([1, 2, 3], dtype=np.int32)
    pair2_left = np.array([2, 1, 1], dtype=np.int32)
    pair2_right = np.array([3, 3, 2], dtype=np.int32)

    split_key_chunks = []
    split_weight_chunks = []
    for cls_idx in range(3):
        cls_weights = avg_weights_int[:, cls_idx]
        mask = cls_weights > 0
        if not np.any(mask):
            continue

        q_sel = quartets_unique[mask]
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
        print("[ERROR] 四元组均值权重全为 0，无法组装")
        return ""

    # 这里的 split key 来自已去重 quartet 的一对一映射：
    # 不同 quartet/class 不会映射到同一个 split key，可直接拼接使用，
    # 无需再做一次全量 np.unique + np.bincount。
    split_keys_unique = np.concatenate(split_key_chunks)
    split_weight_sums = np.concatenate(split_weight_chunks).astype(np.int64, copy=False)
    print(f"[INFO] 聚合统计: unique_quartets={agg_quartet_keys.size}, split_lines={split_keys_unique.size}")

    shm_dir = Path("temp/shm")
    if selected_assembler == "tree-qmc" and shm_dir.exists() and shm_dir.is_dir():
        result_dir = shm_dir / "publish_code_qf2_result"
    else:
        result_dir = Path("temp/result")
    result_dir.mkdir(parents=True, exist_ok=True)
    input_file = result_dir / f"qfm_input_{output_tree_path.stem}.txt"
    output_file = result_dir / f"qfm_output_{output_tree_path.stem}.txt"
    
    # 流式分块写入，保持同样内容语义，同时避免一次性构造超大 lines 列表
    write_chunk_size = 500_000
    with open(input_file, "w", buffering=16 * 1024 * 1024) as f:
        for start in range(0, split_keys_unique.size, write_chunk_size):
            end = min(start + write_chunk_size, split_keys_unique.size)
            keys_chunk = split_keys_unique[start:end]
            weights_chunk = split_weight_sums[start:end]

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
    mlp_weight = Path("model/gene") / "best_mlp_model.pth"
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
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        w_map = defaultdict(int)
        for idx, q in enumerate(quartets):
            ws = [int(round(probs[idx][c]*100)) for c in range(3)]
            if max(ws) < 1: ws[np.argmax(probs[idx])] = 1
            q_labels = tuple(_idx_to_label(sp_idx) for sp_idx in q)
            for c, w in enumerate(ws):
                if w > 0:
                    w_map[_canonical_split_from_class(q_labels, c)] += w

        with open(input_file, 'w') as f:
            for s, w in w_map.items(): f.write(f"{s} {w}\n")
        _run_quartet_assembler(input_file, output_file)
        
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
        if not polytomies or iteration > 20: break
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
    return str(output_tree_path)

if __name__ == "__main__":
    # Benchmark code
    import time
    import csv

    species_list = [24, 48, 96, 192]
    dna_len_list = [100000, 1000000, 10000000]
    csv_path = "output/qf2_benchmark.csv"
    Path("output").mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "dna_len", "time_s", "output_tree"])

    for species_num in species_list:
        for dna_len in dna_len_list:
            phy_path = f"/mnt/c/Users/descfly/Desktop/publish_code/data/{species_num}/0/GTR_{dna_len}_MSA.phy"
            out_path = f"output/output_tree_{species_num}_{dna_len}.nwk"
            _t0 = time.perf_counter()
            result_tree = run_QF_framework_2(
                phy_path=phy_path,
                output_tree_path=out_path,
                task_type="multilocus",
                k_param=3.0,
                cleanup_temp_files=True,
                run_mode="fast",
                infer_batch_size=32,
            )
            elapsed = time.perf_counter() - _t0
            print(f"[TIME] species={species_num} len={dna_len} -> {elapsed:.3f}s | {result_tree}")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([species_num, dna_len, f"{elapsed:.3f}", result_tree])

    # parser = argparse.ArgumentParser(description="QF2 standalone 单任务推断")
    # parser.add_argument("--phy", required=True, help="输入对齐文件路径（如 supermatrix.phy）")
    # parser.add_argument("--out", required=True, help="输出树路径（文件或目录）")
    # parser.add_argument("--ref-tree", default="", help="对比参考树路径；提供后输出一个指标数值")
    # parser.add_argument("--metric", choices=["rf_dist", "rf_acc", "quartet"], default="rf_acc", help="输出指标类型")
    # parser.add_argument("--max-quartets", type=int, default=10000, help="quartet 指标采样数")
    # parser.add_argument("--task-type", choices=["multilocus", "gene"], default="multilocus")
    # parser.add_argument("--k-param", type=float, default=3.0)
    # parser.add_argument("--run-mode", choices=["fast", "regular"], default="regular")
    # parser.add_argument("--infer-batch-size", type=int, default=32)
    # parser.add_argument("--no-cleanup-temp-files", action="store_true")
    # args = parser.parse_args()
# 
    # pred_tree_path = run_QF_framework_2(
    #     phy_path=args.phy,
    #     output_tree_path=args.out,
    #     task_type=args.task_type,
    #     k_param=args.k_param,
    #     cleanup_temp_files=not args.no_cleanup_temp_files,
    #     run_mode=args.run_mode,
    #     infer_batch_size=args.infer_batch_size,
    # )
# 
    # if not pred_tree_path:
    #     raise SystemExit(1)
# 
    # if not args.ref_tree:
    #     print(pred_tree_path)
    #     raise SystemExit(0)
# 
    # if args.metric == "rf_dist":
    #     metric_value = model.compute_tree_difference(pred_tree_path, args.ref_tree, mode="rf")
    # elif args.metric == "rf_acc":
    #     rf_dist = model.compute_tree_difference(pred_tree_path, args.ref_tree, mode="rf")
    #     metric_value = 1.0 - rf_dist
    # else:
    #     metric_value = model.compute_tree_difference(
    #         pred_tree_path,
    #         args.ref_tree,
    #         mode="quartet",
    #         max_quartets=args.max_quartets,
    #     )
    # print(f"{metric_value:.6f}")

