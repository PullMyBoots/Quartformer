import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import pickle
import itertools
import numpy as np
from torch.utils.data import TensorDataset
import torch.utils
from sparse_attn_kernel import (
    _attention,
    get_SparseLayout_SpeciesEncoding,
)
from ete3 import Tree
import subprocess
import shutil
import os
from pathlib import Path
import sequence_processor as sp
import sys
import importlib.util
from collections import defaultdict
import numpy as np
import csv

# Auto-tuned JVM args for QFM-FI on 32-core CPU, 110GB RAM host
# System: 32 cores, 110GB RAM (detected 2026-01-28)
QFMFI_JVM_ARGS = [
    "-Xms24g",                                              # 初始堆24GB，减少动态扩容
    "-Xmx88g",                                              # 最大堆88GB（预留22GB给OS和其他进程）
    "-XX:+UseG1GC",                                         # G1垃圾回收器，适合大内存
    "-XX:ParallelGCThreads=16",                             # GC并行线程数（32核的50%）
    "-XX:ConcGCThreads=4",                                  # GC并发线程数（ParallelGCThreads的1/4）
    "-XX:MaxGCPauseMillis=200",                             # 允许GC暂停稍长，提升吞吐量
    "-XX:+UseStringDeduplication",                          # 字符串去重，节省内存
    "-XX:+DisableExplicitGC",                               # 禁用显式GC，避免性能抖动
    "-Djava.util.concurrent.ForkJoinPool.common.parallelism=20",  # 并行流线程数（32核的62.5%）
]


def _canonical_pair(a: str, b: str):
    return (a, b) if a <= b else (b, a)


def _canonical_quartet_newick(pair1, pair2):
    p1 = _canonical_pair(*pair1)
    p2 = _canonical_pair(*pair2)
    if p2 < p1:
        p1, p2 = p2, p1
    return f"(({p1[0]},{p1[1]}),({p2[0]},{p2[1]}));"


def _canonical_split_from_class(quartet_names, cls: int):
    a, b, c, d = quartet_names
    if cls == 0:
        pair1, pair2 = (a, b), (c, d)
    elif cls == 1:
        pair1, pair2 = (a, c), (b, d)
    else:
        pair1, pair2 = (a, d), (b, c)
    return _canonical_quartet_newick(pair1, pair2)


def _encode_split_key(pair1, pair2):
    """
    将规范化后的两条边编码为 uint64 key: (p1a<<48)|(p1b<<32)|(p2a<<16)|p2b
    约束：物种数 < 65536。
    """
    p1a = pair1[:, 0].astype(np.int64)
    p1b = pair1[:, 1].astype(np.int64)
    p2a = pair2[:, 0].astype(np.int64)
    p2b = pair2[:, 1].astype(np.int64)
    return (p1a << 48) | (p1b << 32) | (p2a << 16) | p2b


def _decode_key_to_split_str(key: int, species_names):
    p1a = (key >> 48) & 0xFFFF
    p1b = (key >> 32) & 0xFFFF
    p2a = (key >> 16) & 0xFFFF
    p2b = key & 0xFFFF
    return f"(({species_names[p1a]},{species_names[p1b]}),({species_names[p2a]},{species_names[p2b]}));"


def _iter_split_weight_items(weighted_quartet_map: dict, species_names):
    for split, weight in weighted_quartet_map.items():
        if isinstance(split, (int, np.integer)):
            split_str = _decode_key_to_split_str(int(split), species_names)
        else:
            split_str = split
        weight_int = int(weight)
        if weight_int > 0:
            yield split_str, weight_int


def _find_tree_qmc_executable() -> str | None:
    which_bin = shutil.which("tree-qmc")
    if which_bin:
        return which_bin

    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "quartet_assemble_method" / "TREE-QMC" / "build_local" / "tree-qmc",
        repo_root / "quartet_assemble_method" / "TREE-QMC" / "build" / "tree-qmc",
        repo_root / "TREE-QMC" / "build_local" / "tree-qmc",
        repo_root / "TREE-QMC" / "build" / "tree-qmc",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _is_tree_qmc_usable(tree_qmc_path: str) -> bool:
    try:
        result = subprocess.run(
            [tree_qmc_path, "-h"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception:
        return False


def _build_tree_qmc_for_current_env() -> str | None:
    repo_root = Path(__file__).resolve().parent
    tree_qmc_roots = [
        repo_root / "quartet_assemble_method" / "TREE-QMC",
        repo_root / "TREE-QMC",
    ]

    for root in tree_qmc_roots:
        src_dir = root / "src"
        mqlib_dir = root / "external" / "MQLib"
        mqlib_static = mqlib_dir / "bin" / "MQLib.a"
        toms_cpp = root / "external" / "toms743" / "toms743.cpp"
        version_file = root / "version.txt"
        if not src_dir.exists() or not mqlib_dir.exists() or not toms_cpp.exists():
            continue

        build_dir = root / "build_local"
        build_dir.mkdir(parents=True, exist_ok=True)
        out_bin = build_dir / "tree-qmc"

        print(f"[INFO] 正在为当前环境重编 TREE-QMC: {root}")
        # 关键：重编 MQLib，避免链接到不兼容系统的旧静态库
        subprocess.run(["make", "clean"], cwd=str(mqlib_dir), check=False)
        mqlib_make = subprocess.run(["make"], cwd=str(mqlib_dir), check=False)
        if mqlib_make.returncode != 0 or not mqlib_static.exists():
            print(f"[WARN] MQLib 编译失败: {mqlib_dir}")
            continue

        try:
            version = version_file.read_text().strip()
        except Exception:
            version = "local"

        src_files = sorted(src_dir.glob("*.cpp"))
        compile_cmd = [
            "g++",
            "-std=c++11",
            "-O2",
            "-I",
            str(root / "external" / "MQLib" / "include"),
            "-I",
            str(root / "external" / "toms743"),
            "-o",
            str(out_bin),
            *[str(p) for p in src_files],
            str(toms_cpp),
            str(mqlib_static),
            "-lm",
            f'-DVERSION="{version}"',
        ]
        compile_result = subprocess.run(compile_cmd, check=False)
        if compile_result.returncode != 0 or not out_bin.exists():
            print(f"[WARN] TREE-QMC 编译失败: {root}")
            continue
        if _is_tree_qmc_usable(str(out_bin)):
            return str(out_bin)
    return None


def _ensure_tree_qmc_executable() -> str | None:
    existing = _find_tree_qmc_executable()
    if existing and _is_tree_qmc_usable(existing):
        return existing
    if existing:
        print(f"[WARN] 检测到不兼容 tree-qmc，尝试重编: {existing}")

    rebuilt = _build_tree_qmc_for_current_env()
    if rebuilt and _is_tree_qmc_usable(rebuilt):
        return rebuilt
    return None


def _accumulate_weighted_quartets_numpy(
    weights_cpu: torch.Tensor,
    quartets_global,
    species_names,
    weighted_quartet_map: dict,
):
    """
    高速累加：NumPy 向量化 + uint64 编码，避免 Python 级双重循环与重复字符串构造。
    """
    if weights_cpu.numel() == 0:
        return

    weights_np = torch.round(weights_cpu).to(torch.int32).numpy()  # (Q,3)
    mask = weights_np > 0
    idx_q, idx_cls = np.nonzero(mask)
    if idx_q.size == 0:
        return

    quartets_np = np.asarray(quartets_global, dtype=np.int32)  # (Q,4)
    quartets_sel = quartets_np[idx_q]  # (K,4)

    pair1_left = np.array([0, 0, 0], dtype=np.int32)
    pair1_right = np.array([1, 2, 3], dtype=np.int32)
    pair2_left = np.array([2, 1, 1], dtype=np.int32)
    pair2_right = np.array([3, 3, 2], dtype=np.int32)

    rows = np.arange(idx_cls.size, dtype=np.int32)
    p1a = quartets_sel[rows, pair1_left[idx_cls]]
    p1b = quartets_sel[rows, pair1_right[idx_cls]]
    p2a = quartets_sel[rows, pair2_left[idx_cls]]
    p2b = quartets_sel[rows, pair2_right[idx_cls]]

    pair1 = np.stack([np.minimum(p1a, p1b), np.maximum(p1a, p1b)], axis=1)
    pair2 = np.stack([np.minimum(p2a, p2b), np.maximum(p2a, p2b)], axis=1)

    swap = (pair2[:, 0] < pair1[:, 0]) | ((pair2[:, 0] == pair1[:, 0]) & (pair2[:, 1] < pair1[:, 1]))
    if swap.any():
        pair1_swapped = pair1.copy()
        pair2_swapped = pair2.copy()
        pair1[swap] = pair2_swapped[swap]
        pair2[swap] = pair1_swapped[swap]

    keys = _encode_split_key(pair1, pair2)
    vals = weights_np[idx_q, idx_cls].astype(np.int64)

    uniq_keys, inv = np.unique(keys, return_inverse=True)
    summed = np.bincount(inv, weights=vals)

    for k, v in zip(uniq_keys.tolist(), summed.tolist()):
        weighted_quartet_map[k] += int(v)

def compute_tree_difference(tree_path1, tree_path2, mode: str = "rf"):
    """
    计算两棵树的差异/一致程度。

    mode:
      - "rf": 返回 RF 准确率 (= 1 - RF 距离 / 最大距离)，越接近 1 越好。
      - "quartet": 返回共同叶子上的四元组一致率，取值[0,1]。
    """
    try:
        tree_path1 = Path(tree_path1)
        tree_path2 = Path(tree_path2)
        if not tree_path1.exists() or not tree_path2.exists():
            raise FileNotFoundError(f"树文件不存在: {tree_path1} 或 {tree_path2}")

        def _load_tree(path: Path):
            content = path.read_text().strip()
            if not content:
                raise ValueError(f"树文件为空: {path}")
            return Tree(content)

        tree1 = _load_tree(tree_path1)
        tree2 = _load_tree(tree_path2)
        tree1.unroot()
        tree2.unroot()
        # 对齐根节点以避免实现差异
        leaves = sorted(tree1.get_leaf_names())
        if leaves:
            tree1.set_outgroup(leaves[0])
            tree2.set_outgroup(leaves[0])
        if mode == "rf":
            rf_result = tree1.robinson_foulds(tree2, unrooted_trees=True)
            rf_distance = rf_result[0]
            max_rf_distance = rf_result[1]
            rf_distance = rf_distance / max_rf_distance
            return rf_distance

        if mode == "quartet":
            try:
                from data_maker import _extract_quartet  # type: ignore

                common_leaves = sorted(set(tree1.get_leaf_names()) & set(tree2.get_leaf_names()))
                quartet_total = 0
                quartet_match = 0
                mismatch_examples = []
                if len(common_leaves) >= 4:
                    for quartet in itertools.combinations(common_leaves, 4):
                        label1 = _extract_quartet(tree1, quartet, return_branchlen=False)
                        label2 = _extract_quartet(tree2, quartet, return_branchlen=False)
                        if label1 is None or label2 is None:
                            continue
                        quartet_total += 1
                        if label1.argmax() == label2.argmax():
                            quartet_match += 1
                        elif len(mismatch_examples) < 3:
                            mismatch_examples.append(quartet)
                quartet_accuracy = (quartet_match / quartet_total) if quartet_total > 0 else 0.0
                print(f"[DEBUG] Quartet accuracy: {quartet_accuracy:.4f} ({quartet_match}/{quartet_total})")
                if mismatch_examples:
                    print(f"[DEBUG] Quartet mismatches (up to 3): {mismatch_examples}")
                return quartet_accuracy
            except Exception as exc:
                print(f"[DEBUG] 四元组一致率计算失败: {exc}")
                return 0.0

        raise ValueError(f"未知 mode: {mode}")
    except Exception as exc:
        print(f"计算RF距离时出错: {exc}")
    return 0.0
########################################################################################################################

# MLP
class MLP(nn.Module):
    def __init__(self, isencoder=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 128)  # 分类输出3个类别
        self.fc3 = nn.Linear(128, 3)  # 分类输出3个类别
        self.isencoder = isencoder
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.isencoder:
            return x
        else:   
            return self.fc3(x)

class MLP2(nn.Module):
    def __init__(self, isencoder=False):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(256 + 3, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
        self.isencoder = isencoder
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.isencoder:
            return x
        else:   
            return self.fc3(x)

# flash_attn
class MultiHeadSelfAttention_flash(nn.Module):
    def __init__(self, token_dim, num_heads):
        super(MultiHeadSelfAttention_flash, self).__init__()
        assert token_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.o_proj = nn.Linear(token_dim, token_dim)
        self.F = _attention.apply

    def forward(self, x, coeff_blocks, quartet_matrix):
        batch_size, seq_len, token_dim = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = self.F(q, k, v, coeff_blocks, quartet_matrix)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, token_dim)
        return self.o_proj(attn_output)

class FeedForwardNetwork_flash(nn.Module):
    def __init__(self, token_dim, hidden_dim):
        super(FeedForwardNetwork_flash, self).__init__()
        self.fc1 = nn.Linear(token_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, token_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerEncoderBlock_flash(nn.Module):
    def __init__(self, token_dim, num_heads, hidden_dim):
        super(TransformerEncoderBlock_flash, self).__init__()
        self.self_attention = MultiHeadSelfAttention_flash(token_dim, num_heads)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = FeedForwardNetwork_flash(token_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x, coeff_blocks, quartet_matrix):
        x = self.norm1(x + self.self_attention(x, coeff_blocks, quartet_matrix))
        x = self.norm2(x + self.ffn(x))
        return x

# QuartFormer
class QuartFormer(nn.Module):
    def __init__(self, species_num):         
        super(QuartFormer, self).__init__()
        # 模型结构超参数
        token_dim=256
        num_heads=16
        hidden_dim=16
        num_layers=3
        num_classes=3

        self.species_num = species_num
        self.mlp_layer = MLP(True)
        self.input_layer = nn.Sequential(nn.Linear(self.species_num + 128, token_dim), nn.ReLU())     
        self.layers = nn.ModuleList([TransformerEncoderBlock_flash(token_dim, num_heads, hidden_dim) for _ in range(num_layers)])         
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, x, coeff_blocks, quartet_matrix):         

        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size * seq_len, feature_dim)
 
        x_mlp = x[:, -256:]
        x_mlp = self.mlp_layer(x_mlp)
        x = torch.cat([x[:, :self.species_num], x_mlp], dim=-1)

        # input_layer
        x = self.input_layer(x)
        x = x.view(batch_size, seq_len, -1)

        # sparse_attention
        for layer in self.layers:             
            x = layer(x, coeff_blocks, quartet_matrix)
        logits = self.classifier(x)
        return logits

class QuartFormer2(nn.Module):

    def __init__(self, species_num):
        super(QuartFormer2, self).__init__()
        
        token_dim=256
        num_heads=16
        hidden_dim=16
        num_layers=3
        
        self.species_num = species_num

        # pattern 部分先走 MLP2 encoder，再与 attn 概率、物种 one-hot 拼接
        self.mlp_layer = MLP2(isencoder=True)
        enc_dim = 128  # MLP2 encoder 输出维度
        self.input_layer = nn.Sequential(nn.Linear(self.species_num + enc_dim + 3, token_dim), nn.ReLU())
        self.layers = nn.ModuleList([TransformerEncoderBlock_flash(token_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(token_dim, 1)

    def forward(self, x, coeff_blocks, quartet_matrix):
        batch_size, seq_len, feature_dim = x.shape
        orig_seq_len = seq_len
        
        flat = x.view(batch_size * seq_len, feature_dim)
        pattern_feats = flat[:, -259:]
        attn_probs = flat[:, -3:].view(batch_size, seq_len, 3)

        # 编码器：若外部传入 mlp2_encoder 则复用其权重（encoder 模式）
        pattern_emb = self.mlp_layer(pattern_feats).view(batch_size, seq_len, -1)

        tokens = torch.cat([x[:, :, :self.species_num], pattern_emb, attn_probs], dim=-1)
        tokens_1 = tokens[0]
        tokens = self.input_layer(tokens)
        for layer in self.layers:
            tokens = layer(tokens, coeff_blocks, quartet_matrix)
        logits = self.classifier(tokens)
        logits = logits[:, :orig_seq_len, :]
        return logits

###########################################################################################################################

def find_polytomy_representative_leaves(tree: Tree, include_root_triplet: bool = False):
    """
    识别树中所有多分叉节点，从每个分支中选择一个代表性叶子。

    重要：正确处理无根树！节点的degree包括children和父节点方向的分支。

    返回值：列表，每个元素为 {"node": 节点对象, "degree": 子分支数, "representative_leaves": [代表叶子列表]}
    说明：
      - 对于无根树：degree = len(children) + (1 if has_parent else 0)
      - 例如：5个children + 1个parent方向 = 6分叉
      - 从每个方向（包括父节点方向）选择一个代表叶子
      - 默认跳过无根树根部的三分叉（无根二叉树常见情形）
    """
    results = []

    # 确保树是无根的
    tree.unroot()

    for node in tree.traverse("preorder"):
        # 计算无根树中的实际degree（包括父节点方向）
        num_children = len(node.children)
        has_parent = node.up is not None
        degree = num_children + (1 if has_parent else 0)

        # 跳过二叉分叉
        if degree <= 3:
            continue

        # 从每个分支方向选择一个代表叶子
        representative_leaves = []

        # 1. 从子节点方向选择
        for child in node.children:
            leaves = sorted(child.get_leaf_names())
            representative_leaves.append(leaves[0])

        # 2. 从父节点方向选择（如果有）
        if has_parent:
            # 找到父节点方向的所有叶子（排除当前节点及其子树）
            all_leaves = set(tree.get_leaf_names())
            current_subtree_leaves = set(node.get_leaf_names())
            parent_direction_leaves = all_leaves - current_subtree_leaves

            if parent_direction_leaves:
                # 选择字典序第一个
                representative_leaves.append(sorted(parent_direction_leaves)[0])
            else:
                # 理论上不应该发生，除非树有问题
                raise ValueError(f"节点 {node} 有父节点但父节点方向没有叶子")

        results.append({
            "node": node,
            "degree": degree,
            "representative_leaves": representative_leaves
        })

    return results


def run_QF_framework(
    phy_path,
    output_tree_path,
    task_type="gene",
    use_confidence=True,
    calculated_load=3.0,
    fast_mode: bool = False,
    quartet_assembler: str = "qmc",
):
    """
    统一封装 QuartFormer + QuartFormer2 + MLP 的推断流程（不依赖 QuartFormer_infer / QuartFormer_dual_infer）。

    - 物种数 8~36：加载对应物种数的 QuartFormer/QuartFormer2 权重，直接拼整树。
    - 物种数 >36：固定加载 24 物种模型拼主树，若存在多分叉：
        1) find_polytomies 找多分叉节点及其叶集
        2) 用 MLP 对每个多分叉叶集推断完全分叉子树
        3) 将子树合成引导树，resolve_polytomies_with_guide 细化

    use_confidence: True 时加载并使用 QuartFormer2 置信度模型；False 则只用 QuartFormer 进行推断。
    fast_mode: 启用权重累加与写文件的轻量加速实现（减少 Python 循环与 I/O 开销）。
    quartet_assembler: 四元组组装后端，可选 "qmc"（TREE-QMC）或 "qfm_fi"（兼容旧流程）。
    """
    quartet_assembler = str(quartet_assembler).lower()
    if quartet_assembler not in {"qmc", "qfm_fi"}:
        raise ValueError("quartet_assembler 仅支持 'qmc' 或 'qfm_fi'")

    def _model_dir(species_for_model: int) -> Path:
        if task_type == "gene":
            return Path("model_gene") / str(species_for_model)
        return Path("model_ml/QuartFormer_multilocus") / str(species_for_model)

    def _load_quartformer_stack(species_for_model: int, load_qf2: bool = True):
        model_dir = _model_dir(species_for_model)
        if not model_dir.exists():
            raise FileNotFoundError(f"未找到模型目录: {model_dir}")
        quartets_pkl = model_dir / "quartets_list.pkl"
        if not quartets_pkl.exists():
            legacy_pkl = model_dir / f"{species_for_model}species_sample_quartets_del_ratio_0.pkl"
            if legacy_pkl.exists():
                quartets_pkl = legacy_pkl
            else:
                raise FileNotFoundError(f"缺少采样四元组: {quartets_pkl}")
        with open(quartets_pkl, "rb") as f:
            sample_quartets = pickle.load(f)

        coeff_blocks = None
        quartet_matrix = None
        species_encoding = None
        coeff_file = model_dir / "coeff_blocks.pt"
        quartet_file = model_dir / "quartet_matrix.pt"
        encoding_file = model_dir / "species_encoding.pt"
        if coeff_file.exists() and quartet_file.exists() and encoding_file.exists():
            coeff_blocks = torch.load(coeff_file, map_location="cpu", weights_only=False)
            quartet_matrix = torch.load(quartet_file, map_location="cpu", weights_only=False)
            species_encoding = torch.load(encoding_file, map_location="cpu", weights_only=False)
        if coeff_blocks is None or quartet_matrix is None or species_encoding is None:
            coeff_blocks, quartet_matrix, species_encoding = get_SparseLayout_SpeciesEncoding(
                sample_quartets, species_for_model
            )

        weight_candidates = [model_dir / "qf1.pt", model_dir / "atten_block.pt"]
        weight_path = next((p for p in weight_candidates if p.exists()), None)
        if weight_path is None:
            raise FileNotFoundError(f"缺少 QuartFormer 权重: {weight_candidates[0]}")

        attn_model = QuartFormer(species_num=species_for_model)
        attn_model.species_encoding = species_encoding
        attn_model.coeff_blocks = coeff_blocks
        attn_model.quartet_matrix = quartet_matrix
        attn_state = torch.load(weight_path, map_location=device, weights_only=False)
        attn_model.load_state_dict(attn_state)
        attn_model = attn_model.to(device)
        attn_model.eval()

        qf2_model = None
        if load_qf2:
            qf2_model = QuartFormer2(species_num=species_for_model)
            qf2_weight = model_dir / "qf2.pt"
            if qf2_weight.exists():
                qf2_state = torch.load(qf2_weight, map_location=device, weights_only=False)
                qf2_model.load_state_dict(qf2_state)
            elif task_type == "gene":
                print(f"[WARN] 缺少 QuartFormer2 权重: {qf2_weight}，使用未训练权重")
            qf2_model = qf2_model.to(device)
            qf2_model.eval()

        return sample_quartets, attn_model, qf2_model

    def _assemble_from_quartets(
        weighted_quartet_map: dict,
        species_names_local,
        output_file: Path,
        unweighted_qfm: bool = False,
        unweighted_lines: list[str] | None = None,
    ) -> bool:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if quartet_assembler == "qmc":
            tree_qmc = _ensure_tree_qmc_executable()
            if tree_qmc is None:
                raise FileNotFoundError(
                    "未找到可运行的 tree-qmc（已尝试自动重编译）。"
                    "请检查 g++/make 环境，或临时使用 quartet_assembler='qfm_fi'。"
                )

            input_file = output_file.with_suffix(".qmc_input.txt")
            split_counter = defaultdict(int)
            if unweighted_qfm:
                for line in (unweighted_lines or []):
                    split = line.strip()
                    if split:
                        split_counter[split] += 1
            else:
                for split, w in _iter_split_weight_items(weighted_quartet_map, species_names_local):
                    split_counter[split] += w

            with open(input_file, "w") as f:
                for split, w in split_counter.items():
                    f.write(f"{split}{float(w):.6f}\n")

            result = subprocess.run(
                [
                    tree_qmc,
                    "-i",
                    str(input_file),
                    "--quartets",
                    "--quartetformat",
                    "((___,___),(___,___));___",
                    "-o",
                    str(output_file),
                    "--override",
                ],
                check=False,
            )
            if result.returncode != 0 or not output_file.exists():
                print(f"[ERROR] TREE-QMC 运行失败或未生成输出，returncode={result.returncode}")
                return False
            return True

        input_file = output_file.with_suffix(".qfm_fi_input.txt")
        qfm_jar = "quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar"
        quartet_type = "1" if unweighted_qfm else "4"

        if unweighted_qfm:
            with open(input_file, "w") as f:
                f.writelines(unweighted_lines or [])
        else:
            with open(input_file, "w") as f:
                for split, w in _iter_split_weight_items(weighted_quartet_map, species_names_local):
                    f.write(f"{split} {w}\n")

        result = subprocess.run(
            ["java"] + QFMFI_JVM_ARGS + ["-jar", qfm_jar, str(input_file), str(output_file), quartet_type],
            check=False,
        )
        if result.returncode != 0 or not output_file.exists():
            print(f"[ERROR] QFM-FI 运行失败或未生成输出，returncode={result.returncode}")
            return False
        return True

    def _infer_tree_dual_small(
        sample_quartets,
        attn_model: QuartFormer,
        qf2_model: QuartFormer2 | None,
        species_for_model: int,
        seq_tensor,  # 新增：直接传入，避免重复加载
        species_names,  # 新增：直接传入
        phy_path,  # 保留接口参数
        output_path: Path,
        unweighted_qfm: bool = False,
        fast_mode: bool = False,
    ):
        """<=36 物种：单 block 推断。若 qf2_model 为 None 则仅用 QuartFormer。"""
        num_species_local = seq_tensor.shape[0]
        blocks = [list(range(num_species_local))]

        seq_cuda = torch.from_numpy(seq_tensor).to(device) if use_cuda else None
        static_species_encoding = attn_model.species_encoding.to(device)
        coeff_blocks = attn_model.coeff_blocks.to(device)
        quartet_matrix = attn_model.quartet_matrix.to(device)
        target_len = quartet_matrix.shape[0]
        if static_species_encoding.shape[0] < target_len:
            pad_rows = target_len - static_species_encoding.shape[0]
            static_species_encoding = torch.cat(
                [static_species_encoding, torch.zeros(pad_rows, static_species_encoding.shape[1], device=device)],
                dim=0,
            )
        elif static_species_encoding.shape[0] > target_len:
            static_species_encoding = static_species_encoding[:target_len]
        attn_model.eval()
        if qf2_model is not None:
            qf2_model.eval()

        weighted_quartet_map = defaultdict(int)
        unweighted_lines = []
        if seq_cuda is None:
            raise RuntimeError("pattern频率计算已配置为仅GPU，但当前未获得CUDA张量。")
        try:
            import pattern_freq_cuda  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pattern频率计算已配置为仅GPU，但无法导入 pattern_freq_cuda。"
            ) from exc

        def _pad_2d(tensor, target):
            cur = tensor.shape[0]
            if cur == target:
                return tensor
            if cur > target:
                return tensor[:target]
            pad = torch.zeros(target - cur, tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)

        for block in blocks:
            block_sorted = sorted(block)
            quartets_global = list(itertools.combinations(block_sorted, 4))

            idx_cuda = torch.tensor(quartets_global, dtype=torch.long, device=device)
            try:
                pattern_tensor = pattern_freq_cuda.compute_pattern_frequencies_cuda(seq_cuda, idx_cuda)
            except Exception as exc:
                raise RuntimeError("GPU pattern频率计算失败（_infer_tree_dual_small）。") from exc

            pattern_tensor = _pad_2d(pattern_tensor, target_len)
            species_enc = _pad_2d(static_species_encoding, target_len)
            input_tensor = torch.cat([species_enc, pattern_tensor], dim=1).unsqueeze(0)

            with torch.no_grad():
                attn_logits = attn_model(input_tensor, coeff_blocks, quartet_matrix)
                attn_logits = attn_logits[:, : len(quartets_global), :]
                attn_probs = torch.softmax(attn_logits.squeeze(0), dim=-1)

            if qf2_model is not None:
                attn_probs_pad = _pad_2d(attn_probs, target_len)
                conf_input = torch.cat([species_enc, pattern_tensor, attn_probs_pad], dim=1).unsqueeze(0)
                with torch.no_grad():
                    conf_logits = qf2_model(conf_input, coeff_blocks, quartet_matrix)
                    conf_logits = conf_logits[:, : len(quartets_global), :]
                    conf_scores = torch.sigmoid(conf_logits.squeeze(0))
                weights = attn_probs * (conf_scores * 100.0)
            else:
                weights = attn_probs * 100.0
            weights_max, max_idx = weights.max(dim=-1)
            needs_boost = weights_max < 1.0
            if needs_boost.any():
                weights = weights.clone()
                weights[needs_boost, max_idx[needs_boost]] = 1.0

            attn_probs_cpu = attn_probs.detach().cpu()
            weights_cpu = weights.detach().cpu()

            if unweighted_qfm:
                for quartet_idx, quartet in enumerate(quartets_global):
                    quartet_names = tuple(species_names[g] for g in quartet)
                    cls = int(attn_probs_cpu[quartet_idx].argmax().item())
                    split = _canonical_split_from_class(quartet_names, cls)
                    unweighted_lines.append(f"{split}\n")
                continue

            if fast_mode:
                _accumulate_weighted_quartets_numpy(weights_cpu, quartets_global, species_names, weighted_quartet_map)
            else:
                for quartet_idx, quartet in enumerate(quartets_global):
                    quartet_names = tuple(species_names[g] for g in quartet)
                    quartet_weights = weights_cpu[quartet_idx]
                    for cls_idx, weight in enumerate(quartet_weights):
                        weight_int = int(weight.item())
                        if weight_int <= 0:
                            continue
                        split = _canonical_split_from_class(quartet_names, cls_idx)
                        weighted_quartet_map[split] += weight_int

        result_dir = Path("wQFM-2020-master/result")
        result_dir.mkdir(parents=True, exist_ok=True)
        suffix = "unweighted" if unweighted_qfm else "weighted"
        output_file = result_dir / f"{quartet_assembler}_output_v5_dual_{suffix}.nwk"

        ok = _assemble_from_quartets(
            weighted_quartet_map=weighted_quartet_map,
            species_names_local=species_names,
            output_file=output_file,
            unweighted_qfm=unweighted_qfm,
            unweighted_lines=unweighted_lines,
        )
        if not ok:
            return str(output_file)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, output_path)
            return str(output_path)
        return str(output_file)

    def _infer_tree_dual_large(
        sample_quartets,
        attn_model: QuartFormer,
        qf2_model: QuartFormer2 | None,
        species_for_model: int,
        seq_tensor,  # 新增：直接传入，避免重复加载
        species_names,  # 新增：直接传入
        phy_path,  # 保留接口参数
        output_path: Path,
        unweighted_qfm: bool = False,
        block_batch_size: int = 64,
        fast_mode: bool = False,
    ):
        """>36 物种：使用 V5 分批，并按 block 批量推断以提升吞吐。

        若 qf2_model 为 None，则仅用 QuartFormer 概率赋权。
        """
        num_species_local = seq_tensor.shape[0]

        repo_root = Path(__file__).resolve().parent
        so_candidates = [
            repo_root / "batching_algorithms*.so",
            repo_root / "cpp_source" / "batching_algorithms*.so",
            repo_root / "quartet_assemble_method" / "cpp" / "batching_algorithms*.so",
        ]
        so_path = None
        for pattern in so_candidates:
            matches = sorted(pattern.parent.glob(pattern.name))
            if matches:
                so_path = str(matches[0])
                break
        if so_path is None:
            raise FileNotFoundError("batching_algorithms*.so not found under repo; please build it first.")
        spec = importlib.util.spec_from_file_location("batching_algorithms", so_path)
        batching_algorithms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batching_algorithms)  # type: ignore
        blocks = batching_algorithms.pair_balanced_block_design_v5(
            num_species=num_species_local,
            k_param=calculated_load,
            batch_size=species_for_model,
            seed=42,
            species_weight=2.0,
            threads=1,
        )

        seq_cuda = torch.from_numpy(seq_tensor).to(device) if use_cuda else None
        static_species_encoding = attn_model.species_encoding.to(device)
        coeff_blocks = attn_model.coeff_blocks.to(device)
        quartet_matrix = attn_model.quartet_matrix.to(device)
        target_len = quartet_matrix.shape[0]
        if static_species_encoding.shape[0] < target_len:
            pad_rows = target_len - static_species_encoding.shape[0]
            static_species_encoding = torch.cat(
                [static_species_encoding, torch.zeros(pad_rows, static_species_encoding.shape[1], device=device)],
                dim=0,
            )
        elif static_species_encoding.shape[0] > target_len:
            static_species_encoding = static_species_encoding[:target_len]
        attn_model.eval()
        if qf2_model is not None:
            qf2_model.eval()

        weighted_quartet_map = defaultdict(int)
        unweighted_lines = []

        def _pad_2d(tensor, target):
            cur = tensor.shape[0]
            if cur == target:
                return tensor
            if cur > target:
                return tensor[:target]
            pad = torch.zeros(target - cur, tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)

        def _pad_3d_batch(tensor, target):
            cur = tensor.shape[1]
            if cur == target:
                return tensor
            if cur > target:
                return tensor[:, :target, :]
            pad = torch.zeros(
                tensor.shape[0],
                target - cur,
                tensor.shape[2],
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return torch.cat([tensor, pad], dim=1)

        # 预先一次性准备 species_enc（相同内容，后面 expand）
        species_enc = _pad_2d(static_species_encoding, target_len)
        if seq_cuda is None:
            raise RuntimeError("pattern频率计算已配置为仅GPU，但当前未获得CUDA张量。")
        try:
            import pattern_freq_cuda as pattern_freq_cuda_mod  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pattern频率计算已配置为仅GPU，但无法导入 pattern_freq_cuda。"
            ) from exc
        grouped_uniform_fn = None
        if hasattr(pattern_freq_cuda_mod, "compute_pattern_frequencies_cuda_grouped_uniform"):
            grouped_uniform_fn = pattern_freq_cuda_mod.compute_pattern_frequencies_cuda_grouped_uniform
        scalar_pattern_fn = None
        if hasattr(pattern_freq_cuda_mod, "compute_pattern_frequencies_cuda"):
            scalar_pattern_fn = pattern_freq_cuda_mod.compute_pattern_frequencies_cuda
        if grouped_uniform_fn is None and scalar_pattern_fn is None:
            raise RuntimeError("pattern_freq_cuda 缺少可用的 pattern 频率计算接口。")

        local_quartets_cpu = torch.tensor(
            list(itertools.combinations(range(species_for_model), 4)),
            dtype=torch.long,
        )
        len_q_local = int(local_quartets_cpu.shape[0])
        from tqdm import tqdm
        total_batches = (len(blocks) + block_batch_size - 1) // block_batch_size

        for start in tqdm(range(0, len(blocks), block_batch_size),
                          total=total_batches,
                          desc="处理 blocks",
                          unit="batch"):
            batch_blocks = blocks[start : start + block_batch_size]
            valid_blocks = []

            for block in batch_blocks:
                if len(block) < 4 or len(block) != species_for_model:
                    continue
                valid_blocks.append(sorted(block))

            if not valid_blocks:
                continue

            blocks_cpu = torch.tensor(valid_blocks, dtype=torch.long)
            quartet_idx_cpu = blocks_cpu[:, local_quartets_cpu]  # [B, Q, 4]
            quartet_idx_cuda = quartet_idx_cpu.to(device=device, dtype=torch.long)
            try:
                if grouped_uniform_fn is not None:
                    pattern_batch = grouped_uniform_fn(seq_cuda, quartet_idx_cuda)
                else:
                    pattern_batch_list = []
                    for i in range(quartet_idx_cuda.shape[0]):
                        pattern_batch_list.append(scalar_pattern_fn(seq_cuda, quartet_idx_cuda[i]))
                    pattern_batch = torch.stack(pattern_batch_list, dim=0)
            except Exception as exc:
                raise RuntimeError("GPU pattern频率计算失败（_infer_tree_dual_large）。") from exc
            pattern_batch = _pad_3d_batch(pattern_batch, target_len)

            quartet_idx_np = quartet_idx_cpu.numpy()
            valid_info = [
                (valid_blocks[i], quartet_idx_np[i], len_q_local)
                for i in range(len(valid_blocks))
            ]

            species_batch = species_enc.unsqueeze(0).expand(pattern_batch.size(0), -1, -1)
            input_batch = torch.cat([species_batch, pattern_batch], dim=2)

            with torch.no_grad():
                attn_logits = attn_model(input_batch, coeff_blocks, quartet_matrix)
                attn_probs = torch.softmax(attn_logits, dim=-1)

            if qf2_model is not None:
                attn_probs_padded = attn_probs
                conf_input_batch = torch.cat([species_batch, pattern_batch, attn_probs_padded], dim=2)
                with torch.no_grad():
                    conf_logits = qf2_model(conf_input_batch, coeff_blocks, quartet_matrix)
                    conf_scores = torch.sigmoid(conf_logits)
            else:
                conf_scores = None

            for idx, (block_sorted, quartets_global, len_q) in enumerate(valid_info):
                attn_probs_block = attn_probs[idx, :len_q, :]
                if conf_scores is not None:
                    conf_scores_block = conf_scores[idx, :len_q, :]
                    weights = attn_probs_block * (conf_scores_block * 100.0)
                else:
                    weights = attn_probs_block * 100.0
                weights_max, max_idx = weights.max(dim=-1)
                needs_boost = weights_max < 1.0
                if needs_boost.any():
                    weights = weights.clone()
                    weights[needs_boost, max_idx[needs_boost]] = 1.0

                attn_probs_cpu = attn_probs_block.detach().cpu()
                weights_cpu = weights.detach().cpu()

                if unweighted_qfm:
                    for quartet_idx, quartet in enumerate(quartets_global):
                        quartet_names = tuple(species_names[g] for g in quartet)
                        cls = int(attn_probs_cpu[quartet_idx].argmax().item())
                        split = _canonical_split_from_class(quartet_names, cls)
                        unweighted_lines.append(f"{split}\n")
                    continue

                if fast_mode:
                    _accumulate_weighted_quartets_numpy(weights_cpu, quartets_global, species_names, weighted_quartet_map)
                else:
                    for quartet_idx, quartet in enumerate(quartets_global):
                        quartet_names = tuple(species_names[g] for g in quartet)
                        quartet_weights = weights_cpu[quartet_idx]
                        for cls_idx, weight in enumerate(quartet_weights):
                            weight_int = int(weight.item())
                            if weight_int <= 0:
                                continue
                            split = _canonical_split_from_class(quartet_names, cls_idx)
                            weighted_quartet_map[split] += weight_int

        result_dir = Path("wQFM-2020-master/result")
        result_dir.mkdir(parents=True, exist_ok=True)
        suffix = "unweighted" if unweighted_qfm else "weighted"
        output_file = result_dir / f"{quartet_assembler}_output_v5_dual_{suffix}.nwk"

        ok = _assemble_from_quartets(
            weighted_quartet_map=weighted_quartet_map,
            species_names_local=species_names,
            output_file=output_file,
            unweighted_qfm=unweighted_qfm,
            unweighted_lines=unweighted_lines,
        )
        if not ok:
            return str(output_file)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, output_path)
            return str(output_path)
        return str(output_file)

    def _load_mlp():
        if task_type == "gene":
            mlp_weight = Path("model_gene/MLP/best_mlp_model.pth")
        else:
            mlp_weight = Path("model_ml/MLP_tuned/best_mlp_model.pth")
        if not mlp_weight.exists():
            raise FileNotFoundError(f"缺少 MLP 权重: {mlp_weight}")
        mlp_model = MLP()
        mlp_state = torch.load(mlp_weight, map_location=device, weights_only=False)
        mlp_model.load_state_dict(mlp_state)
        mlp_model = mlp_model.to(device)
        mlp_model.eval()
        return mlp_model

    def _infer_subtree_with_mlp(
        mlp_model: MLP,
        all_representatives: list,
        child_representatives: list,
        seq_tensor,  # 用于 CUDA 加速
        species_names,  # 用于索引映射
        phy_path,
    ) -> Tree:
        """
        使用 MLP 模型推断指定物种子集的系统发育子树（带 CUDA 加速）。

        策略：
          1. 用 MLP 推断 all_representatives 的完整无根树（CUDA 加速）
          2. 将 all_representatives 中但不在 child_representatives 中的物种作为外群
          3. 用外群定根（set_outgroup）
          4. Prune 到 child_representatives，得到有根的 guide_tree

        这样模拟了 guide_tree.prune(child_representatives, preserve_branch_length=False) 的效果。
        """
        # 1. 确定外群：在 all_representatives 中但不在 child_representatives 中的物种
        all_set = set(all_representatives)
        child_set = set(child_representatives)
        outgroup_candidates = sorted(all_set - child_set)

        if len(outgroup_candidates) > 1:
            outgroup = outgroup_candidates[0]
        elif len(outgroup_candidates) == 1:
            outgroup = outgroup_candidates[0]
        else:
            outgroup = None

        print(f"[DEBUG] infer_subtree: 全部={sorted(all_representatives)}, 子节点={sorted(child_representatives)}, 外群={outgroup}")

        # 2. 用 MLP 推断 all_representatives 的完整树
        quartets = list(itertools.combinations(all_representatives, 4))
        if len(quartets) == 0:
            tree = Tree()
            for sp in child_representatives:
                tree.add_child(name=sp)
            tree.unroot()
            return tree

        # 3. 仅使用 GPU 计算 pattern 频率
        if not use_cuda:
            raise RuntimeError("pattern频率计算已配置为仅GPU，但当前环境未启用CUDA。")
        try:
            import pattern_freq_cuda  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pattern频率计算已配置为仅GPU，但无法导入 pattern_freq_cuda。"
            ) from exc

        species_to_idx = {name: idx for idx, name in enumerate(species_names)}
        quartets_global = [
            tuple(species_to_idx[name] for name in quartet) for quartet in quartets
        ]

        seq_cuda = torch.from_numpy(seq_tensor).to(device)
        idx_cuda = torch.tensor(quartets_global, dtype=torch.long, device=device)
        try:
            pattern_tensor = pattern_freq_cuda.compute_pattern_frequencies_cuda(seq_cuda, idx_cuda)
        except Exception as exc:
            raise RuntimeError("GPU pattern频率计算失败（_infer_subtree_with_mlp）。") from exc

        # 4. MLP 推断（输入就是 pattern_tensor，无需拼接）
        mlp_model.eval()
        with torch.no_grad():
            logits = mlp_model(pattern_tensor)
        probs = torch.softmax(logits.cpu(), dim=-1)

        # 5. 将推断结果转换为 quartet 权重
        weighted_quartet_map = defaultdict(int)
        for idx, quartet in enumerate(quartets):
            weights = [
                int((probs[idx][0] * 100).item()),
                int((probs[idx][1] * 100).item()),
                int((probs[idx][2] * 100).item()),
            ]
            if max(weights) < 1:
                weights[int(probs[idx].argmax(dim=-1).item())] = 1
            for cls, w in enumerate(weights):
                if w <= 0:
                    continue
                split = _canonical_split_from_class(quartet, cls)
                weighted_quartet_map[split] += w

        # 6. 使用 quartet assembler（默认 TREE-QMC）从 quartet 组装树
        import tempfile
        with tempfile.TemporaryDirectory(prefix="quartet_assemble_") as temp_dir:
            temp_output = Path(temp_dir) / "assembled_subtree.nwk"

            ok = _assemble_from_quartets(
                weighted_quartet_map=weighted_quartet_map,
                species_names_local=species_names,
                output_file=temp_output,
                unweighted_qfm=False,
                unweighted_lines=None,
            )
            if not ok:
                raise RuntimeError(f"{quartet_assembler} 推断失败，未生成输出文件")

            # 读取推断的树
            tree_txt = temp_output.read_text().strip()
            tree = Tree(tree_txt)
            tree.unroot()

            # 7. 用外群定根（如果有）
            if outgroup is not None and outgroup in [leaf.name for leaf in tree.iter_leaves()]:
                tree.set_outgroup(outgroup)

            # 8. Prune 到 child_representatives
            tree.prune(child_representatives, preserve_branch_length=False)

            print(f"[DEBUG] infer_subtree: 推断完成，叶节点={sorted(tree.get_leaf_names())}")
            return tree

    def _resolve_polytomies_with_mlp(tree: Tree, mlp_model: MLP) -> Tree:
        """使用 MLP 模型推断多分叉节点的拓扑结构，逐个解析多分叉。"""
        iteration = 0
        while True:
            iteration += 1

            # 1. 检测所有多分叉节点
            polytomies = find_polytomy_representative_leaves(tree, include_root_triplet=False)

            if not polytomies:
                break

            # 2. 取第一个多分叉进行解析
            poly_info = polytomies[0]
            target_node = poly_info["node"]
            degree = poly_info["degree"]
            all_representatives = poly_info["representative_leaves"]

            # 获取子节点方向的代表叶子（用于构建 rep_to_clade）
            num_children = len(target_node.children)
            child_representatives = all_representatives[:num_children]

            if len(all_representatives) < 4:
                print(f"[WARN] 多分叉总分支数={len(all_representatives)} < 4，无法用 quartet 推断，强制二叉化")
                children = list(target_node.children)
                while len(children) > 2:
                    parent = Tree()
                    parent.add_child(children.pop(0))
                    parent.add_child(children.pop(0))
                    children.append(parent)
                for ch in list(target_node.children):
                    ch.detach()
                for ch in children:
                    target_node.add_child(ch)
                continue

            print(f"[INFO] 第 {iteration} 轮：解析多分叉 (degree={degree}, 子分支数={num_children})")
            print(f"      全部代表={all_representatives}, 子节点代表={child_representatives}")

            # 3. 为每个子树选择代表叶子，构建映射（只包括子节点方向）
            rep_to_clade = {}
            for child in target_node.children:
                leaves = sorted(child.get_leaf_names())
                rep_name = leaves[0]
                rep_to_clade[rep_name] = child

            # 4. 使用 MLP 推断所有代表的子树（带 CUDA 加速）
            guide_tree = _infer_subtree_with_mlp(
                mlp_model=mlp_model,
                all_representatives=all_representatives,
                child_representatives=child_representatives,
                seq_tensor=seq_tensor,
                species_names=species_names,
                phy_path=phy_path,
            )

            # 5. 用 guide tree 的结构重构多分叉节点
            def build_resolved_structure(guide_node):
                if guide_node.is_leaf():
                    original_clade = rep_to_clade[guide_node.name]
                    return original_clade
                else:
                    new_node = Tree()
                    for child in guide_node.children:
                        resolved_child = build_resolved_structure(child)
                        new_node.add_child(resolved_child)
                    return new_node

            # 分离旧子节点
            for child in list(target_node.children):
                child.detach()

            # 将重构的子树挂回 target_node
            for guide_child in guide_tree.children:
                resolved_branch = build_resolved_structure(guide_child)
                target_node.add_child(resolved_branch)
            print(f"[INFO] 成功解析多分叉")

        tree.unroot()
        return tree

    phy_path = Path(phy_path)
    output_tree_path = Path(output_tree_path)
    output_tree_path.parent.mkdir(parents=True, exist_ok=True)
    if task_type not in ("gene", "ml"):
        raise ValueError("task_type 仅支持 'gene' 或 'ml'")

    # 一次性加载 PHY 文件，避免重复 I/O
    try:
        seq_tensor, species_names = sp.load_phy_to_tensor(str(phy_path))
    except Exception as exc:
        raise ValueError(f"读取PHY失败: {phy_path}") from exc
    num_species = seq_tensor.shape[0]
    # 注意：不删除 seq_tensor，而是传递给子函数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if not use_cuda:
        raise RuntimeError("当前 run_QF_framework 已配置为仅GPU计算 pattern 频率，未检测到可用CUDA设备。")
    try:
        import pattern_freq_cuda  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "当前 run_QF_framework 已配置为仅GPU计算 pattern 频率，但无法导入 pattern_freq_cuda。"
        ) from exc


    # 8~36 物种：直接推断并返回
    if num_species <= 36:
        sample_quartets, attn_model, qf2_model = _load_quartformer_stack(num_species, load_qf2=use_confidence)
        tree_path = _infer_tree_dual_small(
            sample_quartets,
            attn_model,
            qf2_model,
            num_species,
            seq_tensor,  # ✓ 传递已加载的数据
            species_names,  # ✓ 传递已加载的数据
            phy_path,  # ✓ 传递 PHY 路径
            output_tree_path,
            unweighted_qfm=False,
            fast_mode=fast_mode,
        )
        # 推断完成后释放内存
        del seq_tensor
        return str(tree_path)

    # >36 物种：先用24物种模型拼主树，再用MLP细化多分叉
    sample_quartets, attn_model, qf2_model = _load_quartformer_stack(24, load_qf2=use_confidence)
    polytomy_tree_path = output_tree_path.with_suffix(".polytomy.nwk")
    
    inferred_path = _infer_tree_dual_large(
        sample_quartets,
        attn_model,
        qf2_model,
        24,
        seq_tensor,  # ✓ 传递已加载的数据
        species_names,  # ✓ 传递已加载的数据
        phy_path,  # ✓ 传递 PHY 路径
        polytomy_tree_path,
        unweighted_qfm=False,
        block_batch_size=32,
        fast_mode=fast_mode,
    )

    base_tree_txt = Path(inferred_path).read_text().strip()
    base_tree = Tree(base_tree_txt)
    base_tree.unroot()
    # print(base_tree)

    # 检测多分叉
    polytomies = find_polytomy_representative_leaves(base_tree, include_root_triplet=False)
    if not polytomies:
        print("[INFO] 未检测到多分叉，直接输出主树")
        # 可以释放 seq_tensor
        del seq_tensor
        base_tree.write(outfile=str(output_tree_path))
        return str(output_tree_path)

    print(f"[INFO] 检测到 {len(polytomies)} 个多分叉节点，开始使用 MLP 推断解析")

    # 加载 MLP 模型
    mlp_model = _load_mlp()

    # 使用 MLP 推断并解析多分叉（带 CUDA 加速）
    try:
        resolved_tree = _resolve_polytomies_with_mlp(base_tree, mlp_model)
        # print(resolved_tree)

        # 推断完成后释放 seq_tensor
        del seq_tensor

        # 验证是否还有多分叉
        remaining_polytomies = find_polytomy_representative_leaves(resolved_tree, include_root_triplet=False)
        if remaining_polytomies:
            print(f"[WARN] 解析后仍有 {len(remaining_polytomies)} 个多分叉")

        resolved_tree.write(outfile=str(output_tree_path))
        return str(output_tree_path)
    except Exception as exc:
        print(f"[ERROR] MLP 多分叉解析失败: {exc}")
        import traceback
        traceback.print_exc()
        base_tree.write(outfile=str(output_tree_path))
        return str(output_tree_path)



##########################################################################################################################

if __name__ == "__main__":

    #  
    species_list = [96]
    dna_len_list = [10000000]
    csv_path = "output/qf_benchmark.csv"
    Path("output").mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "dna_len", "time_s", "output_tree"])
    for species_num in species_list:
        for dna_len in dna_len_list:
            phy_path = f"/mnt/c/Users/descfly/Desktop/publish_code/data/{species_num}/0/GTR_{dna_len}_MSA.phy"
            out_path = f"output/output_tree_{species_num}_{dna_len}.nwk"
            _t0 = time.perf_counter()
            result_tree = run_QF_framework(
                phy_path,
                out_path,
                task_type="gene",
                use_confidence=True,
                fast_mode=True,
                calculated_load=3,
                
            )
            elapsed = time.perf_counter() - _t0
            print(f"[TIME] species={species_num} len={dna_len} -> {elapsed:.3f}s | {result_tree}")
            #with open(csv_path, "a", newline="") as f:
            #    writer = csv.writer(f)
            #    writer.writerow([species_num, dna_len, f"{elapsed:.3f}", result_tree])
