# Utility functions for QuartFormer2
from pathlib import Path
from ete3 import Tree
import itertools
import math
import numpy as np


def _support_label_for_plot(clade):
    if clade.is_terminal():
        return None
    confidence = getattr(clade, "confidence", None)
    if confidence is None:
        return None
    text = f"{float(confidence):.2f}"
    return text.rstrip("0").rstrip(".")


def draw_newick_tree_figure(
    tree_path: Path,
    out_path: Path,
    title: str | None = None,
    show_support: bool = False,
) -> None:
    """
    Render a Newick tree to a static figure file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Bio import Phylo

    tree = Phylo.read(str(tree_path), "newick")
    tree.ladderize()

    leaf_count = len(tree.get_terminals())
    fig_height = max(8.0, leaf_count * 0.32)
    fig_width = 18.0 if show_support else 15.0

    plt.rcParams.update(
        {
            "font.size": 8,
            "lines.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        show_confidence=False,
        label_func=lambda clade: clade.name if clade.is_terminal() else None,
        branch_labels=_support_label_for_plot if show_support else None,
    )

    ax.set_title(title or tree_path.name, fontsize=14, pad=14)
    ax.set_xlabel("Branch length")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _canonical_split_from_class(quartet_names, cls: int):
    """
    Generate canonical Newick format split string from quartet names and class.

    Args:
        quartet_names: Quartet species names (a, b, c, d)
        cls: Topology class (0, 1, 2)

    Returns:
        Newick format quartet string, e.g., "((A,B),(C,D));"
    """
    a, b, c, d = quartet_names

    # Determine pairing based on class
    if cls == 0:
        pair1, pair2 = (a, b), (c, d)
    elif cls == 1:
        pair1, pair2 = (a, c), (b, d)
    else:  # cls == 2
        pair1, pair2 = (a, d), (b, c)

    # Normalize species order in each pair
    p1 = pair1 if pair1[0] <= pair1[1] else (pair1[1], pair1[0])
    p2 = pair2 if pair2[0] <= pair2[1] else (pair2[1], pair2[0])

    # Ensure pairs are in lexicographic order
    if p2 < p1:
        p1, p2 = p2, p1

    return f"(({p1[0]},{p1[1]}),({p2[0]},{p2[1]}));"


def _extract_quartet(tree: Tree, quartet: tuple, return_branchlen: bool = False) -> np.ndarray:
    """
    Extract quartet topology label from tree.

    Args:
        tree: ete3 Tree object
        quartet: Tuple of 4 species names
        return_branchlen: Whether to return branch length (currently not implemented)

    Returns:
        np.ndarray: One-hot encoded 3D vector representing one of three quartet topologies
                   [1,0,0] = ((a,b),(c,d))
                   [0,1,0] = ((a,c),(b,d))
                   [0,0,1] = ((a,d),(b,c))
        None: If quartet species not in tree
    """
    a, b, c, d = quartet

    # Check if all species exist in tree
    tree_leaves = set(tree.get_leaf_names())
    if not all(sp in tree_leaves for sp in quartet):
        return None

    # Get MRCA topology
    try:
        # Calculate pairwise distances
        dists = {}
        for sp1, sp2 in itertools.combinations(quartet, 2):
            dists[(sp1, sp2)] = tree.get_distance(sp1, sp2)

        # Determine topology: check which pair distance equals sum of other pairs
        # Topology 0: (a,b)|(c,d) - dist(a,b) + dist(c,d) minimal
        # Topology 1: (a,c)|(b,d) - dist(a,c) + dist(b,d) minimal
        # Topology 2: (a,d)|(b,c) - dist(a,d) + dist(b,c) minimal

        sum_01 = dists[(a, b)] + dists[(c, d)]
        sum_02 = dists[(a, c)] + dists[(b, d)]
        sum_03 = dists[(a, d)] + dists[(b, c)]

        # Find minimum sum, corresponding topology is correct quartet topology
        min_sum = min(sum_01, sum_02, sum_03)

        if abs(min_sum - sum_01) < 1e-6:
            return np.array([1, 0, 0], dtype=np.float32)
        elif abs(min_sum - sum_02) < 1e-6:
            return np.array([0, 1, 0], dtype=np.float32)
        else:
            return np.array([0, 0, 1], dtype=np.float32)
    except Exception:
        return None


def compute_tree_difference(tree_path1, tree_path2, mode: str = "rf", max_quartets: int = None):
    """
    Compute difference/similarity between two trees.

    Args:
        tree_path1, tree_path2: Paths to two trees
        mode: Evaluation mode
            - "rf": Return RF accuracy (= 1 - RF distance / max distance), closer to 1 is better
            - "quartet": Return quartet concordance on common leaves, range [0,1]
        max_quartets: Maximum number of quartets to sample (quartet mode only)
            - If None, compute all quartets
            - If common_leaves > 96 and not specified, default to 10000
            - If specified, use specified value
    """
    try:
        tree_path1 = Path(tree_path1)
        tree_path2 = Path(tree_path2)
        if not tree_path1.exists() or not tree_path2.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path1} or {tree_path2}")

        def _load_tree(path: Path):
            content = path.read_text().strip()
            if not content:
                raise ValueError(f"Empty tree file: {path}")
            return Tree(content)

        tree1 = _load_tree(tree_path1)
        tree2 = _load_tree(tree_path2)
        tree1.unroot()
        tree2.unroot()
        # Align root nodes to avoid implementation differences
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
                import random
                common_leaves = sorted(set(tree1.get_leaf_names()) & set(tree2.get_leaf_names()))
                quartet_total = 0
                quartet_match = 0
                mismatch_examples = []

                if len(common_leaves) >= 4:
                    n = len(common_leaves)
                    total_possible = math.comb(n, 4)

                    # If common_leaves > 96 and max_quartets not specified, default to 10000
                    if len(common_leaves) > 96 and max_quartets is None:
                        max_quartets = 10000

                    def _unrank_combination(n_items: int, k_items: int, rank: int):
                        """Map rank in [0, C(n,k)) to the k-combination (lexicographic order)."""
                        result = []
                        start = 0
                        remaining = rank
                        for choose_idx in range(k_items, 0, -1):
                            for first in range(start, n_items - choose_idx + 1):
                                count = math.comb(n_items - first - 1, choose_idx - 1)
                                if remaining < count:
                                    result.append(first)
                                    start = first + 1
                                    break
                                remaining -= count
                        return tuple(result)

                    # Uniform sample without replacement from all C(n,4) combinations.
                    if max_quartets is not None and max_quartets < total_possible:
                        print(f"[INFO] Total {total_possible:,} quartets, sampling {max_quartets:,}")
                        sampled_ranks = random.sample(range(total_possible), max_quartets)
                        quartets_iter = (
                            tuple(common_leaves[idx] for idx in _unrank_combination(n, 4, rank))
                            for rank in sampled_ranks
                        )
                    else:
                        quartets_iter = itertools.combinations(common_leaves, 4)

                    for quartet in quartets_iter:
                        label1 = _extract_quartet(tree1, quartet, return_branchlen=False)
                        label2 = _extract_quartet(tree2, quartet, return_branchlen=False)
                        if label1 is None or label2 is None:
                            continue
                        quartet_total += 1
                        if label1.argmax() == label2.argmax():
                            quartet_match += 1
                        elif len(mismatch_examples) < 3:
                            mismatch_examples.append(quartet)

                    # If sampled, adjust display based on sample count
                    if max_quartets is not None and max_quartets < total_possible:
                        print(f"[INFO] Actually computed {quartet_total:,} quartets")

                quartet_accuracy = (quartet_match / quartet_total) if quartet_total > 0 else 0.0
                print(f"[DEBUG] Quartet accuracy: {quartet_accuracy:.4f} ({quartet_match}/{quartet_total})")
                if mismatch_examples:
                    print(f"[DEBUG] Quartet mismatches (up to 3): {mismatch_examples}")
                return quartet_accuracy
            except Exception as exc:
                print(f"[DEBUG] Quartet concordance calculation failed: {exc}")
                return 0.0

        raise ValueError(f"Unknown mode: {mode}")
    except Exception as exc:
        print(f"Error computing RF distance: {exc}")
    return 0.0


def find_polytomy_representative_leaves(tree: Tree, include_root_triplet: bool = False):
    """
    Identify all polytomy nodes in tree, select one representative leaf from each branch.

    Important: Correctly handles unrooted trees! Node degree includes children and parent branch.

    Returns: List, each element is {"node": node object, "degree": num branches, "representative_leaves": [leaf list]}
    Notes:
      - For unrooted trees: degree = len(children) + (1 if has_parent else 0)
      - Example: 5 children + 1 parent direction = 6-way fork
      - Select one representative leaf from each direction (including parent direction)
      - Skip 3-way fork at unrooted tree root by default (common in unrooted binary trees)
    """
    results = []

    # Ensure tree is unrooted
    tree.unroot()

    for node in tree.traverse("preorder"):
        # Calculate actual degree in unrooted tree (including parent direction)
        num_children = len(node.children)
        has_parent = node.up is not None
        degree = num_children + (1 if has_parent else 0)

        # Skip binary bifurcations
        if degree <= 3:
            continue

        # Select one representative leaf from each branch direction
        representative_leaves = []

        # 1. Select from child directions
        for child in node.children:
            leaves = sorted(child.get_leaf_names())
            representative_leaves.append(leaves[0])

        # 2. Select from parent direction (if exists)
        if has_parent:
            # Find all leaves in parent direction (excluding current node and its subtree)
            all_leaves = set(tree.get_leaf_names())
            current_subtree_leaves = set(node.get_leaf_names())
            parent_direction_leaves = all_leaves - current_subtree_leaves

            if parent_direction_leaves:
                # Select first in lexicographic order
                representative_leaves.append(sorted(parent_direction_leaves)[0])
            else:
                # Should not happen unless tree has issues
                raise ValueError(f"Node {node} has parent but no leaves in parent direction")

        results.append({
            "node": node,
            "degree": degree,
            "representative_leaves": representative_leaves
        })

    return results
