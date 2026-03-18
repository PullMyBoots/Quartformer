#!/usr/bin/env python3
"""
Evaluate whether branch support is reliable:
- higher support should correspond to higher correctness vs reference tree
- incorrect branches should be enriched in the low-support tail
- support should be reasonably calibrated (support% vs empirical accuracy)

Inputs per dataset (default under data/ml_rf_real/<dataset>/):
- reference tree: tree_best.newick
- inferred support tree: test_output_*.support.nwk
"""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ete3 import Tree
from scipy.stats import mannwhitneyu, spearmanr


Q1_PATTERN = re.compile(r"_q1_([0-9]+(?:\.[0-9]+)?)_")


@dataclass
class BranchRecord:
    dataset: str
    branch_id: int
    support: float  # 0..100
    is_correct: bool


def _canonical_split(side: set[str], all_taxa: set[str]) -> tuple[str, ...]:
    comp = all_taxa - side
    left = tuple(sorted(side))
    right = tuple(sorted(comp))
    if len(left) < len(right):
        return left
    if len(right) < len(left):
        return right
    return left if left <= right else right


def _reference_splits(tree_path: Path) -> set[tuple[str, ...]]:
    tree = Tree(str(tree_path), format=1)
    all_taxa = set(tree.get_leaf_names())
    splits: set[tuple[str, ...]] = set()
    for node in tree.traverse():
        if node.is_leaf() or node.is_root():
            continue
        side = set(node.get_leaf_names())
        if len(side) <= 1 or len(side) >= len(all_taxa) - 1:
            continue
        splits.add(_canonical_split(side, all_taxa))
    return splits


def _parse_support_percent(node_name: str | None) -> float | None:
    if node_name is None:
        return None
    s = str(node_name).strip()
    if not s:
        return None

    m = Q1_PATTERN.search(s)
    if m:
        return float(m.group(1)) * 100.0

    # Fallback: node name itself might be numeric support.
    try:
        v = float(s)
        if 0.0 <= v <= 1.0:
            return v * 100.0
        if 0.0 <= v <= 100.0:
            return v
    except ValueError:
        pass
    return None


def _load_branch_records(
    dataset_dir: Path,
    support_glob: str,
    ref_tree_name: str,
) -> list[BranchRecord]:
    ref_tree = dataset_dir / ref_tree_name
    support_candidates = sorted(dataset_dir.glob(support_glob))
    if not ref_tree.exists() or not support_candidates:
        return []

    support_tree = support_candidates[0]
    tree = Tree(str(support_tree), format=1, quoted_node_names=True)
    all_taxa = set(tree.get_leaf_names())
    ref_splits = _reference_splits(ref_tree)
    seen_splits: set[tuple[str, ...]] = set()

    records: list[BranchRecord] = []
    branch_id = 0
    for node in tree.traverse():
        if node.is_leaf() or node.is_root():
            continue
        side = set(node.get_leaf_names())
        if len(side) <= 1 or len(side) >= len(all_taxa) - 1:
            continue

        split = _canonical_split(side, all_taxa)
        if split in seen_splits:
            continue
        seen_splits.add(split)

        support = _parse_support_percent(node.name)
        if support is None:
            continue
        support = min(max(support, 0.0), 100.0)

        branch_id += 1
        records.append(
            BranchRecord(
                dataset=dataset_dir.name,
                branch_id=branch_id,
                support=support,
                is_correct=(split in ref_splits),
            )
        )
    return records


def _auc_correct_above_incorrect(records: list[BranchRecord]) -> float:
    correct = [r.support for r in records if r.is_correct]
    incorrect = [r.support for r in records if not r.is_correct]
    if not correct or not incorrect:
        return float("nan")
    wins = 0.0
    total = len(correct) * len(incorrect)
    for c in correct:
        for ic in incorrect:
            if c > ic:
                wins += 1.0
            elif c == ic:
                wins += 0.5
    return wins / total


def _tail_recall(records: list[BranchRecord], fraction: float) -> tuple[int, int, float]:
    total_incorrect = sum(1 for r in records if not r.is_correct)
    if total_incorrect == 0:
        return 0, 0, float("nan")
    cutoff = max(1, math.ceil(len(records) * fraction))
    ranked = sorted(records, key=lambda r: (r.support, r.branch_id))
    incorrect_in_tail = sum(1 for r in ranked[:cutoff] if not r.is_correct)
    return incorrect_in_tail, total_incorrect, incorrect_in_tail / total_incorrect


def _fraction_to_capture_all_incorrect(records: list[BranchRecord]) -> float:
    total_incorrect = sum(1 for r in records if not r.is_correct)
    if total_incorrect == 0:
        return 0.0
    ranked = sorted(records, key=lambda r: (r.support, r.branch_id))
    seen = 0
    for idx, rec in enumerate(ranked, start=1):
        if not rec.is_correct:
            seen += 1
            if seen == total_incorrect:
                return idx / len(records)
    return 1.0


def _mann_whitney_p(records: list[BranchRecord]) -> float:
    correct = [r.support for r in records if r.is_correct]
    incorrect = [r.support for r in records if not r.is_correct]
    if not correct or not incorrect:
        return float("nan")
    result = mannwhitneyu(correct, incorrect, alternative="greater")
    return float(result.pvalue)


def _spearman_support_correctness(records: list[BranchRecord]) -> tuple[float, float]:
    if not records:
        return float("nan"), float("nan")
    supports = np.array([r.support for r in records], dtype=float)
    labels = np.array([1.0 if r.is_correct else 0.0 for r in records], dtype=float)
    res = spearmanr(supports, labels)
    return float(res.statistic), float(res.pvalue)


def _brier_score(records: list[BranchRecord]) -> float:
    if not records:
        return float("nan")
    p = np.array([r.support / 100.0 for r in records], dtype=float)
    y = np.array([1.0 if r.is_correct else 0.0 for r in records], dtype=float)
    return float(np.mean((p - y) ** 2))


def _calibration_bins(records: list[BranchRecord], num_bins: int = 10) -> tuple[list[dict], float]:
    if not records:
        return [], float("nan")
    supports = np.array([r.support for r in records], dtype=float)
    labels = np.array([1.0 if r.is_correct else 0.0 for r in records], dtype=float)
    edges = np.linspace(0.0, 100.0, num_bins + 1)

    rows: list[dict] = []
    ece = 0.0
    n_total = len(records)
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < num_bins - 1:
            mask = (supports >= lo) & (supports < hi)
        else:
            mask = (supports >= lo) & (supports <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_support = float(supports[mask].mean()) / 100.0
        accuracy = float(labels[mask].mean())
        ece += abs(accuracy - mean_support) * (count / n_total)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_support_prob": mean_support,
                "empirical_accuracy": accuracy,
            }
        )
    return rows, ece


def _write_summary(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "dataset",
        "branches",
        "incorrect",
        "incorrect_rate",
        "mean_support_correct",
        "mean_support_incorrect",
        "support_gap",
        "auc_correct_above_incorrect",
        "pvalue_correct_gt_incorrect",
        "spearman_support_correctness",
        "spearman_pvalue",
        "brier_score",
        "ece",
        "tail10_recall",
        "tail20_recall",
        "tail30_recall",
        "fraction_needed_for_all_incorrect",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_branch_records(records: list[BranchRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["dataset", "branch_id", "support", "is_correct"])
        for r in records:
            writer.writerow([r.dataset, r.branch_id, f"{r.support:.6f}", int(r.is_correct)])


def _plot_tail_curve(records: list[BranchRecord], out_path: Path, title: str) -> None:
    if not records:
        return
    total_incorrect = sum(1 for r in records if not r.is_correct)
    ranked = sorted(records, key=lambda r: (r.support, r.branch_id))
    xs = []
    ys = []
    seen = 0
    for idx, rec in enumerate(ranked, start=1):
        if not rec.is_correct:
            seen += 1
        xs.append(idx / len(ranked))
        ys.append(0.0 if total_incorrect == 0 else seen / total_incorrect)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=220)
    ax.plot(xs, ys, color="#0f766e", linewidth=2.0, label="Observed")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2, label="Random baseline")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Lowest-support branch fraction")
    ax.set_ylabel("Recall of incorrect branches")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(records: list[BranchRecord], out_path: Path, title: str) -> None:
    rows, _ = _calibration_bins(records, num_bins=10)
    if not rows:
        return
    xs = [r["mean_support_prob"] for r in rows]
    ys = [r["empirical_accuracy"] for r in rows]
    sizes = [18 + 3.5 * r["count"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=220)
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.2, label="Perfect calibration")
    ax.scatter(xs, ys, s=sizes, color="#2563eb", alpha=0.8, edgecolor="white", linewidth=0.8, label="Bins")
    ax.plot(xs, ys, color="#1d4ed8", linewidth=1.3, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted support probability")
    ax.set_ylabel("Empirical correctness")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assess branch-support reliability against reference trees."
    )
    parser.add_argument(
        "--data-dir",
        default="data/ml_rf_real",
        help="Directory containing dataset subdirectories",
    )
    parser.add_argument(
        "--support-glob",
        default="test_output_*.support.nwk",
        help="Glob for inferred support tree in each dataset directory",
    )
    parser.add_argument(
        "--ref-tree-name",
        default="tree_best.newick",
        help="Reference tree filename inside each dataset directory",
    )
    parser.add_argument(
        "--out-dir",
        default="data/ml_rf_real/support_reliability",
        help="Output directory for summary tables/figures",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    all_records: list[BranchRecord] = []
    summary_rows: list[dict] = []

    for dataset_dir in dataset_dirs:
        records = _load_branch_records(dataset_dir, args.support_glob, args.ref_tree_name)
        if not records:
            continue
        all_records.extend(records)
        correct_supports = [r.support for r in records if r.is_correct]
        incorrect_supports = [r.support for r in records if not r.is_correct]
        _, _, tail10 = _tail_recall(records, 0.10)
        _, _, tail20 = _tail_recall(records, 0.20)
        _, _, tail30 = _tail_recall(records, 0.30)
        rho, rho_p = _spearman_support_correctness(records)
        _, ece = _calibration_bins(records, num_bins=10)

        mean_correct = (sum(correct_supports) / len(correct_supports)) if correct_supports else float("nan")
        mean_incorrect = (sum(incorrect_supports) / len(incorrect_supports)) if incorrect_supports else float("nan")
        summary_rows.append(
            {
                "dataset": dataset_dir.name,
                "branches": len(records),
                "incorrect": len(incorrect_supports),
                "incorrect_rate": len(incorrect_supports) / len(records),
                "mean_support_correct": mean_correct,
                "mean_support_incorrect": mean_incorrect,
                "support_gap": mean_correct - mean_incorrect if (correct_supports and incorrect_supports) else float("nan"),
                "auc_correct_above_incorrect": _auc_correct_above_incorrect(records),
                "pvalue_correct_gt_incorrect": _mann_whitney_p(records),
                "spearman_support_correctness": rho,
                "spearman_pvalue": rho_p,
                "brier_score": _brier_score(records),
                "ece": ece,
                "tail10_recall": tail10,
                "tail20_recall": tail20,
                "tail30_recall": tail30,
                "fraction_needed_for_all_incorrect": _fraction_to_capture_all_incorrect(records),
            }
        )

        # Per-dataset diagnostic plots
        _plot_tail_curve(
            records,
            out_dir / f"{dataset_dir.name}.tail_recall.png",
            f"{dataset_dir.name}: incorrect-branch recall in low-support tail",
        )
        _plot_calibration(
            records,
            out_dir / f"{dataset_dir.name}.calibration.png",
            f"{dataset_dir.name}: support calibration",
        )

    _write_summary(summary_rows, out_dir / "summary.tsv")
    _write_branch_records(all_records, out_dir / "branch_records.tsv")
    _plot_tail_curve(
        all_records,
        out_dir / "overall.tail_recall.png",
        "Overall: incorrect-branch recall in low-support tail",
    )
    _plot_calibration(
        all_records,
        out_dir / "overall.calibration.png",
        "Overall: support calibration",
    )

    print(f"[SUCCESS] Summary written to {out_dir / 'summary.tsv'}")
    print(f"[SUCCESS] Branch records written to {out_dir / 'branch_records.tsv'}")
    print(f"[SUCCESS] Overall tail-recall plot written to {out_dir / 'overall.tail_recall.png'}")
    print(f"[SUCCESS] Overall calibration plot written to {out_dir / 'overall.calibration.png'}")
    print()
    print(
        "dataset\tbranches\tincorrect\tincorrect_rate\tmean_correct\tmean_incorrect\tgap\tauc\tpvalue\trho\tece\ttail10\ttail20\ttail30\tall_incorrect_fraction"
    )
    for row in summary_rows:
        print(
            f"{row['dataset']}\t{row['branches']}\t{row['incorrect']}\t"
            f"{row['incorrect_rate']:.4f}\t{row['mean_support_correct']:.2f}\t"
            f"{row['mean_support_incorrect']:.2f}\t{row['support_gap']:.2f}\t"
            f"{row['auc_correct_above_incorrect']:.3f}\t"
            f"{row['pvalue_correct_gt_incorrect']:.3e}\t"
            f"{row['spearman_support_correctness']:.3f}\t"
            f"{row['ece']:.3f}\t"
            f"{row['tail10_recall'] if not math.isnan(row['tail10_recall']) else float('nan'):.3f}\t"
            f"{row['tail20_recall'] if not math.isnan(row['tail20_recall']) else float('nan'):.3f}\t"
            f"{row['tail30_recall'] if not math.isnan(row['tail30_recall']) else float('nan'):.3f}\t"
            f"{row['fraction_needed_for_all_incorrect']:.3f}"
        )

    if all_records:
        overall_auc = _auc_correct_above_incorrect(all_records)
        overall_p = _mann_whitney_p(all_records)
        overall_rho, overall_rho_p = _spearman_support_correctness(all_records)
        _, overall_ece = _calibration_bins(all_records, num_bins=10)
        _, _, overall_tail10 = _tail_recall(all_records, 0.10)
        _, _, overall_tail20 = _tail_recall(all_records, 0.20)
        _, _, overall_tail30 = _tail_recall(all_records, 0.30)
        print()
        print(
            "[OVERALL] "
            f"branches={len(all_records)}, incorrect={sum(1 for r in all_records if not r.is_correct)}, "
            f"auc={overall_auc:.3f}, pvalue={overall_p:.3e}, rho={overall_rho:.3f}, rho_p={overall_rho_p:.3e}, "
            f"ece={overall_ece:.3f}, tail10={overall_tail10:.3f}, tail20={overall_tail20:.3f}, tail30={overall_tail30:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
