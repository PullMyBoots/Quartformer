#!/usr/bin/env python3
"""
Render a Newick tree to a static figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import Phylo


def _support_label(clade):
    if clade.is_terminal():
        return None
    confidence = getattr(clade, "confidence", None)
    if confidence is None:
        return None
    text = f"{float(confidence):.2f}"
    return text.rstrip("0").rstrip(".")


def draw_tree(tree_path: Path, out_path: Path, title: str | None = None, show_support: bool = False):
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
        branch_labels=_support_label if show_support else None,
    )

    ax.set_title(title or tree_path.name, fontsize=14, pad=14)
    ax.set_xlabel("Branch length")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Newick tree to a static figure.")
    parser.add_argument("--tree", required=True, help="Input tree path in Newick format")
    parser.add_argument("--out", required=True, help="Output figure path, e.g. .pdf or .png")
    parser.add_argument("--title", default="", help="Optional figure title")
    parser.add_argument("--show-support", action="store_true", help="Draw internal node support values")
    args = parser.parse_args()

    tree_path = Path(args.tree)
    out_path = Path(args.out)
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")

    draw_tree(
        tree_path=tree_path,
        out_path=out_path,
        title=args.title or None,
        show_support=args.show_support,
    )
    print(f"[SUCCESS] Figure saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
