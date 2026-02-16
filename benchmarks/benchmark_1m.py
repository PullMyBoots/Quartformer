#!/usr/bin/env python3
"""
Benchmark QuartFormer on 1,000,000-site datasets across taxa scales.

Compares:
- two configurable runner scripts (default: run_qf.py vs run_qf.py)

Metrics:
- wall-clock runtime (seconds)
- RF distance to reference tree
- RF distance between the two outputs
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import compute_tree_difference


def run_cmd(cmd: list[str]) -> float:
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark two QuartFormer runner scripts on fixed dataset scales."
    )
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument(
        "--taxa",
        nargs="+",
        type=int,
        default=[24, 48, 96, 192],
        help="Taxa sizes to benchmark",
    )
    parser.add_argument("--k-param", type=float, default=3.0, help="k parameter for inference")
    parser.add_argument("--infer-batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument(
        "--run-mode",
        choices=["fast", "regular"],
        default="fast",
        help="Run mode passed to both scripts",
    )
    parser.add_argument(
        "--script-a",
        default="run_qf.py",
        help="First runner script path",
    )
    parser.add_argument(
        "--script-b",
        default="run_qf.py",
        help="Second runner script path",
    )
    parser.add_argument(
        "--sites",
        type=int,
        default=1_000_000,
        choices=[100_000, 1_000_000, 10_000_000],
        help="Sequence length used to select GTR_<sites>_MSA.phy",
    )
    parser.add_argument(
        "--output-dir",
        default="output/bench_1m",
        help="Directory to store output trees and summary",
    )
    parser.add_argument(
        "--csv-name",
        default="benchmark_1m_summary.csv",
        help="CSV file name under output-dir",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name

    rows: list[dict[str, str | int | float]] = []

    for taxa in args.taxa:
        msa = data_root / str(taxa) / "0" / f"GTR_{args.sites}_MSA.phy"
        ref_tree = data_root / str(taxa) / "0" / "tree.newick"
        if not msa.exists() or not ref_tree.exists():
            print(f"[SKIP] missing files for taxa={taxa}: {msa} / {ref_tree}")
            continue

        a_name = Path(args.script_a).stem
        b_name = Path(args.script_b).stem
        out_a = out_dir / f"{a_name}_{taxa}_{args.sites}.nwk"
        out_b = out_dir / f"{b_name}_{taxa}_{args.sites}.nwk"

        cmd_a = [
            sys.executable,
            args.script_a,
            "--phy",
            str(msa),
            "--out",
            str(out_a),
            "--run-mode",
            args.run_mode,
            "--k-param",
            str(args.k_param),
            "--infer-batch-size",
            str(args.infer_batch_size),
            "--task-type",
            "homogeneous",
        ]
        cmd_b = [
            sys.executable,
            args.script_b,
            "--phy",
            str(msa),
            "--out",
            str(out_b),
            "--run-mode",
            args.run_mode,
            "--k-param",
            str(args.k_param),
            "--infer-batch-size",
            str(args.infer_batch_size),
            "--task-type",
            "homogeneous",
        ]

        print(f"\n[RUN] taxa={taxa} -> {args.script_a}")
        try:
            t_a = run_cmd(cmd_a)
        except Exception as exc:
            print(f"[ERROR] {args.script_a} failed for taxa={taxa}: {exc}")
            continue
        print(f"[DONE] {args.script_a} taxa={taxa} time={t_a:.3f}s")

        print(f"[RUN] taxa={taxa} -> {args.script_b}")
        try:
            t_b = run_cmd(cmd_b)
        except Exception as exc:
            print(f"[ERROR] {args.script_b} failed for taxa={taxa}: {exc}")
            continue
        print(f"[DONE] {args.script_b} taxa={taxa} time={t_b:.3f}s")

        rf_a = compute_tree_difference(out_a, ref_tree, mode="rf")
        rf_b = compute_tree_difference(out_b, ref_tree, mode="rf")
        rf_between = compute_tree_difference(out_a, out_b, mode="rf")

        row = {
            "taxa": taxa,
            "sites": args.sites,
            "run_mode": args.run_mode,
            "script_a": args.script_a,
            "script_b": args.script_b,
            "time_script_a_sec": round(t_a, 6),
            "time_script_b_sec": round(t_b, 6),
            "speedup_a_over_b": round((t_a / t_b) if t_b > 0 else 0.0, 6),
            "rf_script_a_vs_ref": round(float(rf_a), 6),
            "rf_script_b_vs_ref": round(float(rf_b), 6),
            "rf_script_a_vs_script_b": round(float(rf_between), 6),
            "same_rf_to_ref": abs(float(rf_a) - float(rf_b)) < 1e-12,
            "same_tree_output": abs(float(rf_between)) < 1e-12,
        }
        rows.append(row)
        print(f"[RF] taxa={taxa} a={rf_a:.6f} b={rf_b:.6f} a_vs_b={rf_between:.6f}")

    if not rows:
        print("No benchmark rows produced.")
        return 1

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SUMMARY] wrote CSV: {csv_path}")
    print("taxa | a_time(s) | b_time(s) | rf(a,ref) | rf(b,ref) | rf(a,b)")
    for r in rows:
        print(
            f"{r['taxa']:>4} | "
            f"{r['time_script_a_sec']:>9} | "
            f"{r['time_script_b_sec']:>9} | "
            f"{r['rf_script_a_vs_ref']:>9} | "
            f"{r['rf_script_b_vs_ref']:>9} | "
            f"{r['rf_script_a_vs_script_b']:>7}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
