#!/usr/bin/env python3
"""
Grid-search confidence-amplitude margin strength on ml_rf_real datasets.

Data requirement follows test_accuracy.py:
- each dataset under data/ml_rf_real/<dataset_name>/
- needs MSA.phy and tree_best.newick
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# 在这里直接设置参数（不使用 CLI）
# ============================================================
DATA_DIR = Path("/mnt/c/Users/descfly/Desktop/publish_code/data/ml_rf_real")
RUN_QF_SCRIPT = Path("/mnt/c/Users/descfly/Desktop/publish_code/run_qf.py")
RUN_MODE = "regular"  # "fast" / "regular" / "slow"
INFER_BATCH_SIZE = 32
K_PARAM = 3.0
TASK_TYPE = "heterogeneous"  # "homogeneous" / "heterogeneous"
COMPUTE_BRANCH_SUPPORT = False
MARGINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
MAX_DATASETS = 0  # 0 表示全部
QUIET = True  # True 时不打印 run_qf.py 实时日志
SUMMARY_CSV = Path("/mnt/c/Users/descfly/Desktop/publish_code/output/margin_search_summary.csv")
DETAIL_CSV = Path("/mnt/c/Users/descfly/Desktop/publish_code/output/margin_search_detail.csv")
# ============================================================

RF_PATTERN = re.compile(r"RF distance:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class SingleResult:
    dataset: str
    margin: float
    status: str
    rf_distance: float | None
    elapsed: float
    error: str = ""


def format_margin_suffix(v: float) -> str:
    s = f"{v:g}"
    return s.replace(".", "p")


def extract_rf(text: str) -> float | None:
    match = RF_PATTERN.search(text)
    if match:
        return float(match.group(1))
    for line in text.splitlines():
        if "RF distance" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                continue
    return None


def run_single_test(
    data_subdir: Path,
    run_qf_script: Path,
    run_mode: str,
    infer_batch_size: int,
    k_param: float,
    task_type: str,
    compute_branch_support: bool,
    margin: float,
    quiet: bool,
) -> SingleResult:
    dataset_name = data_subdir.name
    msa_path = data_subdir / "MSA.phy"
    ref_tree_path = data_subdir / "tree_best.newick"
    output_path = data_subdir / f"test_output_{dataset_name}_m{format_margin_suffix(margin)}.nwk"

    if not msa_path.exists():
        return SingleResult(dataset=dataset_name, margin=margin, status="SKIP", rf_distance=None, elapsed=0.0, error="MSA.phy not found")
    if not ref_tree_path.exists():
        return SingleResult(dataset=dataset_name, margin=margin, status="SKIP", rf_distance=None, elapsed=0.0, error="tree_best.newick not found")

    cmd = [
        sys.executable,
        str(run_qf_script),
        "--phy",
        str(msa_path),
        "--out",
        str(output_path),
        "--ref-tree",
        str(ref_tree_path),
        "--metric",
        "rf",
        "--run-mode",
        run_mode,
        "--infer-batch-size",
        str(infer_batch_size),
        "--k-param",
        str(k_param),
        "--task-type",
        task_type,
    ]
    if margin > 0:
        cmd.extend(["--enable-confidence-amplitude", f"{margin:g}"])
    if compute_branch_support:
        cmd.append("--compute-branch-support")

    print(f"\n{'=' * 60}")
    print(f"[INFO] dataset={dataset_name}, margin={margin:g}")
    print(f"[INFO] cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if not quiet:
                print(line, end="")
            lines.append(line)
        proc.wait()
        elapsed = time.time() - start_time
        full_output = "".join(lines)

        if proc.returncode != 0:
            return SingleResult(
                dataset=dataset_name,
                margin=margin,
                status="ERROR",
                rf_distance=None,
                elapsed=elapsed,
                error=full_output[-800:] if full_output else "unknown error",
            )

        rf = extract_rf(full_output)
        return SingleResult(
            dataset=dataset_name,
            margin=margin,
            status="OK" if rf is not None else "OK_NO_RF",
            rf_distance=rf,
            elapsed=elapsed,
            error="" if rf is not None else "RF distance not found in output",
        )
    except Exception as exc:
        return SingleResult(
            dataset=dataset_name,
            margin=margin,
            status="ERROR",
            rf_distance=None,
            elapsed=time.time() - start_time,
            error=str(exc),
        )


def write_summary_csv(summary_rows: list[dict], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["margin", "ok_with_rf", "total", "avg_rf", "median_rf", "total_elapsed_s"])
        for row in summary_rows:
            writer.writerow(
                [
                    f"{row['margin']:.8g}",
                    row["ok_with_rf"],
                    row["total"],
                    "" if row["avg_rf"] is None else f"{row['avg_rf']:.8f}",
                    "" if row["median_rf"] is None else f"{row['median_rf']:.8f}",
                    f"{row['total_elapsed']:.3f}",
                ]
            )

def init_detail_csv(detail_csv: Path) -> None:
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    with detail_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["margin", "dataset", "status", "rf_distance", "elapsed_s", "error"])


def append_detail_row(detail_csv: Path, result: SingleResult) -> None:
    with detail_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                f"{result.margin:.8g}",
                result.dataset,
                result.status,
                "" if result.rf_distance is None else f"{result.rf_distance:.8f}",
                f"{result.elapsed:.3f}",
                result.error,
            ]
        )


def median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def build_summary_rows(detail_rows: list[SingleResult], margins: list[float]) -> list[dict]:
    summary_rows: list[dict] = []
    for margin in margins:
        runs = [r for r in detail_rows if r.margin == margin]
        rf_values = [r.rf_distance for r in runs if r.status == "OK" and r.rf_distance is not None]
        total_elapsed = sum(r.elapsed for r in runs)
        avg_rf = (sum(rf_values) / len(rf_values)) if rf_values else None
        med_rf = median(rf_values)
        summary_rows.append(
            {
                "margin": margin,
                "ok_with_rf": len(rf_values),
                "total": len(runs),
                "avg_rf": avg_rf,
                "median_rf": med_rf,
                "total_elapsed": total_elapsed,
            }
        )
    return summary_rows


def main() -> int:
    if not RUN_QF_SCRIPT.exists():
        print(f"[ERROR] run_qf.py not found: {RUN_QF_SCRIPT}")
        return 1
    if not DATA_DIR.exists():
        print(f"[ERROR] data dir not found: {DATA_DIR}")
        return 1

    if any(v < 0 for v in MARGINS):
        print(f"[ERROR] invalid MARGINS (must be >= 0): {MARGINS}")
        return 1
    if not MARGINS:
        print("[ERROR] MARGINS is empty")
        return 1
    if RUN_MODE not in {"fast", "regular", "slow"}:
        print(f"[ERROR] RUN_MODE must be one of fast/regular/slow, got: {RUN_MODE}")
        return 1
    if TASK_TYPE not in {"homogeneous", "heterogeneous"}:
        print(f"[ERROR] TASK_TYPE must be homogeneous/heterogeneous, got: {TASK_TYPE}")
        return 1

    datasets = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    if MAX_DATASETS > 0:
        datasets = datasets[:MAX_DATASETS]
    if not datasets:
        print(f"[ERROR] no dataset directories under: {DATA_DIR}")
        return 1

    print(f"[INFO] datasets={len(datasets)}")
    print(f"[INFO] margins={MARGINS}")
    print(
        f"[INFO] config: run_mode={RUN_MODE}, batch_size={INFER_BATCH_SIZE}, "
        f"k={K_PARAM}, task_type={TASK_TYPE}, compute_branch_support={COMPUTE_BRANCH_SUPPORT}, "
        f"max_datasets={MAX_DATASETS}, quiet={QUIET}"
    )

    init_detail_csv(DETAIL_CSV)
    write_summary_csv(build_summary_rows([], MARGINS), SUMMARY_CSV)

    detail_rows: list[SingleResult] = []
    # 改为: 每个数据集依次跑完全部 margin，再进入下一个数据集
    for idx, d in enumerate(datasets, start=1):
        print(f"\n{'#' * 70}")
        print(f"[DATASET] {idx}/{len(datasets)}: {d.name}")
        print(f"{'#' * 70}")
        for margin in MARGINS:
            result = run_single_test(
                data_subdir=d,
                run_qf_script=RUN_QF_SCRIPT,
                run_mode=RUN_MODE,
                infer_batch_size=INFER_BATCH_SIZE,
                k_param=K_PARAM,
                task_type=TASK_TYPE,
                compute_branch_support=COMPUTE_BRANCH_SUPPORT,
                margin=margin,
                quiet=QUIET,
            )
            detail_rows.append(result)
            append_detail_row(DETAIL_CSV, result)
            write_summary_csv(build_summary_rows(detail_rows, MARGINS), SUMMARY_CSV)

    summary_rows = build_summary_rows(detail_rows, MARGINS)
    for margin, row in zip(MARGINS, summary_rows):
        rf_values_count = row["ok_with_rf"]
        runs_count = row["total"]
        avg_rf = row["avg_rf"]
        total_elapsed = row["total_elapsed"]
        print(
            f"[RESULT] margin={margin:g} | ok={rf_values_count}/{runs_count} | "
            f"avg_rf={('-' if avg_rf is None else f'{avg_rf:.4f}')} | elapsed={total_elapsed:.1f}s"
        )

    valid = [row for row in summary_rows if row["avg_rf"] is not None]
    best = min(valid, key=lambda x: (x["avg_rf"], -x["ok_with_rf"], x["total_elapsed"])) if valid else None

    print("\n" + "=" * 70)
    print("Margin Search Summary")
    print("=" * 70)
    print(f"{'margin':<10} {'ok/total':<10} {'avg_rf':<12} {'median_rf':<12} {'elapsed_s':<10}")
    print("-" * 70)
    for row in sorted(summary_rows, key=lambda x: x["margin"]):
        avg_rf_str = "-" if row["avg_rf"] is None else f"{row['avg_rf']:.4f}"
        med_rf_str = "-" if row["median_rf"] is None else f"{row['median_rf']:.4f}"
        ok_total_str = f"{row['ok_with_rf']}/{row['total']}"
        print(
            f"{row['margin']:<10g} {ok_total_str:<10} "
            f"{avg_rf_str:<12} {med_rf_str:<12} {row['total_elapsed']:<10.1f}"
        )
    if best is None:
        print("[BEST] no valid margin found (no RF parsed).")
    else:
        print(
            f"[BEST] margin={best['margin']:g}, avg_rf={best['avg_rf']:.4f}, "
            f"ok={best['ok_with_rf']}/{best['total']}"
        )

    write_summary_csv(summary_rows, SUMMARY_CSV)
    print(f"[INFO] summary csv: {SUMMARY_CSV}")
    print(f"[INFO] detail csv:  {DETAIL_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
