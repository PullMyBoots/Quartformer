#!/usr/bin/env python3
"""批量测试脚本：可选择运行一个脚本，或运行全部脚本做对照。"""

import subprocess
import sys
from pathlib import Path
import time
from typing import Any

# ============================================================
# 在这里直接设置参数
# ============================================================

# 数据集目录
DATA_DIR = Path("/mnt/c/Users/descfly/Desktop/publish_code/data/ml_rf_real")

# 对照脚本路径
RUN_QF_V2_SCRIPT = Path("/mnt/c/Users/descfly/Desktop/publish_code/run_qf_v2.py")
RUN_QF_BASE_SCRIPT = Path("/mnt/c/Users/descfly/Desktop/publish_code/run_qf.py")
RUN_QF_FAST_SCRIPT = Path("/mnt/c/Users/descfly/Desktop/publish_code/run_qf_fastaggregate.py")
SCRIPT_MAP = {
    "v2": RUN_QF_V2_SCRIPT,
    "base": RUN_QF_BASE_SCRIPT,
    "fastagg": RUN_QF_FAST_SCRIPT,
}
SCRIPT_VARIANTS = [
    ("v2", RUN_QF_V2_SCRIPT),
    ("base", RUN_QF_BASE_SCRIPT),
    ("fastagg", RUN_QF_FAST_SCRIPT),
]
# 选择要测试的脚本: "v2" / "base" / "fastagg" / "all"
SELECT_SCRIPT = "all"

# 运行参数 (直接写死)
RUN_MODE = "fast"          # "fast" 或 "regular" 或 "slow"
INFER_BATCH_SIZE = 32
K_PARAM = 3.0
TASK_TYPE = "homogeneous"     # "homogeneous" 或 "heterogeneous"
COMPUTE_BRANCH_SUPPORT = False  # True 时额外输出支持度树与支持度表
AGGREGATE_MODE = "full"          # "full" / "batch_only" / "off"

# 是否并行运行 (1 为串行，>1 为并行)
MAX_WORKERS = 1

# ============================================================


def run_single_test(data_subdir: Path, script_name: str, script_path: Path) -> dict[str, Any]:
    """对单个数据集、单个脚本运行推断并计算 RF 距离。"""
    dataset_name = data_subdir.name
    msa_path = data_subdir / "MSA.phy"
    ref_tree_path = data_subdir / "tree_best.newick"
    output_path = data_subdir / f"test_output_{dataset_name}.{script_name}.nwk"

    if not msa_path.exists():
        return {
            "dataset": dataset_name,
            "script": script_name,
            "status": "SKIP",
            "error": "MSA.phy not found",
            "output_path": str(output_path),
        }
    if not ref_tree_path.exists():
        return {
            "dataset": dataset_name,
            "script": script_name,
            "status": "SKIP",
            "error": "tree_best.newick not found",
            "output_path": str(output_path),
        }

    # 构建命令
    cmd = [
        sys.executable,
        str(script_path),
        "--phy", str(msa_path),
        "--out", str(output_path),
        "--ref-tree", str(ref_tree_path),
        "--metric", "rf",
        "--run-mode", RUN_MODE,
        "--infer-batch-size", str(INFER_BATCH_SIZE),
        "--k-param", str(K_PARAM),
        "--task-type", TASK_TYPE,
    ]
    if script_name == "fastagg":
        cmd.extend(["--aggregate-mode", AGGREGATE_MODE])
    if COMPUTE_BRANCH_SUPPORT:
        cmd.append("--compute-branch-support")

    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {dataset_name} ({script_name})")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        # 实时显示子进程输出 (继承父进程 stdout)，并捕获输出用于提取 RF 距离
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # 实时打印输出并收集
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # 实时打印
            output_lines.append(line)
        process.wait()
        full_output = ''.join(output_lines)
        elapsed = time.time() - start_time

        if process.returncode != 0:
            return {
                "dataset": dataset_name,
                "script": script_name,
                "status": "ERROR",
                "rf_distance": None,
                "elapsed": elapsed,
                "error": full_output[-500:] if full_output else "unknown error",
                "output_path": str(output_path),
            }

        # 从输出中提取 RF 距离
        rf_distance = None
        for line in full_output.split('\n'):
            if "RF distance" in line:
                try:
                    rf_distance = float(line.split(":")[-1].strip())
                except:
                    pass

        return {
            "dataset": dataset_name,
            "script": script_name,
            "status": "OK",
            "rf_distance": rf_distance,
            "elapsed": elapsed,
            "output_path": str(output_path),
        }

    except Exception as e:
        return {
            "dataset": dataset_name,
            "script": script_name,
            "status": "ERROR",
            "elapsed": time.time() - start_time,
            "error": str(e),
            "output_path": str(output_path),
        }


def main():
    if SELECT_SCRIPT == "all":
        selected_variants = SCRIPT_VARIANTS
    else:
        if SELECT_SCRIPT not in SCRIPT_MAP:
            print(f"[ERROR] SELECT_SCRIPT 无效: {SELECT_SCRIPT}，可选值: v2/base/fastagg/all")
            return 1
        selected_variants = [(SELECT_SCRIPT, SCRIPT_MAP[SELECT_SCRIPT])]

    for script_name, script_path in selected_variants:
        if not script_path.exists():
            print(f"[ERROR] 未找到推断脚本({script_name}): {script_path}")
            return 1

    # 查找所有数据集
    datasets = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    if not datasets:
        print(f"[ERROR] 未找到任何数据集: {DATA_DIR}")
        return 1

    print(f"[INFO] 找到 {len(datasets)} 个数据集: {[d.name for d in datasets]}")
    print(f"[INFO] 选择脚本: {SELECT_SCRIPT}")
    print("[INFO] 实际运行脚本:")
    for script_name, script_path in selected_variants:
        print(f"  - {script_name}: {script_path}")
    print(f"[INFO] 数据目录: {DATA_DIR}")
    print(
        f"[INFO] 运行参数: run_mode={RUN_MODE}, batch_size={INFER_BATCH_SIZE}, "
        f"k={K_PARAM}, task_type={TASK_TYPE}, aggregate_mode={AGGREGATE_MODE}, "
        f"compute_branch_support={COMPUTE_BRANCH_SUPPORT}"
    )

    # 运行测试
    results = []
    for d in datasets:
        for script_name, script_path in selected_variants:
            results.append(run_single_test(d, script_name, script_path))

    # 打印汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'数据集':<15} {'脚本':<10} {'状态':<10} {'RF距离':<12} {'耗时(s)':<10}")
    print("-"*70)

    total_elapsed = 0
    ok_count = 0
    rf_values = []

    for r in sorted(results, key=lambda x: x.get("dataset", "")):
        dataset = r.get("dataset", "unknown")
        script_name = r.get("script", "unknown")
        status = r.get("status", "ERROR")
        rf = r.get("rf_distance")
        elapsed = r.get("elapsed", 0)

        if status == "OK" and rf is not None:
            rf_str = f"{rf:.4f}"
            rf_values.append(rf)
            ok_count += 1
        else:
            rf_str = "-"

        total_elapsed += elapsed
        print(f"{dataset:<15} {script_name:<10} {status:<10} {rf_str:<12} {elapsed:<10.1f}")

        if status != "OK":
            error = r.get("error", "")
            if error:
                print(f"  -> Error: {error[:100]}")

    print("-"*70)

    if SELECT_SCRIPT == "all":
        print("\n[INFO] 已完成 all 模式：若需脚本间树差异对照，请使用旧版对照脚本或后续再加 compare 选项。")

    if rf_values:
        avg_rf = sum(rf_values) / len(rf_values)
        print(f"\n[汇总] 成功: {ok_count}/{len(results)}")
        print(f"[汇总] 平均 RF 距离: {avg_rf:.4f}")
        print(f"[汇总] 总耗时: {total_elapsed:.1f}s")
    else:
        print(f"\n[汇总] 无有效结果")

    return 0


if __name__ == "__main__":
    sys.exit(main())
