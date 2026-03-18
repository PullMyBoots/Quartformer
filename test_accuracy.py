#!/usr/bin/env python3
"""
批量测试脚本：运行 run_qf.py 推断所有 ml_rf_real 数据集的树，
并与参考树 tree_best.newick 比较 RF 距离。
"""

import subprocess
import sys
from pathlib import Path
import time

# ============================================================
# 在这里直接设置参数
# ============================================================

# 数据集目录
DATA_DIR = Path("/mnt/c/Users/descfly/Desktop/publish_code/data/ml_rf_real")

# 推断脚本路径
RUN_QF_SCRIPT = Path("/mnt/c/Users/descfly/Desktop/publish_code/run_qf.py")

# 运行参数 (直接写死)
RUN_MODE = "extra_fast"          # "fast" 或 "regular" 或 "slow"
INFER_BATCH_SIZE = 32
K_PARAM = 3.0
TASK_TYPE = "heterogeneous"     # "homogeneous" 或 "heterogeneous"
COMPUTE_BRANCH_SUPPORT = False  # True 时额外输出支持度树与支持度表

# 是否并行运行 (1 为串行，>1 为并行)
MAX_WORKERS = 1

# ============================================================


def run_single_test(data_subdir: Path) -> dict:
    """对单个数据集运行推断并计算 RF 距离"""
    dataset_name = data_subdir.name
    msa_path = data_subdir / "MSA.phy"
    ref_tree_path = data_subdir / "tree_best.newick"
    output_path = data_subdir / f"test_output_{dataset_name}.nwk"

    if not msa_path.exists():
        return {"dataset": dataset_name, "status": "SKIP", "error": "MSA.phy not found"}
    if not ref_tree_path.exists():
        return {"dataset": dataset_name, "status": "SKIP", "error": "tree_best.newick not found"}

    # 构建命令
    cmd = [
        sys.executable,
        str(RUN_QF_SCRIPT),
        "--phy", str(msa_path),
        "--out", str(output_path),
        "--ref-tree", str(ref_tree_path),
        "--metric", "rf",
        "--run-mode", RUN_MODE,
        "--infer-batch-size", str(INFER_BATCH_SIZE),
        "--k-param", str(K_PARAM),
        "--task-type", TASK_TYPE,
    ]
    if COMPUTE_BRANCH_SUPPORT:
        cmd.append("--compute-branch-support")

    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {dataset_name}")
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
                "status": "ERROR",
                "rf_distance": None,
                "elapsed": elapsed,
                "error": full_output[-500:] if full_output else "unknown error"
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
            "status": "OK",
            "rf_distance": rf_distance,
            "elapsed": elapsed,
        }

    except Exception as e:
        return {"dataset": dataset_name, "status": "ERROR", "elapsed": time.time()-start_time, "error": str(e)}


def main():
    if not RUN_QF_SCRIPT.exists():
        print(f"[ERROR] 未找到推断脚本: {RUN_QF_SCRIPT}")
        return 1

    # 查找所有数据集
    datasets = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    if not datasets:
        print(f"[ERROR] 未找到任何数据集: {DATA_DIR}")
        return 1

    print(f"[INFO] 找到 {len(datasets)} 个数据集: {[d.name for d in datasets]}")
    print(f"[INFO] 推断脚本路径: {RUN_QF_SCRIPT}")
    print(f"[INFO] 数据目录: {DATA_DIR}")
    print(
        f"[INFO] 运行参数: run_mode={RUN_MODE}, batch_size={INFER_BATCH_SIZE}, "
        f"k={K_PARAM}, task_type={TASK_TYPE}, compute_branch_support={COMPUTE_BRANCH_SUPPORT}"
    )

    # 运行测试
    results = []
    for d in datasets:
        results.append(run_single_test(d))

    # 打印汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'数据集':<15} {'状态':<10} {'RF距离':<12} {'耗时(s)':<10}")
    print("-"*70)

    total_elapsed = 0
    ok_count = 0
    rf_values = []

    for r in sorted(results, key=lambda x: x.get("dataset", "")):
        dataset = r.get("dataset", "unknown")
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
        print(f"{dataset:<15} {status:<10} {rf_str:<12} {elapsed:<10.1f}")

        if status != "OK":
            error = r.get("error", "")
            if error:
                print(f"  -> Error: {error[:100]}")

    print("-"*70)
    if rf_values:
        avg_rf = sum(rf_values) / len(rf_values)
        print(f"\n[汇总] 成功: {ok_count}/{len(datasets)}")
        print(f"[汇总] 平均 RF 距离: {avg_rf:.4f}")
        print(f"[汇总] 总耗时: {total_elapsed:.1f}s")
    else:
        print(f"\n[汇总] 无有效结果")

    return 0


if __name__ == "__main__":
    sys.exit(main())
