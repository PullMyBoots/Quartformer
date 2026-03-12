#!/usr/bin/env python3
"""
QF2 性能基准测试脚本
测试不同物种数和序列长度的推断运行时间
"""
import sys
import time
import csv
from pathlib import Path

# 添加 cpp_source 相关目录到模块搜索路径
cpp_source_path = Path(__file__).parent / "cpp_source"
if str(cpp_source_path) not in sys.path:
    sys.path.insert(0, str(cpp_source_path))
cuda_pattern_path = cpp_source_path / "cuda13_pattern_freq"
if str(cuda_pattern_path) not in sys.path:
    sys.path.insert(0, str(cuda_pattern_path))

from run_qf_copy import run_qf


def run_benchmark(
    species_list=None,
    dna_len_list=None,
    task_type="gene",
    k_param=3.0,
    run_mode="regular",
    infer_batch_size=32,
    cleanup=True,
    output_csv="output/benchmark_results.csv"
):
    """
    运行性能基准测试

    参数:
        species_list: 要测试的物种数列表，如 [24, 48, 96]
        dna_len_list: 要测试的DNA长度列表，如 [100000, 1000000]
        task_type: 任务类型，"multilocus" 或 "gene"
        k_param: k参数，用于block采样
        run_mode: "fast" 或 "regular"
        infer_batch_size: 推断批次大小
        cleanup: 是否清理临时文件
        output_csv: 结果CSV输出路径
    """
    if species_list is None:
        species_list = [24, 48, 96, 192]
    if dna_len_list is None:
        dna_len_list = [100000, 1000000, 10000000]

    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化CSV文件
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "species",
                "dna_len",
                "task_type",
                "k_param",
                "run_mode",
                "infer_batch_size",
                "time_s",
                "output_tree",
                "status"
            ])

    # 统计信息
    total_tests = len(species_list) * len(dna_len_list)
    completed = 0
    failed = 0

    print("=" * 80)
    print(f"QF2 性能基准测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  物种数: {species_list}")
    print(f"  DNA长度: {dna_len_list}")
    print(f"  任务类型: {task_type}")
    print(f"  k参数: {k_param}")
    print(f"  运行模式: {run_mode}")
    print(f"  推理批次大小: {infer_batch_size}")
    print(f"  总测试数: {total_tests}")
    print("=" * 80)
    print()

    # 遍历所有测试组合
    for species_num in species_list:
        for dna_len in dna_len_list:
            test_name = f"species={species_num}, dna_len={dna_len}"
            print(f"\n{'='*80}")
            print(f"[{completed+failed+1}/{total_tests}] 测试: {test_name}")
            print(f"{'='*80}")

            # 检查输入文件是否存在
            phy_path = Path(f"data/{species_num}/0/GTR_{dna_len}_MSA.phy")
            if not phy_path.exists():
                print(f"[SKIP] 输入文件不存在: {phy_path}")
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        species_num, dna_len, task_type, k_param, run_mode,
                        infer_batch_size, 0, "", "SKIP_FILE_NOT_FOUND"
                    ])
                failed += 1
                continue

            # 输出树路径
            out_path = f"output/benchmark_tree_{species_num}_{dna_len}_{run_mode}.nwk"

            # 运行测试
            try:
                print(f"[INFO] 开始推断...")
                t0 = time.perf_counter()

                result_tree = run_qf(
                    phy_path=str(phy_path),
                    output_tree_path=out_path,
                    task_type=task_type,
                    k_param=k_param,
                    cleanup_temp_files=cleanup,
                    run_mode=run_mode,
                    infer_batch_size=infer_batch_size,
                )

                elapsed = time.perf_counter() - t0

                if result_tree:
                    print(f"[SUCCESS] 完成！耗时: {elapsed:.3f}秒")
                    print(f"[INFO] 输出树: {result_tree}")
                    status = "SUCCESS"
                    completed += 1
                else:
                    print(f"[FAILED] 推断失败")
                    elapsed = 0
                    status = "FAILED"
                    failed += 1

            except Exception as e:
                elapsed = 0
                result_tree = ""
                status = f"ERROR: {str(e)[:100]}"
                print(f"[ERROR] {e}")
                failed += 1

            # 写入结果
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    species_num, dna_len, task_type, k_param, run_mode,
                    infer_batch_size, f"{elapsed:.3f}", result_tree, status
                ])

    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {total_tests}")
    print(f"成功: {completed}")
    print(f"失败: {failed}")
    print(f"跳过: {total_tests - completed - failed}")
    print(f"结果已保存至: {csv_path}")
    print("=" * 80)

    return completed, failed


if __name__ == "__main__":
    # ==================== 配置参数（在此修改）====================

    # 要测试的物种数列表
    SPECIES_LIST = [24, 96, 320]

    # 要测试的DNA长度列表
    DNA_LEN_LIST = [1000000]

    # 运行模式: "fast" (只保留top1) 或 "regular" (保留全部拓扑)
    RUN_MODE = "regular"

    # 任务类型: "heterogeneous" 或 "homogeneous"
    TASK_TYPE = "homogeneous"

    # k参数（用于block采样）
    K_PARAM = 3

    # 推断批次大小
    INFER_BATCH_SIZE = 16

    # 是否清理临时文件
    CLEANUP = True

    # 结果输出路径
    OUTPUT_CSV = "benchmark_results.csv"

    # ===========================================================

    run_benchmark(
        species_list=SPECIES_LIST,
        dna_len_list=DNA_LEN_LIST,
        task_type=TASK_TYPE,
        k_param=K_PARAM,
        run_mode=RUN_MODE,
        infer_batch_size=INFER_BATCH_SIZE,
        cleanup=CLEANUP,
        output_csv=OUTPUT_CSV
    )


    
