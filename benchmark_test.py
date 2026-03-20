#!/usr/bin/env python3
"""
QF2 性能基准测试脚本
测试不同物种数和序列长度的推断运行时间
"""
import sys
import time
import csv
import os
import re
from pathlib import Path

from infer_tree import run_qf


def _load_config_ranges(config_dir: Path) -> list[tuple[int, int, Path]]:
    ranges: list[tuple[int, int, Path]] = []
    if not config_dir.exists():
        return ranges
    for p in sorted(config_dir.glob("*.jsonc")):
        m = re.fullmatch(r"(\d+)\s*~\s*(\d+)\.jsonc", p.name)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        ranges.append((lo, hi, p))
    return ranges


def _select_config_for_species(
    species_num: int,
    config_ranges: list[tuple[int, int, Path]],
    fallback_config: Path,
) -> Path:
    for lo, hi, cfg in config_ranges:
        if lo <= species_num <= hi:
            return cfg
    return fallback_config


def run_benchmark(
    species_list=None,
    dna_len_list=None,
    task_type="heterogeneous",
    infer_batch_size=32,
    compute_branch_support=False,
    output_csv="output/benchmark_results.csv",
    config_path="infer_config.jsonc",
    config_dir="config_benchmark",
):
    """
    运行性能基准测试

    参数:
        species_list: 要测试的物种数列表，如 [24, 48, 96]
        dna_len_list: 要测试的DNA长度列表，如 [100000, 1000000]
        task_type: 任务类型，"homogeneous" 或 "heterogeneous"
        infer_batch_size: 推断批次大小
        compute_branch_support: 是否计算分枝支持度
        output_csv: 结果CSV输出路径
        config_path: infer_tree 默认配置 JSONC 路径（区间未命中时使用）
        config_dir: 区间配置目录，文件名格式为 <min>~<max>.jsonc
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
    fallback_config = Path(config_path).resolve()
    config_ranges = _load_config_ranges(Path(config_dir).resolve())

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "species",
                "dna_len",
                "task_type",
                "infer_batch_size",
                "compute_branch_support",
                "config_file",
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
    print(f"  推理批次大小: {infer_batch_size}")
    print(f"  计算支持度: {compute_branch_support}")
    print(f"  默认配置: {fallback_config}")
    print(f"  分段配置目录: {Path(config_dir).resolve()}")
    if config_ranges:
        print("  已识别配置区间:")
        for lo, hi, cfg in config_ranges:
            print(f"    - {lo}~{hi}: {cfg}")
    else:
        print("  [WARN] 未识别到区间配置，将统一使用默认配置")
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
            selected_config = _select_config_for_species(
                species_num=species_num,
                config_ranges=config_ranges,
                fallback_config=fallback_config,
            )
            if not selected_config.exists():
                print(f"[SKIP] 配置文件不存在: {selected_config}")
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        species_num, dna_len, task_type,
                        infer_batch_size, compute_branch_support,
                        str(selected_config), 0, "", "SKIP_CONFIG_NOT_FOUND"
                    ])
                failed += 1
                continue
            if not phy_path.exists():
                print(f"[SKIP] 输入文件不存在: {phy_path}")
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        species_num, dna_len, task_type,
                        infer_batch_size, compute_branch_support,
                        str(selected_config), 0, "", "SKIP_FILE_NOT_FOUND"
                    ])
                failed += 1
                continue

            # 输出树路径
            out_path = f"output/benchmark_tree_{species_num}_{dna_len}.nwk"

            # 运行测试
            try:
                print(f"[INFO] 开始推断... 使用配置: {selected_config}")
                t0 = time.perf_counter()
                os.environ["QF_INFER_CONFIG"] = str(selected_config)

                result_tree = run_qf(
                    phy_path=str(phy_path),
                    output_tree_path=out_path,
                    task_type=task_type,
                    infer_batch_size=infer_batch_size,
                    compute_branch_support=compute_branch_support,
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
                    species_num, dna_len, task_type,
                    infer_batch_size, compute_branch_support,
                    str(selected_config), f"{elapsed:.3f}", result_tree, status
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
    SPECIES_LIST = [1024]

    # 要测试的DNA长度列表
    DNA_LEN_LIST = [100000]

    # 任务类型: "heterogeneous" 或 "homogeneous"
    TASK_TYPE = "homogeneous"

    # 推断批次大小
    INFER_BATCH_SIZE = 8

    # 是否计算分枝支持度
    COMPUTE_BRANCH_SUPPORT = False

    # 默认细粒度参数（quartet_assembler/qmc_iter_limit/k_param/aggregate_mode等）
    # 请修改 infer_config.jsonc；分段覆盖请修改 config_benchmark/*.jsonc
    CONFIG_PATH = "infer_config.jsonc"
    CONFIG_DIR = "config_benchmark"

    # 结果输出路径
    OUTPUT_CSV = "benchmark_results.csv"

    # ===========================================================

    run_benchmark(
        species_list=SPECIES_LIST,
        dna_len_list=DNA_LEN_LIST,
        task_type=TASK_TYPE,
        infer_batch_size=INFER_BATCH_SIZE,
        compute_branch_support=COMPUTE_BRANCH_SUPPORT,
        output_csv=OUTPUT_CSV,
        config_path=CONFIG_PATH,
        config_dir=CONFIG_DIR,
    )


    
