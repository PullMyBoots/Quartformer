#!/usr/bin/env python3
"""
测试 run_qf_copy.py 中 fast 和 regular 模式的准确性和速度对比
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


def test_fast_vs_regular(
    species_list=None,
    dna_len_list=None,
    task_type="homogeneous",
    k_param=3.0,
    infer_batch_size=32,
    cleanup=True,
    output_csv="output/fast_vs_regular_results.csv"
):
    """
    测试 fast 和 regular 模式的性能和准确性对比

    参数:
        species_list: 要测试的物种数列表
        dna_len_list: 要测试的DNA长度列表
        task_type: 任务类型
        k_param: k参数
        infer_batch_size: 推断批次大小
        cleanup: 是否清理临时文件
        output_csv: 结果CSV输出路径
    """
    if species_list is None:
        species_list = [24, 48, 96]
    if dna_len_list is None:
        dna_len_list = [100000, 1000000]

    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化CSV文件
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            "species",
            "dna_len",
            "time_s",
            "rf_distance",
            "rf_accuracy",
            "quartet_distance",
            "quartet_accuracy",
            "output_tree",
            "status"
        ])

    # 统计信息
    total_tests = len(species_list) * len(dna_len_list) * 2  # fast + regular
    completed = 0
    failed = 0

    print("=" * 100)
    print(f"Fast vs Regular 模式对比测试")
    print("=" * 100)
    print(f"配置:")
    print(f"  物种数: {species_list}")
    print(f"  DNA长度: {dna_len_list}")
    print(f"  任务类型: {task_type}")
    print(f"  k参数: {k_param}")
    print(f"  推理批次大小: {infer_batch_size}")
    print(f"  测试模式: fast vs regular")
    print(f"  总测试数: {total_tests}")
    print("=" * 100)
    print()

    # 遍历所有测试组合
    for species_num in species_list:
        for dna_len in dna_len_list:
            # 检查输入文件和参考树是否存在
            phy_path = Path(f"data/{species_num}/0/GTR_{dna_len}_MSA.phy")
            ref_tree_path = Path(f"data/{species_num}/0/tree.newick")

            if not phy_path.exists():
                print(f"[SKIP] 输入文件不存在: {phy_path}")
                failed += 2
                continue

            has_ref_tree = ref_tree_path.exists()
            if not has_ref_tree:
                print(f"[WARN] 参考树不存在: {ref_tree_path}，将跳过准确性评估")

            print(f"\n{'='*100}")
            print(f"测试数据: species={species_num}, dna_len={dna_len}")
            print(f"{'='*100}")

            # 测试两种模式
            modes = ["fast", "regular"]

            for mode in modes:
                test_count = completed + failed + 1
                print(f"\n[{test_count}/{total_tests}] 测试: {mode} 模式")
                print("-" * 100)

                # 输出树路径
                out_path = f"output/test_{mode}_tree_{species_num}_{dna_len}.nwk"

                # 运行测试
                try:
                    print(f"[INFO] 开始推断...")
                    t0 = time.perf_counter()

                    if has_ref_tree:
                        # 使用内置metric接口
                        result_tree, rf_metric = run_qf(
                            phy_path=str(phy_path),
                            output_tree_path=out_path,
                            task_type=task_type,
                            k_param=k_param,
                            cleanup_temp_files=cleanup,
                            run_mode=mode,
                            infer_batch_size=infer_batch_size,
                            ref_tree_path=str(ref_tree_path),
                            metric="rf"
                        )

                        # 再测试Quartet指标
                        try:
                            _, q_metric = run_qf(
                                phy_path=str(phy_path),
                                output_tree_path=out_path,
                                task_type=task_type,
                                k_param=k_param,
                                cleanup_temp_files=False,
                                run_mode=mode,
                                infer_batch_size=infer_batch_size,
                                ref_tree_path=str(ref_tree_path),
                                metric="quartet"
                            )
                        except:
                            q_metric = None
                    else:
                        # 没有参考树，只运行推断
                        result_tree = run_qf(
                            phy_path=str(phy_path),
                            output_tree_path=out_path,
                            task_type=task_type,
                            k_param=k_param,
                            cleanup_temp_files=cleanup,
                            run_mode=mode,
                            infer_batch_size=infer_batch_size,
                        )
                        rf_metric = None
                        q_metric = None

                    elapsed = time.perf_counter() - t0

                    if not result_tree:
                        print(f"[FAILED] 推断失败")
                        with open(csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                mode, species_num, dna_len, 0,
                                "", "", "", "", "", "FAILED"
                            ])
                        failed += 1
                        continue

                    print(f"[SUCCESS] 完成！耗时: {elapsed:.3f}秒")
                    print(f"[INFO] 输出树: {result_tree}")

                    # 打印metric结果
                    if rf_metric is not None:
                        rf_acc = 1.0 - rf_metric
                        print(f"[INFO] RF距离: {rf_metric:.6f}, RF准确率: {rf_acc:.6f}")
                    else:
                        rf_acc = None

                    if q_metric is not None:
                        q_acc = 1.0 - q_metric
                        print(f"[INFO] Quartet距离: {q_metric:.6f}, Quartet准确率: {q_acc:.6f}")
                    else:
                        q_acc = None

                    # 写入结果
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            mode, species_num, dna_len, f"{elapsed:.3f}",
                            f"{rf_metric:.6f}" if rf_metric is not None else "",
                            f"{rf_acc:.6f}" if rf_acc is not None else "",
                            f"{q_metric:.6f}" if q_metric is not None else "",
                            f"{q_acc:.6f}" if q_acc is not None else "",
                            result_tree, "SUCCESS"
                        ])

                    completed += 1

                except Exception as e:
                    elapsed = 0
                    print(f"[ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1

                    # 写入错误结果
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            mode, species_num, dna_len, f"{elapsed:.3f}",
                            "", "", "", "", f"ERROR: {str(e)[:50]}", "ERROR"
                        ])

    # 打印总结
    print("\n" + "=" * 100)
    print("测试总结")
    print("=" * 100)
    print(f"总测试数: {total_tests}")
    print(f"成功: {completed}")
    print(f"失败: {failed}")
    print(f"结果已保存至: {csv_path}")
    print("=" * 100)

    # 打印模式对比摘要
    print("\n模式对比摘要:")
    print("-" * 100)

    # 读取CSV并统计
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            results = list(reader)

        # 按模式分组
        mode_stats = {
            "fast": {"time": [], "rf_acc": [], "q_acc": []},
            "regular": {"time": [], "rf_acc": [], "q_acc": []}
        }

        for row in results:
            mode = row["mode"]
            if row["status"] == "SUCCESS":
                if row["time_s"]:
                    mode_stats[mode]["time"].append(float(row["time_s"]))
                if row["rf_accuracy"]:
                    mode_stats[mode]["rf_acc"].append(float(row["rf_accuracy"]))
                if row["quartet_accuracy"]:
                    mode_stats[mode]["q_acc"].append(float(row["quartet_accuracy"]))

        # 打印统计
        for mode in ["fast", "regular"]:
            stats = mode_stats[mode]
            if stats["time"]:
                avg_time = sum(stats["time"]) / len(stats["time"])
                avg_rf = sum(stats["rf_acc"]) / len(stats["rf_acc"]) if stats["rf_acc"] else 0
                avg_q = sum(stats["q_acc"]) / len(stats["q_acc"]) if stats["q_acc"] else 0

                print(f"\n{mode.upper()} 模式:")
                print(f"  测试数: {len(stats['time'])}")
                print(f"  平均耗时: {avg_time:.3f}秒")
                print(f"  平均RF准确率: {avg_rf:.6f}")
                print(f"  平均Quartet准确率: {avg_q:.6f}")

        # 对比分析
        if mode_stats["fast"]["time"] and mode_stats["regular"]["time"]:
            fast_time = sum(mode_stats["fast"]["time"]) / len(mode_stats["fast"]["time"])
            reg_time = sum(mode_stats["regular"]["time"]) / len(mode_stats["regular"]["time"])
            speedup = reg_time / fast_time

            fast_rf = sum(mode_stats["fast"]["rf_acc"]) / len(mode_stats["fast"]["rf_acc"]) if mode_stats["fast"]["rf_acc"] else 0
            reg_rf = sum(mode_stats["regular"]["rf_acc"]) / len(mode_stats["regular"]["rf_acc"]) if mode_stats["regular"]["rf_acc"] else 0
            rf_diff = fast_rf - reg_rf

            print(f"\n对比分析:")
            print(f"  速度提升: {speedup:.2f}x (fast比regular快{speedup:.2f}倍)")
            print(f"  RF准确率差异: {rf_diff:+.6f} ({'fast更高' if rf_diff > 0 else 'regular更高'})")

    except Exception as e:
        print(f"[ERROR] 生成统计摘要失败: {e}")

    print("=" * 100)

    return completed, failed


if __name__ == "__main__":
    # ==================== 配置参数（在此修改）====================

    # 要测试的物种数列表
    SPECIES_LIST = [24, 48, 96, 128]

    # 要测试的DNA长度列表
    DNA_LEN_LIST = [100000, 1000000]

    # 任务类型: "homogeneous" 或 "heterogeneous"
    TASK_TYPE = "homogeneous"

    # k参数（用于block采样）
    K_PARAM = 3.0

    # 推断批次大小
    INFER_BATCH_SIZE = 32

    # 是否清理临时文件
    CLEANUP = True

    # 结果输出路径
    OUTPUT_CSV = "fast_vs_regular_results.csv"

    # ===========================================================

    test_fast_vs_regular(
        species_list=SPECIES_LIST,
        dna_len_list=DNA_LEN_LIST,
        task_type=TASK_TYPE,
        k_param=K_PARAM,
        infer_batch_size=INFER_BATCH_SIZE,
        cleanup=CLEANUP,
        output_csv=OUTPUT_CSV
    )
