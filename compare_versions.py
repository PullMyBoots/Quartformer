#!/usr/bin/env python3
"""
对比测试 run_qf.py 和 run_qf_copy.py 的推断准确性
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

# 导入两个版本的函数
import importlib.util
spec1 = importlib.util.spec_from_file_location("run_qf_original", Path(__file__).parent / "run_qf.py")
run_qf_module1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(run_qf_module1)

spec2 = importlib.util.spec_from_file_location("run_qf_copy", Path(__file__).parent / "run_qf_copy.py")
run_qf_module2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(run_qf_module2)



def run_comparison_test(
    species_list=None,
    dna_len_list=None,
    task_type="homogeneous",
    k_param=3.0,
    run_mode="regular",
    infer_batch_size=32,
    cleanup=True,
    output_csv="output/compare_results.csv"
):
    """
    运行两个版本的对比测试

    参数:
        species_list: 要测试的物种数列表
        dna_len_list: 要测试的DNA长度列表
        task_type: 任务类型
        k_param: k参数
        run_mode: 运行模式
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
            "version",
            "species",
            "dna_len",
            "task_type",
            "k_param",
            "run_mode",
            "time_s",
            "rf_distance",
            "rf_accuracy",
            "quartet_distance",
            "quartet_accuracy",
            "output_tree",
            "status"
        ])

    # 统计信息
    total_tests = len(species_list) * len(dna_len_list) * 2  # 两个版本
    completed = 0
    failed = 0

    print("=" * 100)
    print(f"QF 版本对比测试")
    print("=" * 100)
    print(f"配置:")
    print(f"  物种数: {species_list}")
    print(f"  DNA长度: {dna_len_list}")
    print(f"  任务类型: {task_type}")
    print(f"  k参数: {k_param}")
    print(f"  运行模式: {run_mode}")
    print(f"  推理批次大小: {infer_batch_size}")
    print(f"  对比版本:")
    print(f"    Version 1: run_qf.py (原始版本)")
    print(f"    Version 2: run_qf_copy.py (修改版本)")
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

            if not ref_tree_path.exists():
                print(f"[WARN] 参考树不存在: {ref_tree_path}，将跳过准确性评估")
                ref_tree_path = None

            print(f"\n{'='*100}")
            print(f"测试数据: species={species_num}, dna_len={dna_len}")
            print(f"{'='*100}")

            # 测试两个版本
            versions = [
                ("run_qf.py (原始版)", run_qf_module1.run_qf, "original"),
                ("run_qf_copy.py (修改版)", run_qf_module2.run_qf, "copy")
            ]

            for version_name, run_qf_func, version_id in versions:
                test_count = completed + failed + 1
                print(f"\n[{test_count}/{total_tests}] 测试: {version_name}")
                print("-" * 100)

                # 输出树路径
                out_path = f"output/compare_{version_id}_tree_{species_num}_{dna_len}_{run_mode}.nwk"

                # 运行测试（如果有参考树，使用内置metric接口）
                try:
                    print(f"[INFO] 开始推断...")
                    t0 = time.perf_counter()

                    if ref_tree_path:
                        # 使用内置metric接口，分别测试RF和Quartet
                        result = run_qf_func(
                            phy_path=str(phy_path),
                            output_tree_path=out_path,
                            task_type=task_type,
                            k_param=k_param,
                            cleanup_temp_files=cleanup,
                            run_mode=run_mode,
                            infer_batch_size=infer_batch_size,
                            ref_tree_path=str(ref_tree_path),
                            metric="rf"  # 先用RF评估
                        )

                        if isinstance(result, tuple) and len(result) == 2:
                            result_tree, rf_metric = result
                        else:
                            result_tree, rf_metric = result, None

                        # 再用Quartet评估
                        try:
                            _, q_metric = run_qf_func(
                                phy_path=str(phy_path),
                                output_tree_path=out_path,  # 使用已有的树文件
                                task_type=task_type,
                                k_param=k_param,
                                cleanup_temp_files=False,  # 不清理，保留树文件
                                run_mode=run_mode,
                                infer_batch_size=infer_batch_size,
                                ref_tree_path=str(ref_tree_path),
                                metric="quartet"
                            )
                        except:
                            q_metric = None
                    else:
                        # 没有参考树，只运行推断
                        result_tree = run_qf_func(
                            phy_path=str(phy_path),
                            output_tree_path=out_path,
                            task_type=task_type,
                            k_param=k_param,
                            cleanup_temp_files=cleanup,
                            run_mode=run_mode,
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
                                version_name, species_num, dna_len, task_type,
                                k_param, run_mode, 0, "", "", "", "", "", "FAILED"
                            ])
                        failed += 1
                        continue

                    print(f"[SUCCESS] 完成！耗时: {elapsed:.3f}秒")
                    print(f"[INFO] 输出树: {result_tree}")

                    # 打印metric结果
                    if rf_metric is not None:
                        print(f"[INFO] RF距离: {rf_metric:.6f}, RF准确率: {1-rf_metric:.6f}")
                    if q_metric is not None:
                        print(f"[INFO] Quartet准确率: {q_metric:.6f}")

                    # 写入结果
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            version_name, species_num, dna_len, task_type,
                            k_param, run_mode, f"{elapsed:.3f}",
                            f"{rf_metric:.6f}" if rf_metric is not None else "",
                            f"{1-rf_metric:.6f}" if rf_metric is not None else "",
                            f"{1-q_metric:.6f}" if q_metric is not None else "",
                            f"{q_metric:.6f}" if q_metric is not None else "",
                            result_tree, "SUCCESS"
                        ])

                    completed += 1

                except Exception as e:
                    elapsed = 0
                    result_tree = ""
                    print(f"[ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1

                    # 写入错误结果
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            version_name, species_num, dna_len, task_type,
                            k_param, run_mode, f"{elapsed:.3f}", "", "", "", "",
                            f"ERROR: {str(e)[:50]}", "ERROR"
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

    # 打印版本对比摘要
    print("\n版本对比摘要:")
    print("-" * 100)

    # 读取CSV并统计
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            results = list(reader)

        # 按版本分组
        version_stats = {}
        for row in results:
            version = row["version"]
            if version not in version_stats:
                version_stats[version] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_rf_acc": 0,
                    "rf_count": 0
                }

            if row["status"] == "SUCCESS":
                version_stats[version]["count"] += 1
                version_stats[version]["total_time"] += float(row["time_s"])

                if row["rf_accuracy"]:
                    version_stats[version]["avg_rf_acc"] += float(row["rf_accuracy"])
                    version_stats[version]["rf_count"] += 1

        # 打印统计
        for version, stats in version_stats.items():
            if stats["count"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                avg_rf = stats["avg_rf_acc"] / stats["rf_count"] if stats["rf_count"] > 0 else 0
                print(f"\n{version}:")
                print(f"  成功数: {stats['count']}")
                print(f"  平均耗时: {avg_time:.3f}秒")
                print(f"  平均RF准确率: {avg_rf:.4f}")

    except Exception as e:
        print(f"[ERROR] 生成统计摘要失败: {e}")

    print("=" * 100)

    return completed, failed


if __name__ == "__main__":
    # ==================== 配置参数（在此修改）====================

    # 要测试的物种数列表
    SPECIES_LIST = [24, 48, 96, 192]

    # 要测试的DNA长度列表
    DNA_LEN_LIST = [100000, 1000000]

    # 运行模式: "fast" 或 "regular"
    RUN_MODE = "regular"

    # 任务类型: "homogeneous" 或 "heterogeneous"
    TASK_TYPE = "homogeneous"

    # k参数（用于block采样）
    K_PARAM = 3.0

    # 推断批次大小
    INFER_BATCH_SIZE = 32

    # 是否清理临时文件
    CLEANUP = True

    # 结果输出路径
    OUTPUT_CSV = "compare_results.csv"

    # ===========================================================

    run_comparison_test(
        species_list=SPECIES_LIST,
        dna_len_list=DNA_LEN_LIST,
        task_type=TASK_TYPE,
        k_param=K_PARAM,
        run_mode=RUN_MODE,
        infer_batch_size=INFER_BATCH_SIZE,
        cleanup=CLEANUP,
        output_csv=OUTPUT_CSV
    )
