from cgi import test
import os
import re
import math
import subprocess
import shutil
from pathlib import Path
from typing import Literal

import random
from numpy.random import dirichlet

from ete3 import Tree
from scipy.stats import gamma, uniform, expon

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
import os, re
import numpy as np

#=====Gene tree simulation=====#

# 模拟基因树
def simulate_phylogeny(path, species_num, tree_num, root=False, append_mode=False):

    tree_set_path = path
    os.makedirs(tree_set_path, exist_ok=True)
    
    # 确定起始索引
    start_idx = 0
    if append_mode:
        existing_dirs = []
        if os.path.exists(tree_set_path):
            for item in os.listdir(tree_set_path):
                item_path = os.path.join(tree_set_path, item)
                if os.path.isdir(item_path) and item.isdigit():
                    existing_dirs.append(int(item))
        
        if existing_dirs:
            start_idx = max(existing_dirs) + 1
            print(f"追加模式：从索引 {start_idx} 开始生成 {tree_num} 个树")
        else:
            print("追加模式：未找到现有文件夹，从索引 0 开始")
    
    for i in range(tree_num):
        idx = start_idx + i
        if species_num == "sample":
            cur_species_num = random.randint(10, 60)
        else:
            cur_species_num = int(species_num)
            
        # 物种名称列表
        species_names = [f"Sp{i}" for i in range(cur_species_num)]
        random.shuffle(species_names)

        t = Tree()
        t.populate(cur_species_num, names_library=species_names, random_branches=True)

        if not root:
            t.unroot(mode='keep')

        # 使用均值为0.1的指数分布设置分支长度
        scales = np.array([0.05])
        pi = np.array([1])        # 或据经验设定/学习

        z = np.random.choice(len(scales), p=pi)
        tree_branch_scale = scales[z]

        for node in t.iter_descendants():
            node.dist = max(expon.rvs(scale=tree_branch_scale), 1e-6)

        # 保存
        tree_path = os.path.join(tree_set_path, f'{idx}')
        os.makedirs(tree_path, exist_ok=True)
        tree_file_path = os.path.join(tree_path, "tree.newick")
        with open(tree_file_path, 'w') as tree_file:
            tree_string = t.write(format=1)
            if tree_string is not None:
                tree_file.write(tree_string)

# 使用simphy模拟多locus的物种树
def _run_simphy_process(path, start_tree, end_tree, process_id, start_idx):
    """单个进程运行SimPhy"""
    try:
        # 创建临时子文件夹
        temp_dir = os.path.join(path, f"temp_{process_id}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 计算当前进程的树数量
        current_tree_num = end_tree - start_tree
        
        # 构建SimPhy命令
        simphy_cmd = [
            "/mnt/c/Users/descfly/Desktop/publish_code/SimPhy_1.0.2/bin/simphy_lnx64",
            "-rs", str(current_tree_num),
            "-rl", "u:50,300",
            "-rg", "1",
            "-sb", "ln:-14,1",
            "-sl", "u:10,70",
            "-st", "u:100000,10000000",
            "-si", "u:1,6",
            "-so", "ln:0,0.1",
            "-gb", "u:-25,-15",
            "-gt", "f:gb",
            "-ld", "ln:gb,0.4",
            "-lb", "f:ld",
            "-lt", "ln:gt,0.4",
            "-lk", "1",
            "-sp", "ln:12,0.5",
            "-sg", "f:1",
            "-hs", "ln:1.5,1",
            "-hl", "ln:1.2,1",
            "-hg", "ln:1.4,1",
            "-su", "e:10000000",
            "-o", temp_dir,
            "-v", "1"
        ]
        
        # 运行SimPhy命令
        result = subprocess.run(simphy_cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"进程 {process_id} SimPhy执行失败: {result.stderr}")
            return 0
        
        print(f"进程 {process_id} 完成，处理了 {current_tree_num} 个树")
        return current_tree_num
        
    except subprocess.TimeoutExpired:
        print(f"进程 {process_id} SimPhy执行超时")
        return 0
    except Exception as e:
        print(f"进程 {process_id} 执行出错: {e}")
        return 0

def _merge_and_rename_simphy_results(path, tree_num, start_idx):
    """合并和重命名SimPhy结果文件"""
    try:
        # 收集所有临时文件夹中的结果
        temp_dirs = []
        for item in os.listdir(path):
            if item.startswith("temp_") and os.path.isdir(os.path.join(path, item)):
                temp_dirs.append(item)
        
        temp_dirs.sort(key=lambda x: int(x.split("_")[1]))  # 按进程ID排序
        
        # 收集所有生成的文件
        all_files = []
        for temp_dir in temp_dirs:
            temp_path = os.path.join(path, temp_dir)
            for file in os.listdir(temp_path):
                if not file.startswith("."):  # 忽略隐藏文件
                    all_files.append((temp_path, file))
        
        # 按文件名排序，确保顺序一致
        all_files.sort(key=lambda x: x[1])
        
        # 重命名并移动到目标位置
        for i, (source_dir, filename) in enumerate(all_files):
            if i >= tree_num:
                break
                
            source_path = os.path.join(source_dir, filename)
            target_folder = os.path.join(path, str(start_idx + i))
            os.makedirs(target_folder, exist_ok=True)
            
            # 如果是文件夹，移动整个文件夹
            if os.path.isdir(source_path):
                target_path = target_folder
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.move(source_path, target_path)
            else:
                # 如果是文件，移动到目标文件夹
                target_path = os.path.join(target_folder, filename)
                shutil.move(source_path, target_path)
        
        # 清理临时文件夹
        for temp_dir in temp_dirs:
            temp_path = os.path.join(path, temp_dir)
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
        
        print(f"文件合并完成，共处理 {min(len(all_files), tree_num)} 个结果")
        
    except Exception as e:
        print(f"合并文件时出错: {e}")

def simulate_multilocus_phylogeny(path, tree_num, append_mode=False):
    """
    使用simphy模拟多locus的物种树
    
    simphy -rs 100 -rl u:50,300 -rg 1 -sb ln:-14,1 -sl u:10,70 -st u:100000,10000000 -si u:1,6 -so ln:0,0.1 -gb u:-25,-15 -gt f:gb -ld ln:gb,0.4 -lb f:ld -lt ln:gt,0.4 -lk 1 -sp ln:12,0.5 -sg f:1 -hs ln:1.5,1 -hl ln:1.2,1 -hg ln:1.4,1 -su e:10000000 -o data/part2/train -v 1
    tree_num 对应 -rs
    path对应 -o
    使用多进程运行，切分tree_num，先在path为每个子进程创建子文件夹temp_pid，然后每个子进程运行simphy，最后将每个子进程的文件合并，按照0~tree_num-1的顺序重命名，（假设是append_mode，那么就先读取当前文件夹最后的文件夹的下标，然后在此基础上增加）
    """
    
    # 确定起始索引
    start_idx = 0
    if append_mode:
        existing_dirs = []
        if os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and item.isdigit():
                    existing_dirs.append(int(item))
        
        if existing_dirs:
            start_idx = max(existing_dirs) + 1
            print(f"追加模式：从索引 {start_idx} 开始生成 {tree_num} 个树")
        else:
            print("追加模式：未找到现有文件夹，从索引 0 开始")
    
    # 创建输出目录
    os.makedirs(path, exist_ok=True)
    
    try:
        tree_num = int(tree_num)
    except (TypeError, ValueError):
        raise ValueError("tree_num 必须是可以转换为整数的值")

    if tree_num <= 0:
        print("tree_num 不大于 0，跳过SimPhy模拟")
        return

    # 获取CPU核心数，设置并行进程数
    max_workers = min(tree_num, multiprocessing.cpu_count(), 16)  # 限制最大进程数

    print(f"使用 {max_workers} 个进程并行处理 {tree_num} 个树")

    # 计算每个进程应该处理的树数量
    trees_per_process = max(1, tree_num // max_workers)
    remaining_trees = tree_num % max_workers
    
    # 使用进程池执行任务，将树分配给各个进程
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 为每个进程分配树的范围
        current_start = 0
        for process_id in range(max_workers):
            if current_start >= tree_num:
                break
                
            # 计算当前进程的树数量
            if process_id < remaining_trees:
                current_count = trees_per_process + 1
            else:
                current_count = trees_per_process
                
            current_end = min(current_start + current_count, tree_num)
            
            if current_start < current_end:
                future = executor.submit(
                    _run_simphy_process,
                    path,
                    current_start,
                    current_end,
                    process_id,
                    start_idx,
                )
                futures.append(future)
                print(f"进程 {process_id} 分配树范围: {current_start} - {current_end-1} (共 {current_end - current_start} 棵树)")
                
            current_start = current_end

        # 等待所有进程完成
        completed_trees = 0
        with tqdm(total=tree_num, desc="SimPhy树生成进度", unit="树") as progress_bar:
            for future in futures:
                try:
                    trees_completed = future.result()
                    completed_trees += trees_completed
                    progress_bar.update(trees_completed)
                    
                    progress_bar.set_postfix({
                        "已完成": f"{completed_trees}/{tree_num}",
                        "进度": f"{completed_trees / tree_num * 100:.1f}%"
                    })
                    progress_bar.refresh()
                    
                except Exception as e:
                    print(f"SimPhy进程执行失败: {e}")
    
    # 合并和重命名文件
    _merge_and_rename_simphy_results(path, tree_num, start_idx)
    print(f"SimPhy模拟完成，共生成 {tree_num} 个树")

# alisim模拟MSA
def _run_alisim(insert_delet:bool, output_path:str, tree_file_path:str, evolution_model:Literal['GTR', 'UNREST'], threads:int=2, mode: Literal['single', 'mixture_models', 'branch_specific_models'] = 'single', seed=None, dna_length=None, multilocus_mode=False, locus_id=None):  
    """
    用 AliSim 模拟 1 份 MSA，输出固定为 output_path/MSA.phy
    如果multilocus_mode=True，则输出为 output_path/g_trees{locus_id}.phy 以避免多基因座文件覆盖
    """
    import os, subprocess, random, tempfile, re
    import numpy as np

    os.makedirs(output_path, exist_ok=True)

    # ---- 抽样与格式化工具 ----
    def sample_DNA_length():
        rand = random.random()
        if rand < 0.6:
            return random.randint(400, 1200)
        elif rand < 0.9:
            return random.randint(1200, 2000)
        else:
            return random.randint(2000, 3000)

    def sample_base_freqs():
        f = np.random.dirichlet([10, 10, 10, 10])
        return [float(f[0]), float(f[1]), float(f[2]), float(f[3])]  # A/C/G/T

    def fmt_freqs(freqs):
        return "F{" + "/".join(f"{x:.6f}" for x in freqs) + "}"

    def sample_pinv_alpha():
        pinv = random.uniform(0.0, 1.0)
        alpha = random.uniform(0.0, 4.0)
        return pinv, alpha

    def fmt_model(model_name):
        if model_name == "GTR":
            rates = [random.uniform(0, 3) for _ in range(5)]
            freqs = sample_base_freqs()
            return "GTR{" + "/".join(f"{r:.6f}" for r in rates) + "}+" + fmt_freqs(freqs)
        
        elif model_name == "UNREST":
            w = [random.uniform(-0.95, 0.95) for _ in range(11)]
            freqs = sample_base_freqs()
            return "12.12{" + "/".join(f"{x:.6f}" for x in w) + "}+" + fmt_freqs(freqs)
        
        elif model_name == "JC":
            return "JC"
        
        elif model_name == "TIM":
            r = [random.uniform(0, 3) for _ in range(3)]
            return "TIM{" + "/".join(f"{x:.6f}" for x in r) + "}+" + fmt_freqs(sample_base_freqs())

    def add_rate_heterogeneity(model_core, pinv, alpha):
        suffix = ""
        if pinv > 0:
            suffix += f"+I{{{pinv:.6f}}}"
        if alpha > 0:
            suffix += f"+G{4}{{{alpha:.6f}}}"
        return model_core + suffix

    def build_mixture_model(pinv, alpha):
        num_components = random.randint(3, 8)
        
        comp = []
        if random.random() < 0.5:
            for _ in range(num_components):
                comp.append(fmt_model("UNREST"))
        else:
            for _ in range(num_components):
                model_choices = ["JC", "GTR", "TIM"]
                comp.append(fmt_model(random.choice(model_choices)))
        
        model = 'MIX{' + ",".join(comp) + '}'
        return add_rate_heterogeneity(model, pinv, alpha)

    def write_random_branch_models(orig_tree_path, out_tree_path, pinv, alpha):
        with open(orig_tree_path, "r") as f:
            newick = f.read().strip()

        # 匹配所有分支（叶子与内部）：形如 ":0.123" 的长度位置
        # 注意：如果原树已有注释，这里简单插入可能会与已有注释并列
        branch_matches = list(re.finditer(r":\s*([0-9.+\-Ee]+)(?!\[)", newick))
        n_branches = len(branch_matches)
        if n_branches == 0:
            # 没有分支长度则不处理
            with open(out_tree_path, "w") as g:
                g.write(newick)
            return

        # 随机挑选要注解的分支个数：约 20%-50% 之间
        k = max(1, int(n_branches * random.uniform(0.2, 0.5)))
        picks = set(random.sample(range(n_branches), k))

        # 生成与全局一致的抽样规则的模型（每条分支可不同）
        def random_branch_model():
            # 随机选一个核心模型并叠加当前 pinv/alpha
            base_choice = random.choice(["JC", "GTR", "TIM", "UNREST"])
            return add_rate_heterogeneity(fmt_model(base_choice), pinv, alpha)

        # 逐个回填注解（从后往前替换，避免索引位移）
        chars = list(newick)
        for idx in sorted(picks, reverse=True):
            m = branch_matches[idx]
            insert_pos = m.end()         # 放在长度后面
            model_text = f"[&model={random_branch_model()}]"
            chars[insert_pos:insert_pos] = list(model_text)

        newick_annot = "".join(chars)
        with open(out_tree_path, "w") as g:
            g.write(newick_annot)

    def indel_args():
        if not insert_delet:
            return []
        return ["--indel", "0.01,0.01"]

    # ---- 主流程（单份 MSA，固定名 MSA.phy） ----
    if dna_length is not None:
        L = dna_length
    else:
        L = sample_DNA_length()
    pinv, alpha = sample_pinv_alpha()

    if mode == 'mixture_models':
        model_full = build_mixture_model(pinv, alpha)
    else:
        model_full = add_rate_heterogeneity(fmt_model(evolution_model), pinv, alpha)

    tree_for_run = tree_file_path
    tmp_tree = None
    if mode == 'branch_specific_models':

        import tempfile
        tmp_tree = tempfile.NamedTemporaryFile(delete=False, suffix=".nwk", dir=output_path)
        tmp_tree.close()
        write_random_branch_models(tree_file_path, tmp_tree.name, pinv, alpha)
        tree_for_run = tmp_tree.name
    

    if multilocus_mode and locus_id is not None:
        # 多基因座模式：使用基因座ID避免文件覆盖
        prefix = os.path.join(output_path, f"g_trees{locus_id}")
    elif mode == 'single':
        prefix = os.path.join(output_path, f"{evolution_model}_{dna_length}_MSA")
    elif mode == "mixture_models":
        prefix = os.path.join(output_path, f"mixture_models_{dna_length}_MSA")
    elif mode == "branch_specific_models":
        prefix = os.path.join(output_path, f"branch_specific_models_{dna_length}_MSA")
    cmd = [
        "iqtree3", "--alisim", prefix,
        "-t", tree_for_run,
        "-m", model_full,
        "--length", str(L),
        "--out-format", "phy",
        "-nt", str(threads)
    ] + indel_args()

    if seed is not None:
        cmd += ["-seed", str(int(seed))]

    # 将输出重定向到日志文件
    log_file = os.path.join(output_path, "alisim.log")
    with open(log_file, 'w') as f:
        subprocess.run(cmd, check=True, cwd=output_path, stdout=f, stderr=subprocess.STDOUT)

def _process_single_alisim_folder(args):
    """处理单个文件夹的AliSim模拟（用于并行处理）"""
    insert_delet, folder_path, tree_file_path, evolution_model, mode, seed, dna_length = args
    try:
        _run_alisim(
            insert_delet=insert_delet,
            output_path=folder_path,
            tree_file_path=tree_file_path,
            evolution_model=evolution_model,
            mode=mode,
            seed=seed,
            dna_length=dna_length
        )
        return f"成功处理: {folder_path}"
    except Exception as e:
        return f"处理失败: {folder_path}, 错误: {str(e)}"

def execute_alisim(insert_delet:bool, tree_set_path:str, evolution_model:Literal['GTR', 'UNREST'], mode: Literal['single', 'mixture_models', 'branch_specific_models'] = 'single', seed=None, append_mode=False, tree_num=10000, dna_length=None):
    """
    批量执行AliSim模拟
    
    Args:
        insert_delet: 是否插入删除
        tree_set_path: 树文件集合路径，例如"/mnt/c/Users/descfly/Desktop/publish_code/data/part1/train/attn"
        evolution_model: 进化模型
        mode: 模拟模式
        seed: 随机种子
        append_mode: 是否为追加模式
        tree_num: 处理的树数量（仅在append_mode=False时使用）
        max_workers: 最大并行工作进程数，默认为CPU核心数
        log_file: 日志文件路径，如果为None则输出到终端
        dna_length: 指定DNA序列长度，如果为None则随机生成
    """


    def scan_tree_folders(tree_set_path, append_mode=False, tree_num=10000):
        """
        扫描树文件夹，返回需要处理的文件夹列表

        Args:
            tree_set_path: 树文件集合路径
            append_mode: 是否为追加模式
            tree_num: 处理的树数量（仅在append_mode=False时使用）

        Returns:
            list: 需要处理的文件夹路径列表，每个元素为 (folder_path, tree_file_path)
        """
        tree_set_path = Path(tree_set_path)
        folders_to_process = []

        if not append_mode:
            # 模式1：扫描0~tree_num-1个文件夹
            for i in tqdm(range(tree_num), desc="扫描文件夹", unit="个"):
                subfolder = tree_set_path / str(i)
                tree_file = subfolder / "tree.newick"

                if tree_file.exists():
                    folders_to_process.append((str(subfolder), str(tree_file)))
                else:
                    tqdm.write(f"警告: {tree_file} 不存在，跳过")
        else:
            # 模式2：扫描所有文件夹，收集不含有MSA.phy的文件夹
            all_folders = [f for f in tree_set_path.iterdir() if f.is_dir()]

            for folder in tqdm(all_folders, desc="扫描文件夹", unit="个"):
                msa_file = folder / "MSA.phy"
                tree_file = folder / "tree.newick"

                if not msa_file.exists() and tree_file.exists():
                    folders_to_process.append((str(folder), str(tree_file)))
                elif not tree_file.exists():
                    tqdm.write(f"警告: {tree_file} 不存在，跳过文件夹 {folder}")

        return folders_to_process

    folders_to_process = scan_tree_folders(tree_set_path, append_mode, tree_num)
    
    print(f"找到 {len(folders_to_process)} 个文件夹需要处理")
    
    # 设置并行工作进程数
    max_workers = min(multiprocessing.cpu_count(), len(folders_to_process))
    log_file = os.path.join(tree_set_path, "alisim_batch.log")

    # 准备参数列表
    args_list = [
        (insert_delet, folder_path, tree_file_path, evolution_model, mode, seed, dna_length)
        for folder_path, tree_file_path in folders_to_process
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single_alisim_folder, args)
            for args in args_list
        ]
        
        # 使用tqdm显示进度条，将详细信息写入日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"AliSim批量处理日志\n")
            f.write(f"开始时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"处理文件夹数量: {len(folders_to_process)}\n")
            f.write(f"并行进程数: {max_workers}\n")
            f.write("-" * 50 + "\n")
            
            # 进度条在终端显示
            for future in tqdm(as_completed(futures), total=len(futures), desc="并行处理进度", unit="文件夹"):
                result = future.result()
                f.write(f"{result}\n")
                f.flush()  # 确保立即写入文件
    
    print("批量AliSim模拟完成")
    print(f"详细日志已保存到: {log_file}")

if __name__ == "__main__":
    
    for species_num in [768, 1024]:
        path = f"/mnt/c/Users/descfly/Desktop/publish_code/data/{species_num}"
        # simulate_phylogeny(path=path, species_num=species_num, tree_num=1, root=False, append_mode=True)
        for dna_length in [1000000, 10000000]:
            execute_alisim(insert_delet=False, tree_set_path=path, evolution_model='GTR', mode="single", tree_num=1, dna_length=dna_length, append_mode=True)

     
    
    
