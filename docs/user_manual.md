<img src="../assets/logo-icon.svg" alt="QuartFormer Icon" width="56">

# QuartFormer User Manual

## 1. Data Preparation

The current release accepts `PHY` (PHYLIP) as input.

If you have multi-locus MSAs, first align taxon names across loci and concatenate them into one supermatrix, then use that file as `phy`.

Recommended checks:

- Keep taxon names consistent across all loci.
- Use standard missing/gap symbols for missing sites.
- Ensure the final file can be parsed by a standard PHYLIP reader.

## 2. Common Configuration Parameters

### 2.1 Configuration Loading and Override Rules (Important)

At runtime, parameters are resolved in this order:

1. By default, the program reads `infer_config.jsonc` from the project root.  
In this repository, the default path is:  
`/mnt/c/Users/descfly/Desktop/publish_code/infer_config.jsonc`
2. If you pass `--config <path>`, that file is used instead.
3. Parameters are then processed in two groups:
- `basic`: `phy/out/task_type/compute_branch_support/plot_tree/ref_tree/metric/quartet_sample_size`
- `advanced`: `k_param/quartet_assembler/qmc_iter_limit/aggregate_mode/...`

Current precedence:

- `basic`: `CLI > config > default`
- `advanced`: `config > CLI > default`

Practical interpretation:

- For temporary changes (input/output, mode, support switch), prefer CLI.
- For stable performance/assembly behavior, define values in `advanced` within config.

Note:

- The current CLI does not provide reverse flags like `--no-compute-branch-support` or `--no-plot-tree`.  
If those options are set to `true` in config, they remain enabled unless the config is changed.

Common (`basic`) parameter block:

```json
{
  "config": "infer_config.jsonc",
  "phy": "",
  "out": "output_tree.nwk",
  "task_type": "homogeneous",
  "compute_branch_support": false,
  "plot_tree": false,
  "ref_tree": "",
  "metric": "rf",
  "quartet_sample_size": 40000
}
```

Parameter notes:

- `phy`: input alignment path (`.phy`).
- `out`: output tree path (`.nwk`).
- `task_type`:
- `homogeneous`: gene-tree construction mode.
- `heterogeneous`: species-tree construction mode. Its training distribution explicitly includes complex events such as ILS, horizontal gene transfer (HGT), and gene copy gain/loss.
- `compute_branch_support`:
- When enabled, TREE-QMC support-only mode is used to annotate internal branches with support values from weighted quartets.
- For large taxon sets (for example, >400 or >500), support computation can significantly increase runtime.
- `plot_tree`: whether to also render tree figures (`.png`).
- `ref_tree` + `metric`:
- Use these when a reference tree is available and you want formal topology comparison.
- `metric="rf"`: normalized RF distance.
- `metric="quartet"`: quartet concordance (large cases typically use sampling internally).
- `quartet_sample_size`:
- Used when `metric="quartet"`.
- `>0`: sample at most this many quartets.
- `0`: compute all quartets (can be very slow on large trees).

## 3. Advanced Parameters

Place advanced settings under `advanced`. Recommended template:

```json
{
  "advanced": {
    "k_param": 3.0,
    "quartet_assembler": "qmc",
    "qmc_iter_limit": 1,
    "aggregate_mode": "off",
    "quartformer_top1_only": false,
    "keep_support_files": false,
    "infer_batch_size": 32
  }
}
```

Tuning guidance:

- `k_param`:
- Usually keep the default.
- A small tuning range can be tested (for example `2.8 ~ 3.2`); values that are too low may reduce accuracy.
- `quartet_assembler`:
- `qfm`: often strong on small to medium datasets.
- `qmc`: usually more robust for larger taxon counts (for example >200), and commonly preferred in large runs.
- `qmc_iter_limit`: controls qmc search depth; larger values are usually slower.
- `aggregate_mode`:
- `full`: global aggregation, typically more stable but with higher time/memory cost.
- `batch_only`: per-batch aggregation only, often a better speed/resource tradeoff.
- `off`: disables global aggregation path, often faster for very large datasets.
- `quartformer_top1_only`:
- `true` can speed up assembly, but may reduce accuracy.
- `keep_support_files`:
- whether to retain `.support.csv` and `.support_quartets.txt`.
- `infer_batch_size` (integer):
- reduce this first when GPU memory is limited.
- Based on your observed setting: for `1024 taxa + 10Mbp`, `infer_batch_size=2` uses about `~3GB` VRAM.

## 4. Typical Usage Scenarios

Scenario A: quick tree inference (without support)

```bash
python infer_tree.py \
  --phy data/example/MSA.phy \
  --out output/example.nwk \
  --task-type heterogeneous
```

Scenario B: infer tree + compute support + render plots

```bash
python infer_tree.py \
  --phy data/example/MSA.phy \
  --out output/example.nwk \
  --task-type heterogeneous \
  --compute-branch-support \
  --plot-tree
```

Scenario C: compare with an existing reference tree

```bash
python infer_tree.py \
  --phy data/example/MSA.phy \
  --out output/example.nwk \
  --ref-tree data/example/tree_best.newick \
  --metric rf
```

Scenario D: conservative setup for large datasets

- `task_type=heterogeneous`
- `quartet_assembler=qmc`
- `aggregate_mode=batch_only` or `off`
- reduce `infer_batch_size` step by step until VRAM usage is stable

---

If your immediate goal is stable reproducibility, keep `advanced` fixed and change only three entry parameters: `phy`, `out`, and `task_type`.

## 5. Local Deployment Guide

This section is a practical setup checklist for running QuartFormer on a new machine.

### 5.1 Recommended Environment

- OS: Linux x86_64 (or WSL2 Ubuntu on Windows)
- Python: `3.10` (required by bundled `cp310` extension binaries)
- GPU: NVIDIA GPU with CUDA runtime available
- Java: `17+` (used by QFM-FI fast assembler)

### 5.2 Create Environment

```bash
conda create -n quartformer python=3.10 -y
conda activate quartformer
```

### 5.3 Install PyTorch (CUDA) and Python Packages

Install a CUDA-compatible PyTorch first (example for CUDA 12.8):

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
```

Then install project dependencies:

```bash
pip install -r requirements.txt
```

### 5.4 Check Required Project Artifacts

Confirm these files exist:

- `backend/sequence_processor_backend.cpython-310-x86_64-linux-gnu.so`
- `backend/pattern_freq_cuda_backend.cpython-310-x86_64-linux-gnu.so`
- `backend/quartet_aggregate_backend.cpython-310-x86_64-linux-gnu.so`
- `quartet_assemble_method/TREE-QMC_fast/build_local/tree-qmc`
- `quartet_assemble_method/qfm_java_fast/QFM-FI_unzipped/QFM-FI-fast.jar`

If your platform changes and native modules are incompatible, rebuild backend extension(s):

```bash
cd backend
python setup_batching.py build_ext --inplace
cd ..
```

### 5.5 Run Environment Preflight

```bash
python scripts/check_env.py
```

If you only want to run diagnostics without GPU hard failure:

```bash
python scripts/check_env.py --allow-no-cuda
```

### 5.6 Smoke Test

```bash
python infer_tree.py \
  --phy examples/24/MSA.phy \
  --out output/example_24.nwk \
  --task-type homogeneous
```

Optional (support + plotting):

```bash
python infer_tree.py \
  --phy examples/24/MSA.phy \
  --out output/example_24.nwk \
  --task-type heterogeneous \
  --compute-branch-support \
  --plot-tree
```
