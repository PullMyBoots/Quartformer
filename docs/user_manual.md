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

At runtime, QuartFormer resolves parameters as follows:

1. It reads `infer_config.jsonc` from the project root by default.  
   In this repository: `/mnt/c/Users/descfly/Desktop/publish_code/infer_config.jsonc`
2. If `--config <path>` is provided, that file is used instead.
3. Parameters are treated as two groups:  
   `basic` (`phy`, `out`, `task_type`, `compute_branch_support`, `plot_tree`, `ref_tree`, `metric`, `quartet_sample_size`) and `advanced` (`k_param`, `quartet_assembler`, `qmc_iter_limit`, `aggregate_mode`, ...).

Precedence rules in current code:

- `basic`: `CLI > config > default`
- `advanced`: `config > CLI > default`

Practical usage:

- Use CLI for run-specific inputs (`phy`, `out`, `task_type`, `metric`).
- Keep stable tuning knobs in config (`advanced` section).

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
  "metric": "rf"
}
```

If you use quartet concordance as the metric, add:

```json
{
  "metric": "quartet",
  "quartet_sample_size": 40000
}
```

### 2.2 Basic Parameter Reference

`phy`: Input alignment path (`.phy`).

`out`: Output tree path (`.nwk`).

`task_type`: Inference mode.  
`homogeneous` is used for gene-tree-oriented settings.  
`heterogeneous` is used for species-tree-oriented settings (training distribution includes ILS, HGT, and gene copy gain/loss effects).

`compute_branch_support`: If enabled, TREE-QMC support-only mode is used to annotate internal branches.

`plot_tree`: If enabled, also render output tree figures (`.png`).

`ref_tree` + `metric`: Optional formal comparison against a reference tree.  
`metric="rf"`: normalized RF distance.  
`metric="quartet"`: quartet concordance.

`quartet_sample_size`: Used only when `metric="quartet"`.  
`>0` means sample up to this many quartets.  
`0` means no sampling (compute all quartets; may be very slow on large trees).

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

### 3.1 `k_param` (Sampling Budget Growth)

`k_param` controls how fast quartet sampling budget grows as taxa count increases.

In current implementation:

```text
target_total_quartets = num_species ^ k_param
```

That target is converted into the number of sampled 24-taxon blocks. So a larger `k_param` means more blocks and more quartets.

Guideline:

- Keep `3.0` as default.
- Tune narrowly (`2.9 ~ 3.1`) only when needed.
- Lower values speed up runtime but increase under-sampling risk.
- Higher values improve coverage/stability but add runtime and aggregation overhead.

### 3.2 `quartet_assembler` and `qmc_iter_limit`

`quartet_assembler` chooses the quartet tree assembler backend:

- `qfm`: often slightly better in accuracy in many settings; a good default for very large taxa counts.
- `qmc`: usually close to `qfm` in accuracy (often no large gap), with different runtime behavior by dataset.

`qmc_iter_limit` applies only when `quartet_assembler="qmc"`. Larger values generally increase runtime.

### 3.3 `aggregate_mode` (Duplicate Quartet Aggregation)

This option controls how repeated quartets are merged across sampled subsets.

Why it matters: repeated quartets can be accumulated. Accumulation may push some quartet weights higher than others, which can introduce weighting-instability risk relative to the original goal of comparable `0~100` quartet weights. This is not a guaranteed accuracy gain or loss; it is mainly a stability/overhead tradeoff.

Modes:

- `full`: global aggregation across all batches.
- `batch_only`: aggregation only within each batch.
- `off`: disable global aggregation path.

Recommended by scale:

- `<300 ~ 500 taxa`: `full` is usually acceptable.
- `>500 taxa`: prefer `batch_only` or `off` to reduce large aggregation overhead.

As taxa count increases, duplicate quartets are typically less frequent under your sampling strategy, so reducing aggregation often has limited stability impact.

### 3.4 Other Advanced Knobs

- `quartformer_top1_only=true`: faster, but can reduce accuracy.
- `keep_support_files=true`: keep `.support.csv` and `.support_quartets.txt`.
- `infer_batch_size`: reduce first when GPU memory is limited.  
  Based on your observed runs: `1024 taxa + 10Mbp` can run around `infer_batch_size=2` with about `~3GB` VRAM.

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
- `quartet_assembler=qfm`
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
