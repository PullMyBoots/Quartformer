# QuartFormer

Deep learning phylogenetic inference for long alignments and medium-to-large taxon sets, designed for fast runtime.

## What This Project Is

QuartFormer infers species trees from concatenated supermatrix alignments in **PHYLIP (`.phy`)** format.  
It targets genome-scale datasets where sequence length is large (for example 1M to 10M sites) and taxon count is moderate to high.
Current version supports inference tasks with **at least 24 taxa** (`num_species >= 24`).

## Key Features

- Fast inference on long-sequence datasets using GPU-accelerated quartet scoring.
- Accuracy has been evaluated with **RF distance** and **quartet concordance**; see `docs/accuracy.md` for details.
- Bundled assembly backends and model weights for reproducible release packaging.
- Example datasets included in `examples/` for quick verification.

## Official Tested Environment

- OS: Ubuntu 22.04 (WSL2 Linux)
- Python: 3.10.18
- PyTorch: 2.8.0 + CUDA 12.8
- Triton: 3.4.0
- Java: OpenJDK 17
- CMake: 3.30+
- GPU: NVIDIA GPU with **>= 8 GB VRAM** (required)

## Installation (Bundled Release)

This repository includes model weights and bundled third-party components (`TREE-QMC`, `QFM-FI`), so no additional external repository checkout is required.

Install CUDA-enabled PyTorch first (example for CUDA 12.8):

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If your local extension binaries are missing, build them from bundled sources:

```bash
python cpp_source/setup_sequence.py build_ext --inplace
python cpp_source/setup_batching.py build_ext --inplace
python cpp_source/cuda13_pattern_freq/setup_pytorch.py build_ext --inplace
```

## Quick Start

Default mode is `regular`.

```bash
# Basic inference
python run_qf.py --phy examples/24/MSA.phy --out output/qf_24.nwk --run-mode regular

# RF distance evaluation
python run_qf.py --phy examples/48/MSA.phy --out output/qf_48.nwk --ref-tree examples/48/tree.nwk --metric rf

# Quartet concordance evaluation
python run_qf.py --phy examples/96/MSA.phy --out output/qf_96.nwk --ref-tree examples/96/tree.nwk --metric quartet
```

## Documentation

- Speed benchmarks: `docs/benchmarks.md`
- Accuracy benchmarks: `docs/accuracy.md`
- Release and server deployment: `docs/deploy.md`

## Input and Output

- Input: concatenated supermatrix alignment in `.phy` (PHYLIP)
- Output: inferred phylogenetic tree in `.nwk`

## Runtime Requirement Note

QuartFormer currently requires CUDA tensors during inference. CPU-only environments are not supported by `run_qf.py` in this release.

## Current Limitation

The current release does **not** explicitly model insertion/deletion (indel) signal from gap characters (`-`).  
Gap characters are used as alignment placeholders, but indel events are not treated as an independent phylogenetic signal source in the current model.

## Experimental Status and Usage Scope

QuartFormer is an **experimental** deep learning-based tree inference method.  
In this release, accuracy claims are primarily supported by simulated-data evaluations (see `docs/accuracy.md`).

For real biological datasets, there is currently no complete theoretical guarantee that inferred topologies are always reliable.  
Use QuartFormer as an auxiliary inference tool, not as the sole basis for high-stakes decisions that may cause financial, clinical, legal, or other material losses.

## Acknowledgments

This project builds on publicly available tools and datasets. We thank the authors and maintainers of:

- IQ-TREE: https://github.com/iqtree/iqtree2
- RAxMLGrove: https://github.com/angtft/RAxMLGrove
- SimPhy: https://github.com/adamallo/SimPhy
- FastTree: https://github.com/morgannprice/fasttree
- ASTER: https://github.com/chaoszhang/ASTER

Please also cite the corresponding papers when using these tools and datasets in academic work.
