# Deployment Guide

This project is intended to be published as a **self-contained repository** (model weights + bundled assemblers included).

## 1. Recommended Release Target

- OS: Ubuntu 22.04/24.04
- Python: 3.10
- GPU: NVIDIA GPU with >= 8 GB VRAM (required)
- CUDA-enabled PyTorch environment

## 2. Publish to GitHub

From the project root:

```bash
git add .
git commit -m "Release v1.0"
git tag v1.0.0
git push origin <your-branch>
git push origin v1.0.0
```

Create a GitHub Release from tag `v1.0.0`.

## 3. Deploy on a Server

```bash
git clone <your-repo-url>
cd publish_code
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -r requirements.txt
python scripts/check_env.py
```

If extension binaries are incompatible with the server Python/system, rebuild:

```bash
python cpp_source/setup_sequence.py build_ext --inplace
python cpp_source/setup_batching.py build_ext --inplace
python cpp_source/cuda13_pattern_freq/setup_pytorch.py build_ext --inplace
python scripts/check_env.py
```

## 4. Smoke Test

```bash
python run_qf.py --phy examples/24/MSA.phy --out output/smoke_24.nwk --run-mode regular
python run_qf.py --phy examples/48/MSA.phy --out output/smoke_48.nwk --ref-tree examples/48/tree.nwk --metric rf
```

## 5. Notes

- Current version supports only `num_species >= 24`.
- CUDA-enabled runtime is required for inference in the current release.
- Input format is PHYLIP (`.phy`) supermatrix alignment.
- Output format is Newick (`.nwk`).
