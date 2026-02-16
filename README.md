# QuartFormer2: Deep Learning-Based Phylogenetic Tree Inference

**QuartFormer2** is a deep learning-based tool for fast and accurate phylogenetic tree inference. By leveraging transformer architectures and GPU acceleration, QuartFormer2 achieves 1-3 orders of magnitude speedup over traditional methods while maintaining competitive accuracy.

## ✨ Key Features

- **⚡ Blazing Fast**: 10x-6000x faster than ASTER and FastTree
- **🎯 High Accuracy**: State-of-the-art quartet topology prediction
- **🚀 GPU Accelerated**: Built on CUDA-enabled deep learning models
- **📊 Scalable**: Efficiently handles datasets from 24 to 192+ taxa
- **🔧 Flexible**: Supports both homogeneous and heterogeneous modes
- **💡 User-Friendly**: Simple command-line interface with sensible defaults

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd publish_code

# Verify Python version (must be 3.10.x)
python --version

# Check your installation
python check_install.py

# Install Python dependencies
pip install -r requirements.txt

# For detailed installation instructions, see INSTALL.md
```

**Requirements**:
- Python 3.10.x (exactly)
- CUDA 12.6+ with NVIDIA GPU
- Linux (Ubuntu/WSL2 tested)

See [INSTALL.md](INSTALL.md) for detailed installation instructions, including recompiling C++ extensions for different platforms.

### Basic Usage

```bash
# Run inference on a phylogenetic alignment
python run_qf.py --phy data.phy --out output.nwk

# With reference tree evaluation
python run_qf.py --phy data.phy --out output.nwk --ref-tree reference.nwk --metric rf

# Fast mode (always uses TREE-QMC)
python run_qf.py --phy data.phy --out output.nwk --run-mode fast

# Homogeneous mode (single tree)
python run_qf.py --phy data.phy --out output.nwk --task-type homogeneous
```

### Example Datasets

Try the provided example datasets:

```bash
cd examples
./run_examples.sh
```

Or run individual examples:

```bash
# Small dataset (24 taxa)
python ../run_qf.py --phy 24/MSA.phy --out output_small.nwk

# With accuracy evaluation (48 taxa)
python ../run_qf.py --phy 48/MSA.phy --out output_48.nwk --ref-tree 48/tree.nwk --metric quartet
```

## 📖 Documentation

- **[Installation Guide](INSTALL.md)** - Detailed installation and troubleshooting
- **[Performance Benchmarks](docs/benchmarks.md)** - Comprehensive speed and accuracy comparisons
- **[Accuracy Evaluation](docs/accuracy.md)** - Accuracy on simulated datasets
- **[Real Dataset Analysis](docs/real_datasets.md)** - Real biological dataset comparisons
- **[Examples](examples/README.md)** - Example datasets and usage scripts
- **[CLAUDE.md](CLAUDE.md)** - Project documentation for developers

## 🎯 Command-Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--phy` | Input alignment file (PHY format) |
| `--out` | Output tree path (file or directory) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task-type` | `heterogeneous` | `homogeneous` (single tree) or `heterogeneous` (multi-partition) |
| `--run-mode` | `regular` | `fast` (TREE-QMC) or `regular` (QFM-FI for ≤96, TREE-QMC for >96) |
| `--k-param` | `3.0` | Sampling parameter: sampled quartets = species_count^k |
| `--infer-batch-size` | `32` | Inference batch size |
| `--ref-tree` | - | Reference tree path for evaluation |
| `--metric` | `rf` | Evaluation metric: `rf` (RF distance) or `quartet` (quartet concordance) |
| `--no-cleanup-temp-files` | - | Keep temporary files |

## 📊 Performance Highlights

Based on benchmarks with Intel Core i9-14900K, RTX 4090, Python 3.10, CUDA 13.x:

### Speed Comparison (1M sequences)

| Taxa | QuartFormer2 | ASTER | FastTree | QF Speedup |
|------|--------------|-------|----------|------------|
| 24   | 0.5s         | 5.1s  | 332.9s   | **9-620x** |
| 48   | 1.8s         | 66.2s | 896.8s   | **38-509x** |
| 96   | 6.8s         | 206.2s | 1946.8s  | **30-287x** |
| 192  | 53.3s        | 707.8s | 3840.0s  | **13-72x** |

### Large-Scale Performance (10M sequences)

- **24 taxa**: 1.5s vs FastTree's 2.5 hours (**6,152x faster**)
- **96 taxa**: 23s vs ASTER's 1.8 hours (**278x faster**)

See [docs/benchmarks.md](docs/benchmarks.md) for complete results.

## 🔬 How It Works

QuartFormer2 uses a two-stage approach:

1. **Quartet Inference**: Deep learning models (QuartFormer + MLP) predict quartet topologies from sequence data
2. **Tree Assembly**: TREE-QMC or QFM-FI assemble quartets into a complete phylogenetic tree

### Key Innovations

- **Sparse Attention Mechanism**: Efficiently processes quartet relationships
- **Batched Inference**: Processes multiple species blocks in parallel
- **GPU Acceleration**: CUDA kernels for pattern frequency computation
- **Automated Polytomy Resolution**: MLP-based resolution of multifurcating nodes

## 📁 Project Structure

```
publish_code/
├── run_qf.py              # Main inference script
├── run_qf2.py             # Alternative inference script (memory-optimized)
├── model.py               # Neural network model definitions
├── utils.py               # Utility functions
├── sparse_attn_kernel.py  # CUDA attention kernels
├── check_install.py       # Installation verification script
├── requirements.txt       # Python dependencies
├── INSTALL.md             # Detailed installation guide
├── cpp_source/            # C++ extensions (pre-compiled)
├── model/                 # Trained model weights
│   ├── homogeneous/       # Single tree models
│   └── heterogeneous/     # Multi-partition models
├── examples/              # Example datasets
├── docs/                  # Documentation
└── software/              # Third-party tools (ASTER, FastTree)
```

## 🛠️ Troubleshooting

### "Module not found" errors
- Ensure you're using Python 3.10.x: `python --version`
- Run `python check_install.py` to diagnose issues
- Reinstall dependencies: `pip install -r requirements.txt`

### CUDA errors
- Check GPU availability: `nvidia-smi`
- Verify CUDA version: `nvcc --version`
- Ensure PyTorch is built with CUDA support
- See [INSTALL.md](INSTALL.md) for detailed troubleshooting

### Out of memory errors
- Reduce `--infer-batch-size`
- Use smaller model or reduce data size

### Slow performance
- Check GPU is being used: `nvidia-smi`
- Try `--run-mode fast` for TREE-QMC (faster for large datasets)
- Increase `--infer-batch-size` if memory allows

## 📝 Citation

If you use QuartFormer2 in your research, please cite:

```
[Add citation information when published]
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **ASTER** and **FastTree** and **IQ-TREE** - Baseline methods for comparison
- **TREE-QMC** and **QFM-FI** - Quartet assembly algorithms
- **PyTorch** and CUDA - Deep learning framework and GPU acceleration
- **SimPhy** and **RAxML-NG** - Simulation and database resources

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact [maintainer email].

---

**Note**: QuartFormer2 is actively under development. Features and APIs may change between versions.
