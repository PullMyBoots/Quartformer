# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuartFormer2 is a deep learning-based phylogenetic tree inference system that uses transformer models with sparse attention to predict species trees from multiple sequence alignments (MSA). The system processes DNA sequence data through GPU-accelerated pattern frequency calculations, applies transformer models for quartet inference, and assembles final trees using quartet max-cut (QMC) methods.

## Core Architecture

### Main Inference Pipeline

The system follows a multi-stage inference pipeline:

1. **Input Processing**: PHY format alignments → tensors via `sequence_processor` (C++ extension)
2. **Pattern Extraction**: GPU-accelerated DNA pattern frequency calculation via `pattern_freq_cuda`
3. **Quartet Inference**: `QuartFormer` transformer predicts quartet topologies (3 classes)
4. **Confidence Scoring**: `QuartFormer2` (optional) provides confidence weights
5. **Tree Assembly**: TREE-QMC or QFM-FI assembles weighted quartets into species tree
6. **Polytomy Resolution**: `MLP` model resolves multi-furcating nodes for large datasets (>36 species)

### Key Components

- **`model.py`** (58KB): Core implementation containing transformer models, inference pipeline, and utilities
  - `QuartFormer`: Sparse attention transformer for quartet classification
  - `QuartFormer2`: Confidence scoring model
  - `MLP`/`MLP2`: Models for subtree resolution and polytomy handling
  - `run_QF_framework()`: Main inference entry point supporting 8-192+ species

- **`run_qf.py`**: Enhanced inference script with batch processing and automated polytomy resolution
  - Supports species ≥24 only (vs. model.py which handles ≥8)
  - Mean-weight aggregation for duplicate quartets
  - Integrated MLP-based polytomy resolution

- **`sparse_attn_kernel.cpp`**: Custom CUDA kernel for sparse attention operations

- **`sequence_processor.cpp`**: C++ extension for PHY file parsing and tensor conversion

- **`batching_algorithms.cpp`**: C++ implementation of pair-balanced block design (V5) for efficient large-scale processing

### Model Structure

**QuartFormer Architecture:**
- Token dimension: 256
- Attention heads: 16
- Hidden dimension: 16
- Encoder layers: 3
- Output: 3-class quartet topology probabilities

**Input Features:**
- Species one-hot encoding (species_num dimensions)
- Pattern frequencies (128 dimensions): GPU-calculated DNA site patterns
- Sparse attention layout based on shared species in quartets

**Pre-trained Models:**
- Located in `model/gene/` and `model/multilocus/`
- Separate models for different species counts (24, 48, 96, etc.)
- Contains: model weights (qf1.pt, qf2.pt), quartet matrices, species encodings

## Build Commands

### C++ Extensions

All C++ extensions are pre-compiled as `.so` files in `cpp_source/`. To rebuild:

```bash
# Sequence processor (PHY parsing)
cd cpp_source
python setup_sequence.py build_ext --inplace

# Batching algorithms (V5 block design)
python setup_batching.py build_ext --inplace

# Merge PHY tool
python setup.py build_ext --inplace

# CUDA pattern frequency
cd cuda13_pattern_freq
python setup.py build_ext --inplace
```

### TREE-QMC Assembler

TREE-QMC source is in `quartet_assemble_method/TREE-QMC/`. To compile:

```bash
cd quartet_assemble_method/TREE-QMC

# Build MQLib dependency
cd external/MQLib
make

# Build TREE-QMC
cd ../../
mkdir -p build build_local
cd build
g++ -std=c++11 -O2 \
    -I ../external/MQLib/include \
    -I ../external/toms743 \
    -o tree-qmc \
    ../src/*.cpp \
    ../external/toms743/toms743.cpp \
    ../external/MQLib/bin/MQLib.a \
    -lm \
    '-DVERSION="local"'

# Copy to expected location
cp tree-qmc ../build_local/tree-qmc
```

**Note**: The repository includes pre-compiled binaries in `quartet_assemble_method/TREE-QMC/build/` and `build_local/`.

## Running Inference

### Using run_qf.py

```python
import run_qf2_standalone

result = run_qf2_standalone.run_QF_framework_2(
    phy_path='data/alignment.phy',
    output_tree_path='output.nwk',
    task_type='gene',          # 'gene' or 'multilocus'
    k_param=3.0,               # Block design parameter
    run_mode='fast',           # 'fast' (QMC) or 'regular' (QFM-FI for ≤96 species)
    infer_batch_size=32        # Batch size for inference
)
```

### Using model.py directly

```python
from model import run_QF_framework

result = run_QF_framework(
    phy_path='data.phy',
    output_tree_path='output.nwk',
    task_type='gene',
    use_confidence=True,       # Use QuartFormer2 for confidence
    calculated_load=3.0,
    fast_mode=True,
    quartet_assembler='qmc'    # 'qmc' or 'qfm_fi'
)
```

### Command-line benchmark

```bash
python run_qf.py
```

The script includes built-in benchmarking code at the bottom (lines 501-532) that tests multiple species counts and sequence lengths.

## Quartet Assembly Methods

The system supports two quartet assembly methods:

1. **TREE-QMC** (`quartet_assembler='qmc'`): C++ implementation using Max-Cut optimization
   - Default for large datasets (>96 species in regular mode)
   - Always used in 'fast' mode
   - Input format: `t0,t1|t2,t3:weight` (compact) or `((t0,t1),(t2,t3));weight` (newick)

2. **QFM-FI** (`quartet_assembler='qfm_fi'`): Java-based implementation
   - Used for ≤96 species in regular mode
   - Requires JVM with optimized settings (88GB max heap configured in QFMFI_JVM_ARGS)
   - Input format: `((t0,t1),(t2,t3)); weight`

The assembler is automatically selected based on dataset size and run mode.

## Species Count Handling

- **8-36 species** (model.py): Loads model matching exact species count
- **≥24 species** (run_qf.py): Only supports ≥24, uses 24-species model
- **>36 species** (model.py): Uses 24-species model for main tree, then resolves polytomies with MLP
- **>96 species** (run_qf.py): Uses batched V5 block design with 24-species model

## GPU Requirements

- **CUDA required**: Pattern frequency calculation is GPU-only
- **Check availability**: `torch.cuda.is_available()` must return True
- **Import check**: `import pattern_freq_cuda` must succeed
- The system will raise RuntimeError if CUDA is unavailable

## Tree Comparison Metrics

The `compute_tree_difference()` function supports two metrics:

- **`mode='rf'`**: Robinson-Foulds distance (normalized 0-1, lower is better)
- **`mode='quartet'`**: Quartet accuracy (0-1, higher is better)

Both metrics require Ete3 Tree objects and handle unrooted trees.

## Git Workflow

This repository uses two branches:

- **`main`**: Full development version with all experimental scripts, test data, and tools
- **`release`**: Clean production version with only core files needed by users

When contributing:
- Development work happens on `main`
- User-facing changes are cherry-picked or merged to `release`
- `release` branch excludes: `benchmark_aster_fasttree.py`, `data_maker.py`, `software/`, `output/`, `temp/`, `test_cpp/`

## Important Implementation Details

### Sparse Attention Layout

The sparse attention mechanism uses quartet-based connectivity: tokens attend only to other tokens that share species in the same quartet. This is encoded in `coeff_blocks` and `quartet_matrix` loaded from pre-trained model directories.

### Pattern Frequency Calculation

DNA pattern frequencies are computed on GPU for all 4-taxa subsets in a batch. The output is a 256-dimensional vector per quartet encoding site pattern frequencies.

### Weighted Quartet Aggregation

Duplicate quartets (from overlapping blocks) have their weights aggregated using NumPy vectorized operations (`_accumulate_weighted_quartets_numpy`) for performance.

### Polytpomy Resolution Flow

For datasets >36 species with polytomies:
1. `find_polytomy_representative_leaves()` identifies multi-furcating nodes
2. For each polytomy, select representative leaves from each branch
3. `MLP` infers complete tree on representatives
4. Guide tree used to resolve polytomy structure
5. Iterates until all polytomies resolved or max iterations (20) reached
