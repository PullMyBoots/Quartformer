# Installation Guide for QuartFormer2

This guide will help you install QuartFormer2 on your system.

## System Requirements

### Hardware
- **CPU**: x86_64 processor (multi-core recommended)
- **GPU**: NVIDIA GPU with CUDA support (>8GB VRAM recommended)
- **RAM**: 16GB+ (32GB+ recommended for large datasets)

### Software
- **Python**: 3.10.x (exactly - precompiled extensions are Python 3.10 specific)
- **CUDA**: 12.6+ (tested with CUDA 12.6/12.8)
- **OS**: Linux (tested on Ubuntu 20.04/22.04, WSL2)

---

## Quick Start Installation

### Step 1: Verify Python Version

```bash
python --version
# Should show Python 3.10.x
```

If you don't have Python 3.10, install it:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv quartformer_env
source quartformer_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If PyTorch doesn't detect CUDA, install CUDA-enabled PyTorch:
```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Installation

```bash
# Test C++ extensions
python -c "import cpp_source.sequence_processor as sp; print('✓ sequence_processor OK')"
python -c "import cpp_source.pattern_freq_cuda as pfc; print('✓ pattern_freq_cuda OK')"
python -c "import cpp_source.batching_algorithms as ba; print('✓ batching_algorithms OK')"

# Test quartet assemblers
ls -lh quartet_assemble_method/TREE-QMC/build/tree-qmc
ls -lh quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar
```

### Step 4: Run Example

```bash
cd examples
./run_examples.sh
```

---

## Detailed Installation

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note**: The precompiled C++ extensions are for **Python 3.10 on Linux x86_64** only.

---

## Recompiling C++ Extensions (Advanced)

If you're not using Python 3.10 or Linux x86_64, you need to recompile:

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cuda-cudart-dev-12-x python3-dev

# Install PyTorch with CUDA support first
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Compile Sequence Processor

```bash
cd cpp_source
python setup_sequence.py build_ext --inplace
cd ..
```

### Compile Batching Algorithms

```bash
cd cpp_source
python setup_batching.py build_ext --inplace
cd ..
```

### Compile CUDA Pattern Frequency

```bash
cd cpp_source/cuda13_pattern_freq
python setup.py build_ext --inplace
cd ../..
```

### Compile Merge PHY Tool (Optional)

```bash
cd cpp_source
python setup.py build_ext --inplace
cd ..
```

### Compile Sparse Attention Kernel (Optional)

```bash
cd sparse_attn_kernel
python setup.py build_ext --inplace
cd ..
```

### Compile TREE-QMC

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
cd ../..
```

---

## Troubleshooting

### "Module not found" errors

**Problem**: Cannot import C++ extensions

**Solution**:
1. Check Python version is exactly 3.10.x
2. Verify you're in the project root directory
3. Try recompiling C++ extensions (see above)

### CUDA errors

**Problem**: CUDA not available or errors

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### TREE-QMC not found

**Problem**: `tree-qmc executable not found`

**Solution**: Ensure TREE-QMC is compiled:
```bash
ls -lh quartet_assemble_method/TREE-QMC/build/tree-qmc
```

If missing, compile it following the instructions above.

### Java errors (QFM-FI)

**Problem**: QFM-FI fails with Java errors

**Solution**: Ensure Java JRE is installed:
```bash
java -version
```

If not installed:
```bash
# Ubuntu/Debian
sudo apt-get install default-jre
```

### Out of memory errors

**Problem**: GPU or system out of memory

**Solution**:
- Reduce batch size: `python run_qf.py --phy data.phy --infer-batch-size 16`
- Use smaller model or dataset
- Close other GPU-intensive applications

---

## Environment Check Script

Run this to verify your installation:

```python
# check_install.py
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import ete3
    print(f"✓ Ete3 {ete3.__version__}")
except ImportError as e:
    print(f"✗ Ete3: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import cpp_source.sequence_processor as sp
    print(f"✓ sequence_processor (C++ extension)")
except Exception as e:
    print(f"✗ sequence_processor: {e}")

try:
    import cpp_source.pattern_freq_cuda as pfc
    print(f"✓ pattern_freq_cuda (CUDA extension)")
except Exception as e:
    print(f"✗ pattern_freq_cuda: {e}")

try:
    import cpp_source.batching_algorithms as ba
    print(f"✓ batching_algorithms (C++ extension)")
except Exception as e:
    print(f"✗ batching_algorithms: {e}")

import os
tree_qmc = "quartet_assemble_method/TREE-QMC/build/tree-qmc"
if os.path.exists(tree_qmc):
    print(f"✓ TREE-QMC: {tree_qmc}")
else:
    print(f"✗ TREE-QMC not found at {tree_qmc}")

qfm_jar = "quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar"
if os.path.exists(qfm_jar):
    print(f"✓ QFM-FI: {qfm_jar}")
else:
    print(f"✗ QFM-FI not found at {qfm_jar}")
```

Run with: `python check_install.py`

---

## Next Steps

After installation:
1. Read [README.md](README.md) for usage instructions
2. Try the examples in [examples/README.md](examples/README.md)
3. Check [docs/benchmarks.md](docs/benchmarks.md) for performance comparisons

---

## Getting Help

If you encounter issues not covered here:
1. Check the [troubleshooting section](README.md#troubleshooting) in README
2. Open an issue on GitHub
3. Contact: [maintainer email]
