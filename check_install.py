#!/usr/bin/env python3
"""
Installation check script for QuartFormer2.
Run this script to verify all dependencies are correctly installed.
"""

import sys
import os

print("=" * 60)
print("QuartFormer2 Installation Check")
print("=" * 60)
print()

all_ok = True

# Python version
print(f"Python version: {sys.version}")
if sys.version_info[:2] != (3, 10):
    print(f"⚠️  WARNING: Python 3.10.x is required (current: {sys.version_info[0]}.{sys.version_info[1]})")
    all_ok = False
else:
    print("✓ Python version OK")
print()

# PyTorch
print("Checking PyTorch...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ⚠️  WARNING: CUDA is not available. QuartFormer2 requires CUDA!")
        all_ok = False
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    print("  Install with: pip install torch")
    all_ok = False
print()

# Ete3
print("Checking Ete3...")
try:
    import ete3
    print(f"✓ Ete3 {ete3.__version__}")
except ImportError as e:
    print(f"✗ Ete3: {e}")
    print("  Install with: pip install ete3")
    all_ok = False
print()

# NumPy
print("Checking NumPy...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")
    print("  Install with: pip install numpy")
    all_ok = False
print()

# Tqdm
print("Checking Tqdm...")
try:
    import tqdm
    print(f"✓ Tqdm installed")
except ImportError:
    print("✗ Tqdm not found")
    print("  Install with: pip install tqdm")
    all_ok = False
print()

# C++ Extensions
print("Checking C++ Extensions...")
try:
    import cpp_source.sequence_processor as sp
    print("✓ sequence_processor (C++ extension)")
except Exception as e:
    print(f"✗ sequence_processor: {e}")
    print("  This extension requires Python 3.10 on Linux x86_64")
    all_ok = False
print()

try:
    import cpp_source.pattern_freq_cuda as pfc
    print("✓ pattern_freq_cuda (CUDA extension)")
except Exception as e:
    print(f"✗ pattern_freq_cuda: {e}")
    print("  This extension requires CUDA and Python 3.10")
    all_ok = False
print()

try:
    import cpp_source.batching_algorithms as ba
    print("✓ batching_algorithms (C++ extension)")
except Exception as e:
    print(f"✗ batching_algorithms: {e}")
    all_ok = False
print()

# External tools
print("Checking External Tools...")

tree_qmc_path = "quartet_assemble_method/TREE-QMC/build/tree-qmc"
if os.path.exists(tree_qmc_path):
    print(f"✓ TREE-QMC: {tree_qmc_path}")
else:
    print(f"✗ TREE-QMC not found at {tree_qmc_path}")
    print("  See INSTALL.md for compilation instructions")
    all_ok = False
print()

qfm_jar_path = "quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar"
if os.path.exists(qfm_jar_path):
    print(f"✓ QFM-FI: {qfm_jar_path}")
else:
    print(f"✗ QFM-FI not found at {qfm_jar_path}")
    all_ok = False
print()

# Model files
print("Checking Model Files...")
model_dirs = [
    "model/heterogeneous/24",
    "model/homogeneous/24"
]
for model_dir in model_dirs:
    if os.path.exists(model_dir):
        required_files = ["qf1.pt", "qf2.pt", "best_mlp_model.pth",
                         "coeff_blocks.pt", "quartet_matrix.pt", "species_encoding.pt"]
        all_present = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        if all_present:
            print(f"✓ {model_dir} (all model files present)")
        else:
            print(f"⚠️  {model_dir} (some model files missing)")
    else:
        print(f"✗ {model_dir} not found")
print()

# Java
print("Checking Java...")
try:
    result = os.popen("java -version 2>&1").read()
    if "java version" in result.lower() or "openjdk" in result.lower():
        print("✓ Java JRE installed")
    else:
        print("⚠️  Java status unclear")
except:
    print("⚠️  Could not check Java version")
print()

# Summary
print("=" * 60)
if all_ok:
    print("✓ All checks passed! Your installation looks good.")
    print()
    print("You can now run QuartFormer2:")
    print("  python run_qf.py --phy data.phy --out output.nwk")
else:
    print("⚠️  Some checks failed. Please install missing dependencies.")
    print()
    print("See INSTALL.md for detailed installation instructions.")
print("=" * 60)
