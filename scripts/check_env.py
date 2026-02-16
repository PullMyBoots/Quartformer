#!/usr/bin/env python3
"""Minimal runtime preflight checks for QuartFormer."""

from pathlib import Path
import argparse
import importlib
import platform
import sys


REQUIRED_PY_MODULES = [
    "torch",
    "triton",
    "numpy",
    "ete3",
    "tqdm",
    "pybind11",
]

REQUIRED_PATHS = [
    "run_qf.py",
    "cpp_source/sequence_processor.cpython-310-x86_64-linux-gnu.so",
    "cpp_source/batching_algorithms.cpython-310-x86_64-linux-gnu.so",
    "cpp_source/cuda13_pattern_freq/pattern_freq_cuda.cpython-310-x86_64-linux-gnu.so",
    "quartet_assemble_method/TREE-QMC/build/tree-qmc",
    "quartet_assemble_method/qfm_java/QFM-FI_unzipped/QFM-FI.jar",
    "model/homogeneous/24/qf1.pt",
    "model/heterogeneous/24/qf1.pt",
    "model/homogeneous/best_mlp_model.pth",
    "model/heterogeneous/best_mlp_model.pth",
]


def check_python_version() -> list[str]:
    errors = []
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        errors.append(
            f"Python {platform.python_version()} detected; Python 3.10 is required "
            "for bundled cp310 extension binaries."
        )
    return errors


def check_modules() -> list[str]:
    errors = []
    for mod in REQUIRED_PY_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            errors.append(f"Missing Python module '{mod}': {exc}")
    return errors


def check_paths() -> list[str]:
    errors = []
    root = Path(__file__).resolve().parent.parent
    for rel in REQUIRED_PATHS:
        p = root / rel
        if not p.exists():
            errors.append(f"Missing required file: {rel}")
    return errors


def check_cuda(strict: bool) -> tuple[list[str], str]:
    errors = []
    try:
        import torch
        if torch.cuda.is_available():
            return errors, "CUDA check: OK (GPU available)"
        msg = "CUDA check: no CUDA device available."
        if strict:
            errors.append("CUDA device is required for inference in this release.")
        return errors, msg
    except Exception as exc:
        if strict:
            errors.append(f"CUDA runtime check failed: {exc}")
        return errors, f"CUDA check skipped: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="QuartFormer environment preflight check")
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Do not fail when CUDA is unavailable (diagnostic mode only)",
    )
    args = parser.parse_args()

    errors = []
    errors.extend(check_python_version())
    errors.extend(check_modules())
    errors.extend(check_paths())
    cuda_errors, cuda_message = check_cuda(strict=not args.allow_no_cuda)
    errors.extend(cuda_errors)
    print(cuda_message)
    if errors:
        print("\n[FAILED] Environment check failed:")
        for err in errors:
            print(f"- {err}")
        print("\nSuggested next steps:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Rebuild extensions if needed:")
        print("  python cpp_source/setup_sequence.py build_ext --inplace")
        print("  python cpp_source/setup_batching.py build_ext --inplace")
        print("  python cpp_source/cuda13_pattern_freq/setup_pytorch.py build_ext --inplace")
        return 1

    print("[OK] Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
