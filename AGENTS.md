# Repository Guidelines

## Project Structure & Module Organization
- `run_qf.py` is the main entry point for inference; core model code lives in `model.py`, `utils.py`, and `sparse_attn_kernel.py`.
- `model/` stores pretrained weights organized by mode (`homogeneous/`, `heterogeneous/`) and taxa count.
- `cpp_source/` contains native extensions and CUDA operators (`setup_sequence.py`, `setup_batching.py`, `cuda13_pattern_freq/`).
- `quartet_assemble_method/` contains tree assembly backends (`TREE-QMC`, `qfm_java`).
- `examples/` provides runnable sample datasets and `run_examples.sh`; `docs/` holds benchmark and accuracy notes.
- `software/` is vendored third-party code; avoid broad edits there unless dependency maintenance is required.

## Build, Test, and Development Commands
- `python check_install.py`: validate Python, CUDA, extensions, model files, and external tools.
- `pip install -r requirements.txt`: install runtime Python dependencies.
- `python run_qf.py --phy <input.phy> --out <output.nwk>`: run inference.
- `cd examples && ./run_examples.sh`: smoke-test end-to-end behavior on bundled datasets.
- Rebuild extensions when environment changes:
  - `cd cpp_source && python setup_sequence.py build_ext --inplace`
  - `cd cpp_source && python setup_batching.py build_ext --inplace`
  - `cd cpp_source/cuda13_pattern_freq && python setup.py build_ext --inplace`

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8-style naming (`snake_case` for functions/variables, `PascalCase` for classes).
- Prefer type hints for new/modified public functions and keep docstrings concise and behavior-focused.
- Keep CLI argument names explicit (`--task-type`, `--infer-batch-size`) and consistent with existing patterns.
- C++ extension filenames and setup scripts should follow existing module naming (`sequence_processor`, `batching_algorithms`).

## Testing Guidelines
- No formal unit-test suite is currently configured at the repository root.
- Minimum validation for code changes:
  - run `python check_install.py`
  - run at least one example from `examples/` (24 taxa recommended for quick checks)
  - for inference logic changes, compare output tree metrics with `--ref-tree` where available.

## Commit & Pull Request Guidelines
- Current commit style is short, imperative summaries (for example, `Add installation and packaging files for user distribution`).
- Keep commits scoped to one concern and describe user-visible impact.
- PRs should include:
  - purpose and changed paths
  - exact validation commands executed
  - environment details when relevant (Python/CUDA/GPU)
  - before/after metrics or runtime notes for algorithmic changes.
