import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    build_dir = root / "build"
    build_dir.mkdir(exist_ok=True)

    python = shutil.which("python") or "python"

    torch_prefix = subprocess.check_output(
        [python, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
        text=True,
    ).strip()

    env = os.environ.copy()
    env["CMAKE_PREFIX_PATH"] = torch_prefix + (os.pathsep + env["CMAKE_PREFIX_PATH"] if env.get("CMAKE_PREFIX_PATH") else "")

    subprocess.check_call(
        ["cmake", "-S", str(root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        env=env,
    )
    subprocess.check_call(["cmake", "--build", str(build_dir), "-j"], env=env)

    # Copy built module next to repo root for `import pattern_freq_cuda`.
    out = next(build_dir.glob("pattern_freq_cuda*.so"), None)
    if out is None:
        out = next((p for p in build_dir.rglob("pattern_freq_cuda*.so")), None)
    if out is None:
        raise RuntimeError("build succeeded but no pattern_freq_cuda*.so found")

    repo_root = root.parent
    shutil.copy2(out, repo_root / out.name)
    print(f"Built: {out}")
    print(f"Copied to: {repo_root / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

