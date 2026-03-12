from pathlib import Path
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _unique_sources() -> list[str]:
    root = Path(__file__).resolve().parent
    build_src = root / "build_src"
    build_src.mkdir(exist_ok=True)

    cpp_src = build_src / "pattern_freq_cuda13_v3_host.cpp"
    cu_src = build_src / "pattern_freq_cuda13_v3_kernel.cu"

    shutil.copy2(root / "pattern_freq_cuda13_v3.cpp", cpp_src)
    shutil.copy2(root / "pattern_freq_cuda13_v3.cu", cu_src)
    return [str(cpp_src), str(cu_src)]


setup(
    name="pattern_freq_cuda_v3",
    ext_modules=[
        CUDAExtension(
            name="pattern_freq_cuda_v3",
            sources=_unique_sources(),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.6",
)
