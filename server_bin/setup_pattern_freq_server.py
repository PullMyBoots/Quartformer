from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="pattern_freq_cuda_server",
    ext_modules=[
        CUDAExtension(
            name="pattern_freq_cuda_server",
            sources=[
                "pattern_freq_cuda_server.cpp",
                "pattern_freq_cuda13.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
