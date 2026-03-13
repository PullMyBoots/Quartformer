from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="pattern_freq_cuda_backend_v2",
    ext_modules=[
        CUDAExtension(
            name="pattern_freq_cuda_backend_v2",
            sources=[
                "pattern_freq_cuda_backend_v2_host.cpp",
                "pattern_freq_cuda_backend_v2.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
