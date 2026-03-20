from setuptools import setup, Extension

ext_modules = [
    Extension(
        "batching_algorithms",
        ["batching_algorithms.cpp"],
        include_dirs=[],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
        extra_link_args=['-lgomp'],
    ),
]

setup(
    name="batching_algorithms",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.6",
)
