from setuptools import Extension, setup

import numpy as np


ext_modules = [
    Extension(
        "quartet_sink",
        ["quartet_sink.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]


setup(
    name="quartet_sink",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.8",
)
