from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "merge_phy_cpp",
        [
            "merge_phy_python.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="merge_phy_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)