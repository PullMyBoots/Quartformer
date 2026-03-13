from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11


ext_modules = [
    Pybind11Extension(
        "sequence_processor_backend_v2",
        ["sequence_processor_backend_v2.cpp"],
        include_dirs=[pybind11.get_cmake_dir() + "/../../../include"],
        define_macros=[("SEQUENCE_PROCESSOR_BACKEND_MODULE_NAME", "sequence_processor_backend_v2")],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="sequence_processor_backend_v2",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
