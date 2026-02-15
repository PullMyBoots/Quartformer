cmake_minimum_required(VERSION 3.18)
project(pattern_freq_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(pattern_freq_cuda MODULE
    pattern_freq_cuda13.cpp
    pattern_freq_cuda13.cu
)

target_include_directories(pattern_freq_cuda PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(pattern_freq_cuda PRIVATE ${TORCH_LIBRARIES} pybind11::module CUDA::cudart)

set_target_properties(pattern_freq_cuda PROPERTIES
    PREFIX ""
    OUTPUT_NAME "pattern_freq_cuda"
)

if (DEFINED Python3_EXTENSION_SUFFIX)
  set_target_properties(pattern_freq_cuda PROPERTIES SUFFIX "${Python3_EXTENSION_SUFFIX}")
else()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
    OUTPUT_VARIABLE PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (PY_EXT_SUFFIX)
    set_target_properties(pattern_freq_cuda PROPERTIES SUFFIX "${PY_EXT_SUFFIX}")
  endif()
endif()

target_compile_definitions(pattern_freq_cuda PRIVATE TORCH_EXTENSION_NAME=pattern_freq_cuda)

