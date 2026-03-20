cmake_minimum_required(VERSION 3.12)
project(merge_phy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找Python和pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 创建Python模块
pybind11_add_module(merge_phy_cpp merge_phy_python.cpp)

# 设置编译选项
target_compile_definitions(merge_phy_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# 链接标准库
target_link_libraries(merge_phy_cpp PRIVATE stdc++fs)