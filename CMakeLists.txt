cmake_minimum_required(VERSION 3.28) #设置CMake最小版本

project (PracticeForCUDA_cmake) #设置工程名

add_executable(Main_cmake Main.cpp) #生成可执行文件

set(MINI_VERSION="3.28")

target_compile_definitions(Main_cmake PRIVATE MINI_VERSION_CPP="${MINI_VERSION}")