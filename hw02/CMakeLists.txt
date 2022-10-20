cmake_minimum_required(VERSION 3.20)
project(hw01)

# set output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set include libraires
include_directories(src)

set(CMAKE_CXX_STANDARD 11)

# add src folder
add_subdirectory(src)

