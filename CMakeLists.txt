cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main_cpp main.cpp)
add_executable(main_cu main.cu)

target_link_libraries(main_cpp "${TORCH_LIBRARIES}")
target_link_libraries(main_cu "${TORCH_LIBRARIES}")

set_property(TARGET main_cpp PROPERTY CXX_STANDARD 17)
set_property(TARGET main_cu PROPERTY CXX_STANDARD 17)