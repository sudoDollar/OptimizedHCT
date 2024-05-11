cmake_minimum_required(VERSION 3.5)

project(custom-dataset)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/scratch/ar7996/HPML/tools/libtorch;/scratch/ar7996/HPML/tools/opencv/local")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(native_hct native_hct.cpp)
target_link_libraries(native_hct "${OpenCV_LIBS}")
target_link_libraries(native_hct "${TORCH_LIBRARIES}")
set_property(TARGET native_hct PROPERTY CXX_STANDARD 17)
