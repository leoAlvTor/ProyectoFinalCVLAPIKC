cmake_minimum_required(VERSION 3.17)
project(ProyectoFinalCV)

SET(OpenCV_DIR /home/leo_mx/opencv-4.5.1/build/)
set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED PATHS "/home/leo_mx/opencv-4.5.1/build/")
find_package(PythonLibs REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(ProyectoFinalCV main.cpp Utilidades.cpp Utilidades.h)
target_link_libraries( ProyectoFinalCV ${OpenCV_LIBS} ${PYTHON_LIBRARIES})