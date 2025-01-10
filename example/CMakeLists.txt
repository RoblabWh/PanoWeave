cmake_minimum_required(VERSION 3.5)
project(PanoWeaverVideo LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS videoio imgcodecs highgui)
find_package(argparse)

add_executable(${PROJECT_NAME} stitch_video.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE PanoWeave opencv_videoio opencv_imgcodecs opencv_highgui)
