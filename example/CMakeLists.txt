cmake_minimum_required(VERSION 3.5)
project(PanoWeaveExamples LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS videoio imgcodecs highgui)
find_package(argparse)

add_executable(stitch_video stitch_video.cpp)
target_link_libraries(stitch_video PRIVATE PanoWeave opencv_videoio opencv_imgcodecs opencv_highgui)

add_executable(bench_stitcher bench_stitcher.cpp)
target_link_libraries(bench_stitcher PRIVATE PanoWeave opencv_videoio opencv_imgcodecs opencv_highgui)
