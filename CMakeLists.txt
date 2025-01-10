cmake_minimum_required(VERSION 3.23)
project(PanoWeave LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc gapi)
find_package(spdlog REQUIRED)
find_package(pybind11 REQUIRED)

subdirs(thirdparty)

# file(GLOB_RECURSE src_list LIST_DIRECTORY false src/*.cpp)
list(APPEND src_list src/panoweave.cpp)
file(GLOB_RECURSE hdr_list LIST_DIRECTORY false include/*.hpp)

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME}
    PRIVATE ${src_list}
    PUBLIC FILE_SET HEADERS BASE_DIRS include FILES ${hdr_list} # VERSION 3.23
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    spdlog::spdlog_header_only
    PUBLIC
    opencv_core opencv_gapi
    basalt::basalt-headers
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
target_compile_options(${PROJECT_NAME} PUBLIC -march=native)

option(BUILD_EXAMPLES "Build example applications for PanoWeave" ON)
if(${BUILD_EXAMPLES})
    subdirs(example)
endif()
