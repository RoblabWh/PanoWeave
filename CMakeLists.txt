cmake_minimum_required(VERSION 3.23)
project(PanoWeave LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
find_package(spdlog REQUIRED)
find_package(pybind11 REQUIRED)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    message(${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
    message(${CMAKE_CXX_FLAGS_DEBUG})
endif(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

add_subdirectory(thirdparty/basalt-headers)

# file(GLOB_RECURSE src_list LIST_DIRECTORY false src/*.cpp)
list(APPEND src_list src/panoweave.cpp)
file(GLOB_RECURSE hdr_list LIST_DIRECTORY false include/*.hpp)
file(GLOB_RECURSE cl_list LIST_DIRECTORY false src/*.cl)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ocl_kernels.cpp"  # don't add .hpp file here to optimize build process
    COMMAND ${CMAKE_COMMAND} "-DMODULE_NAME=${PROJECT_NAME}" "-DCL_DIR=${CMAKE_CURRENT_LIST_DIR}/src" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/ocl_kernels.cpp" -P "${CMAKE_CURRENT_SOURCE_DIR}/cl2cpp.cmake"
    DEPENDS ${cl_list} "${CMAKE_CURRENT_SOURCE_DIR}/cl2cpp.cmake"
    COMMENT "Processing OpenCL kernels"
)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/ocl_kernels.hpp" PROPERTIES GENERATED TRUE)
add_custom_target(cl2cpp DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ocl_kernels.cpp")
list(APPEND src_list "${CMAKE_CURRENT_BINARY_DIR}/ocl_kernels.cpp")

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME}
    PRIVATE ${src_list}
    PUBLIC FILE_SET HEADERS BASE_DIRS include FILES ${hdr_list} # VERSION 3.23
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    spdlog::spdlog_header_only
    PUBLIC
    opencv_core opencv_imgproc
    basalt::basalt-headers
)
target_include_directories(${PROJECT_NAME} PRIVATE build)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

option(BUILD_EXAMPLES "Build example applications for PanoWeave" ON)
if(${BUILD_EXAMPLES})
    add_subdirectory(example)
endif(${BUILD_EXAMPLES})
