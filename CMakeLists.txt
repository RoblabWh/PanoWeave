cmake_minimum_required(VERSION 3.23)
project(PanoWeave LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
set(USE_CUDA_DEFAULT OFF)
if(CMAKE_CUDA_COMPILER)
    set(USE_CUDA_DEFAULT ON)
endif()

option(USE_CUDA "Build with CUDA support instead of T-API" ${USE_CUDA_DEFAULT})
if(USE_CUDA)
    enable_language(CUDA)
    list(APPEND src_list src/correct_response.cu)
    list(APPEND OPENCV_MODULES cudaarithm cudawarping)
    list(APPEND PUBLIC_LINK_LIBS opencv_cudaarithm opencv_cudawarping)
    list(APPEND COMPILE_DEFINITIONS USE_CUDA)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc ${OPENCV_MODULES})
find_package(spdlog REQUIRED)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
    set(BUILD_AUX ON)
else()
    set(BUILD_AUX OFF)
endif(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

add_subdirectory(thirdparty/basalt-headers)

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
    ${PUBLIC_LINK_LIBS}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILE_DEFINITIONS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

option(PYBINDS "Build Python bindings with pybind11" ${BUILD_AUX})
if(PYBINDS)
    find_package(pybind11 REQUIRED)
    find_package(Python COMPONENTS Interpreter)

    string(TOLOWER ${PROJECT_NAME} PYBIND_PROJECT_NAME)
    pybind11_add_module(${PYBIND_PROJECT_NAME} src/panoweave_pybind.cpp)
    target_link_libraries(${PYBIND_PROJECT_NAME} PUBLIC ${PROJECT_NAME})

    if(Python_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11_stubgen"
            RESULT_VARIABLE PYBIND11_STUBGEN_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )

        if (PYBIND11_STUBGEN_RESULT EQUAL 0)
            message(STATUS "Found pybind11_stubgen, Python stubs will be generated")
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_PROJECT_NAME}.pyi
                COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen -o ${CMAKE_CURRENT_BINARY_DIR} ${PYBIND_PROJECT_NAME}
                DEPENDS ${PYBIND_PROJECT_NAME}
                COMMENT "Generating Python stubs"
            )
            add_custom_target(${PYBIND_PROJECT_NAME}_stubs ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PYBIND_PROJECT_NAME}.pyi)
        else()
            message(STATUS "pybind11_stubgen not found, Python stubs will not be generated")
        endif()
    else()
        message(STATUS "Python interpreter not found, Python stubs will not be generated")
    endif()
endif()

option(EXAMPLES "Build example applications" ${BUILD_AUX})
if(EXAMPLES)
    add_subdirectory(thirdparty/argparse)
    add_subdirectory(example)
endif()

option(DOCS "Generate documentation with Doxygen" ${BUILD_AUX})
if(DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/readme.md)
        doxygen_add_docs(${PROJECT_NAME}Docs ${CMAKE_CURRENT_SOURCE_DIR}/readme.md ${CMAKE_CURRENT_SOURCE_DIR}/include)
    else()
        message(STATUS "Doxygen not found, documentation can not be generated")
    endif()
endif()
