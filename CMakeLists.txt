cmake_minimum_required(VERSION 3.18)
project(SkyrmionSim LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Default build type ─────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── CUDA architecture (override with -DCMAKE_CUDA_ARCHITECTURES=75 etc.) ───────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# ── Compiler flags ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG   "-G -g")

# ── Find packages ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Optional: HDF5 for binary output (disable with -DUSE_HDF5=OFF)
option(USE_HDF5 "Enable HDF5 output" OFF)
if(USE_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    add_compile_definitions(HAVE_HDF5)
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cu
    src/params.cpp
    src/simulation.cu
    src/forces.cu
    src/integrators.cu
    src/io.cpp
    src/rng.cu
)

add_executable(skyrmion_sim ${SOURCES})

target_include_directories(skyrmion_sim PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(skyrmion_sim PRIVATE
    CUDA::cudart
    CUDA::curand
    CUDA::nvToolsExt
)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(skyrmion_sim PRIVATE stdc++fs)
endif()

if(USE_HDF5)
    target_link_libraries(skyrmion_sim PRIVATE ${HDF5_CXX_LIBRARIES})
    target_include_directories(skyrmion_sim PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

# ── Separate test executable ───────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS skyrmion_sim DESTINATION bin)

# ── Precision ─────────────────────────────────────────────────────────────────
option(SK_FLOAT "Use single-precision (float) arithmetic on GPU" ON)
if(SK_FLOAT)
    target_compile_definitions(skyrmion_sim PRIVATE SK_FLOAT)
endif()

# ── Print summary ─────────────────────────────────────────────────────────────
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch        : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "HDF5 support     : ${USE_HDF5}")
message(STATUS "Build tests      : ${BUILD_TESTS}")
message(STATUS "Precision        : ${SK_FLOAT}")
message(STATUS "  (set -DSK_FLOAT=ON for float, OFF for double)")
