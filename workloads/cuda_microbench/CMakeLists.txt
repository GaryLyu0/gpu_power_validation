cmake_minimum_required(VERSION 3.22)

project(gpu_power_cuda_microbench LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "90;100")
endif()

option(GPU_POWER_ENABLE_EXPERIMENTAL_TMA "Enable experimental architecture-specific TMA code paths" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(h2d_d2d_copy h2d_d2d_copy.cu)
add_executable(read_write_levels read_write_levels.cu)
add_executable(tensor_core_burn tensor_core_burn.cu)
add_executable(cuda_core_burn cuda_core_burn.cu)
add_executable(l2_hit_sweep l2_hit_sweep.cu)
add_executable(sm_issue_coverage sm_issue_coverage.cu)
add_executable(tma_copy tma_copy.cu)

foreach(target h2d_d2d_copy read_write_levels tensor_core_burn cuda_core_burn l2_hit_sweep sm_issue_coverage tma_copy)
  set_target_properties(
    ${target}
    PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  target_compile_options(
    ${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endforeach()

target_link_libraries(h2d_d2d_copy PRIVATE CUDA::cudart)
target_link_libraries(read_write_levels PRIVATE CUDA::cudart)
target_link_libraries(cuda_core_burn PRIVATE CUDA::cudart)
target_link_libraries(tensor_core_burn PRIVATE CUDA::cudart)
target_link_libraries(l2_hit_sweep PRIVATE CUDA::cudart)
target_link_libraries(sm_issue_coverage PRIVATE CUDA::cudart)
target_link_libraries(tma_copy PRIVATE CUDA::cudart)
target_link_libraries(tensor_core_burn PRIVATE CUDA::cublas)

set(CUTLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include")
set(CUTLASS_SM90_GMMA_HEADER "${CUTLASS_INCLUDE_DIR}/cute/arch/mma_sm90_gmma.hpp")
set(CUTLASS_SM90_SMEM_HELPER "${CUTLASS_INCLUDE_DIR}/cutlass/gemm/collective/builders/sm90_common.inl")
if(EXISTS "${CUTLASS_INCLUDE_DIR}/cute/tensor.hpp")
  target_include_directories(tensor_core_burn PRIVATE "${CUTLASS_INCLUDE_DIR}")
  target_compile_definitions(tensor_core_burn PRIVATE HAVE_CUTLASS_CUTE=1)
  message(STATUS "CUTLASS/CuTe tile burn support: ON")

  if(EXISTS "${CUTLASS_SM90_GMMA_HEADER}" AND EXISTS "${CUTLASS_SM90_SMEM_HELPER}")
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
       CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.0")
      set(GPU_POWER_NVCC_SUPPORTS_SM90A ON)
    else()
      set(GPU_POWER_NVCC_SUPPORTS_SM90A OFF)
    endif()
  else()
    set(GPU_POWER_NVCC_SUPPORTS_SM90A OFF)
    message(STATUS "Hopper WGMMA support: OFF (vendored CUTLASS lacks required SM90 GMMA headers)")
  endif()
  if(GPU_POWER_NVCC_SUPPORTS_SM90A)
    add_library(tensor_core_wgmma_sm90a OBJECT tensor_core_wgmma_sm90.cu)
    set_target_properties(
      tensor_core_wgmma_sm90a
      PROPERTIES
        CUDA_ARCHITECTURES OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(
      tensor_core_wgmma_sm90a
      PRIVATE
        "${CUTLASS_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_compile_options(
      tensor_core_wgmma_sm90a
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_90a,code=sm_90a>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_90a,code=compute_90a>
    )
    target_sources(tensor_core_burn PRIVATE $<TARGET_OBJECTS:tensor_core_wgmma_sm90a>)
    target_compile_definitions(tensor_core_burn PRIVATE HAVE_WGMMA_SM90A=1)
    message(STATUS "Hopper WGMMA support: ON (tensor_core_wgmma_sm90.cu -> sm_90a + compute_90a)")
  elseif(EXISTS "${CUTLASS_SM90_GMMA_HEADER}" AND EXISTS "${CUTLASS_SM90_SMEM_HELPER}")
    message(STATUS "Hopper WGMMA support: OFF (NVIDIA CUDA compiler 12.0+ is required for sm_90a)")
  endif()
else()
  message(STATUS "CUTLASS/CuTe tile burn support: OFF")
  message(STATUS "Hopper WGMMA support: OFF (vendored CUTLASS/CuTe headers unavailable)")
endif()

set(CUSPARSELT_ROOT "" CACHE PATH "Optional cuSPARSELt installation prefix")

set(_CUSPARSELT_PREFIX_HINTS)
if(CUSPARSELT_ROOT)
  list(APPEND _CUSPARSELT_PREFIX_HINTS "${CUSPARSELT_ROOT}")
endif()
if(DEFINED ENV{CUSPARSELT_ROOT} AND NOT "$ENV{CUSPARSELT_ROOT}" STREQUAL "")
  list(APPEND _CUSPARSELT_PREFIX_HINTS "$ENV{CUSPARSELT_ROOT}")
endif()

set(_CUSPARSELT_INCLUDE_HINTS)
set(_CUSPARSELT_LIBRARY_HINTS)
foreach(prefix IN LISTS _CUSPARSELT_PREFIX_HINTS)
  list(APPEND _CUSPARSELT_INCLUDE_HINTS "${prefix}/include")
  list(APPEND _CUSPARSELT_LIBRARY_HINTS "${prefix}/lib" "${prefix}/lib64" "${prefix}/lib/x64")
endforeach()
if(CUDAToolkit_TARGET_DIR)
  list(APPEND _CUSPARSELT_INCLUDE_HINTS "${CUDAToolkit_TARGET_DIR}/include")
  list(APPEND _CUSPARSELT_LIBRARY_HINTS
    "${CUDAToolkit_TARGET_DIR}/lib"
    "${CUDAToolkit_TARGET_DIR}/lib64"
    "${CUDAToolkit_TARGET_DIR}/lib/x64"
  )
endif()
if(CUDAToolkit_INCLUDE_DIRS)
  list(APPEND _CUSPARSELT_INCLUDE_HINTS ${CUDAToolkit_INCLUDE_DIRS})
endif()
if(CUDAToolkit_LIBRARY_DIR)
  list(APPEND _CUSPARSELT_LIBRARY_HINTS "${CUDAToolkit_LIBRARY_DIR}")
endif()

find_path(CUSPARSELT_INCLUDE_DIR NAMES cusparseLt.h HINTS ${_CUSPARSELT_INCLUDE_HINTS})
find_library(CUSPARSELT_LIBRARY NAMES cusparseLt HINTS ${_CUSPARSELT_LIBRARY_HINTS})

if(CUSPARSELT_INCLUDE_DIR AND CUSPARSELT_LIBRARY)
  target_include_directories(tensor_core_burn PRIVATE "${CUSPARSELT_INCLUDE_DIR}")
  target_link_libraries(tensor_core_burn PRIVATE "${CUSPARSELT_LIBRARY}")
  target_compile_definitions(tensor_core_burn PRIVATE HAVE_CUSPARSELT=1)
  message(STATUS "cuSPARSELt support: ON")
  message(STATUS "cuSPARSELt include: ${CUSPARSELT_INCLUDE_DIR}")
  message(STATUS "cuSPARSELt library: ${CUSPARSELT_LIBRARY}")
else()
  message(STATUS "cuSPARSELt support: OFF")
endif()

if(GPU_POWER_ENABLE_EXPERIMENTAL_TMA)
  target_compile_definitions(tma_copy PRIVATE GPU_POWER_ENABLE_EXPERIMENTAL_TMA=1)
endif()
