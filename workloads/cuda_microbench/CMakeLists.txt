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

find_path(
  CUSPARSELT_INCLUDE_DIR
  NAMES cusparseLt.h
  HINTS ENV CUSPARSELT_ROOT
  PATH_SUFFIXES include
)
find_library(
  CUSPARSELT_LIBRARY
  NAMES cusparseLt
  HINTS ENV CUSPARSELT_ROOT
  PATH_SUFFIXES lib lib64
)

if(CUSPARSELT_INCLUDE_DIR AND CUSPARSELT_LIBRARY)
  target_include_directories(tensor_core_burn PRIVATE "${CUSPARSELT_INCLUDE_DIR}")
  target_link_libraries(tensor_core_burn PRIVATE "${CUSPARSELT_LIBRARY}")
  target_compile_definitions(tensor_core_burn PRIVATE HAVE_CUSPARSELT=1)
  message(STATUS "cuSPARSELt enabled for tensor_core_burn: ${CUSPARSELT_LIBRARY}")
else()
  message(STATUS "cuSPARSELt not found; structured_2to4 mode will fail clearly at runtime")
endif()

if(GPU_POWER_ENABLE_EXPERIMENTAL_TMA)
  target_compile_definitions(tma_copy PRIVATE GPU_POWER_ENABLE_EXPERIMENTAL_TMA=1)
endif()
