#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/workloads"
cuda_architectures="${CUDA_ARCHITECTURES:-90}"

cmake_args=(
  -S "${repo_root}/workloads/cuda_microbench"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}"
)

if [[ -n "${CUSPARSELT_ROOT:-}" ]]; then
  cmake_args+=("-DCUSPARSELT_ROOT=${CUSPARSELT_ROOT}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --config Release --parallel

echo "Built ${build_dir}/h2d_d2d_copy"
echo "Built ${build_dir}/read_write_levels"
echo "Built ${build_dir}/tensor_core_burn"
echo "Built ${build_dir}/cuda_core_burn"
echo "Built ${build_dir}/l2_hit_sweep"
echo "Built ${build_dir}/sm_issue_coverage"
echo "Built ${build_dir}/tma_copy"
