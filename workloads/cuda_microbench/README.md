# CUDA Microbenchmarks

This directory contains independent CUDA workloads owned by this framework.
They do not depend on or modify `third_party`.

## h2d_d2d_copy

`h2d_d2d_copy.cu` implements:

- H2D one-way continuous transfer using pinned host memory
- D2D global-memory to global-memory continuous copy on one GPU
- CUDA event based bandwidth measurement
- JSON-lines stdout suitable for runner summary parsing
- Optional pre-sweep byte sizes

Build on the H100/B200 server:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
```

If cuSPARSELt is installed outside the CUDA Toolkit search path, point CMake at
it before building `power_gpu_op_tc_005` support:

```bash
export CUSPARSELT_ROOT=/path/to/cusparselt
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
```

Override architectures when needed:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
CUDA_ARCHITECTURES=100 bash scripts/build_workloads.sh
```

Run:

```bash
./build/workloads/h2d_d2d_copy --device 0 --mode h2d --bytes 268435456 --warmup-sec 10 --steady-sec 60
./build/workloads/h2d_d2d_copy --device 0 --mode d2d --bytes 268435456 --warmup-sec 10 --steady-sec 60
```

## read_write_levels

`read_write_levels.cu` implements the `PWR-MEM-003` read/write load-level sweep.
The buffer size and launch shape stay comparable across levels; `--load-factor`
changes the active memory range.

```bash
./build/workloads/read_write_levels --device 0 --mode read --load-factor 0.125 --buffer-mb 4096 --warmup-sec 10 --steady-sec 60
./build/workloads/read_write_levels --device 0 --mode write --load-factor 1.0 --buffer-mb 4096 --warmup-sec 10 --steady-sec 60
```

## tensor_core_burn

`tensor_core_burn.cu` implements the base Tensor Core cases with two engines:

- `--engine cublas` keeps the original cuBLAS BF16 GEMM behavior and remains
  the default for backward compatibility.
- `--engine wmma_persistent` launches persistent CTAs and uses device-side
  `clock64` control so short active/idle periods do not depend on CPU-side GEMM
  launch timing.
- `--engine cutlass_tile_burn` is an experimental CUTLASS/CuTe MMA atom based
  synthetic Tensor Core burn. It calls the SM80 BF16 MMA atom directly, does not
  use top-level `cutlass::gemm::device::Gemm`, does not load real A/B matrices,
  and does not use shared-memory A/B tiles.

CUTLASS integration remains optional for a later step and is not required by
these cases.

Current Tensor Core limitations:

- BF16 is the baseline implementation.
- FP16, CUTLASS, FP8, and FP4 are future work.
- With `--engine cublas`, `--active-sm-fraction` maps to a cuBLAS SM-count
  target hint where supported.
- With `--engine wmma_persistent`, `--active-sm-fraction` maps to persistent
  CTA coverage using one long-lived CTA per requested SM by default.
- For `wmma_persistent`, `m`, `n`, and `k` are nominal reporting parameters;
  actual MAC pressure is controlled by `--blocks-per-sm`,
  `--mma-iters-per-loop`, `--accumulators-per-warp`, `--atomic-period`,
  `--active-sm-fraction`, `--duty-cycle`, and `--period-ms`. `--period-ms`
  controls active/idle switching cadence, not active compute intensity.
- For `cutlass_tile_burn`, `m`, `n`, and `k` are synthetic logical dimensions
  unless `--synthetic-m`, `--synthetic-n`, and `--synthetic-k` are provided.
  They control reported synthetic atom counts, not real matrix coverage.
  `--cutlass-tile-m`, `--cutlass-tile-n`, and `--cutlass-tile-k` describe the
  synthetic atom grouping, not real tile-local GEMM storage. The prototype
  currently emits SM80 BF16 MMA atoms and reports `matrix_shape_is_real=false`,
  `memory_traffic_minimized=true`, `uses_global_ab=false`, and
  `uses_shared_memory_tiles=false`. A later implementation may add a true
  tile-local shared-memory CUTLASS/CuTe burn that loads A/B tiles once per CTA
  and reuses them; the current version is register-constant atom burn.
- Validate actual spatial coverage and Tensor Core utilization with Nsight
  profiler metrics on the H100 server for both engines.
- `wmma_persistent` reports `occupancy_max_active_blocks_per_sm`,
  `effective_blocks_per_sm_estimate`, and `occupancy_limited` because
  `--blocks-per-sm` is requested launch density, not guaranteed resident CTA
  count.
- `--sparsity-mode dense_zero` inserts zero values into selected dense operands
  and still uses dense Tensor Core instructions. It measures operand zero-value
  effects, not true sparse Tensor Core execution. This maps to
  `power_gpu_op_tc_004`.
- `--sparsity-mode structured_2to4` is reserved for real 2:4 sparse Tensor Core
  execution through cuSPARSELt when `cusparseLt.h` and `libcusparseLt` are found
  at build time. If cuSPARSELt is not available, the workload fails clearly
  rather than falling back to dense execution. This maps to `power_gpu_op_tc_005`.
- `power_gpu_op_tc_001` remains the active SM spatial coverage test. It varies
  `--active-sm-fraction`; it must not be implemented by inserting zeros into
  input matrices.
- Summary JSON separates these dimensions with `spatial_coverage_fraction`,
  `sparsity_test_dimension`, `tensor_core_execution_path`,
  `uses_sparse_tensor_core`, and `dense_mma_instruction_count_unchanged`.

The dimensions are related but not interchangeable:

- Active SM coverage measures how power changes as more SMs or CTAs are
  simultaneously active while each active SM remains Tensor Core saturated.
- Dense-zero input sparsity measures operand zero-value/data-pattern effects
  under unchanged dense MMA instruction execution.
- 2:4 structured sparsity measures true sparse Tensor Core execution with a
  valid NVIDIA 2:4 sparse pattern and a sparse GEMM backend.

For `structured_2to4 --sparse-engine cusparselt`, setup creates dense A/B
buffers, prunes/checks A into a valid 2:4 pattern with cuSPARSELt, compresses A,
measures a short dense cuBLAS baseline using the expanded pruned A and same B,
then repeatedly calls `cusparseLtMatmul` during the sparse steady window.
The sparse cuSPARSELt path uses BF16 A/B and BF16 C/D with FP32 compute; the
dense baseline keeps BF16 input and FP32 output. Prune/compression/setup time is
excluded from `measured_runtime_ms`.

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --warmup-sec 30 --steady-sec 60
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 0.5 --warmup-sec 30 --steady-sec 60
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --sparsity-mode dense_zero --zero-ratio 0.5 --zero-pattern regular_k --sparse-operand A --warmup-sec 5 --steady-sec 10
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --sparsity-mode structured_2to4 --sparse-engine cusparselt --warmup-sec 5 --steady-sec 10
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wmma_persistent --m 8192 --n 8192 --k 8192 --duty-cycle 0.5 --active-sm-fraction 1.0 --period-ms 10 --warmup-sec 5 --steady-sec 10
./build/workloads/tensor_core_burn --device 4 --dtype bf16 --engine wmma_persistent --m 16384 --n 16384 --k 16384 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 2 --mma-iters-per-loop 256 --warmup-sec 5 --steady-sec 10
./build/workloads/tensor_core_burn --device 4 --dtype bf16 --engine wmma_persistent --m 16384 --n 16384 --k 16384 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 3 --mma-iters-per-loop 256 --accumulators-per-warp 4 --atomic-period 8192 --warmup-sec 5 --steady-sec 20
./build/workloads/tensor_core_burn --device 4 --dtype bf16 --engine wmma_persistent --m 16384 --n 16384 --k 16384 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 6 --mma-iters-per-loop 256 --accumulators-per-warp 2 --atomic-period 8192 --warmup-sec 5 --steady-sec 20
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 2 --mma-iters-per-loop 256 --cutlass-tile-m 64 --cutlass-tile-n 64 --cutlass-tile-k 32 --sparsity-mode none --warmup-sec 5 --steady-sec 10
```

`cutlass_tile_burn` cap validation examples:

```bash
# Small synthetic request: synthetic_mma_ops_cap_applied should be false.
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 16 --n 8 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 1 --mma-iters-per-loop 256 --cutlass-tile-m 16 --cutlass-tile-n 8 --cutlass-tile-k 16 --sparsity-mode none --warmup-sec 1 --steady-sec 2

# Large synthetic request: synthetic_mma_ops_cap_applied should be true.
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 2 --mma-iters-per-loop 256 --cutlass-tile-m 64 --cutlass-tile-n 64 --cutlass-tile-k 32 --sparsity-mode none --warmup-sec 5 --steady-sec 10
```

## cuda_core_burn

`cuda_core_burn.cu` implements the base CUDA Core cases with custom kernels for
floating-point FMA and integer/logical activity.

```bash
./build/workloads/cuda_core_burn --device 0 --mode fp32_fma --duty-cycle 1.0 --buffer-mb 1024 --warmup-sec 30 --steady-sec 60
./build/workloads/cuda_core_burn --device 0 --mode int32_logic --duty-cycle 1.0 --buffer-mb 1024 --warmup-sec 30 --steady-sec 60
```

## Advanced Memory

`tma_copy.cu`, `l2_hit_sweep.cu`, and `sm_issue_coverage.cu` cover the staged
advanced memory cases. See `docs/advanced_memory_cases.md` for assumptions and
limitations.
