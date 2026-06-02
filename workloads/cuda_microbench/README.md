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
- Validate actual spatial coverage and Tensor Core utilization with Nsight
  profiler metrics on the H100 server for both engines.
- `wmma_persistent` reports `occupancy_max_active_blocks_per_sm`,
  `effective_blocks_per_sm_estimate`, and `occupancy_limited` because
  `--blocks-per-sm` is requested launch density, not guaranteed resident CTA
  count.

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --warmup-sec 30 --steady-sec 60
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wmma_persistent --m 8192 --n 8192 --k 8192 --duty-cycle 0.5 --active-sm-fraction 1.0 --period-ms 10 --warmup-sec 5 --steady-sec 10
./build/workloads/tensor_core_burn --device 4 --dtype bf16 --engine wmma_persistent --m 16384 --n 16384 --k 16384 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 2 --mma-iters-per-loop 256 --warmup-sec 5 --steady-sec 10
./build/workloads/tensor_core_burn --device 4 --dtype bf16 --engine wmma_persistent --m 16384 --n 16384 --k 16384 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 3 --mma-iters-per-loop 256 --accumulators-per-warp 4 --atomic-period 8192 --warmup-sec 5 --steady-sec 20
./build/workloads/tensor_core_burn --device 4 --dtype bf16 --engine wmma_persistent --m 16384 --n 16384 --k 16384 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 6 --mma-iters-per-loop 256 --accumulators-per-warp 2 --atomic-period 8192 --warmup-sec 5 --steady-sec 20
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
