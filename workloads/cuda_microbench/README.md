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

`tensor_core_burn` implements the base Tensor Core cases with several independent
engines:

- `--engine cublas` keeps the original cuBLAS BF16 GEMM behavior and remains
  the default for backward compatibility.
- `--engine wmma_persistent` launches persistent CTAs and uses device-side
  `clock64` control so short active/idle periods do not depend on CPU-side GEMM
  launch timing.
- `--engine cutlass_tile_burn` is an experimental CUTLASS/CuTe MMA atom based
  synthetic Tensor Core burn. It calls the SM80 BF16 MMA atom directly, does not
  use top-level `cutlass::gemm::device::Gemm`, does not load real A/B matrices,
  and does not use shared-memory A/B tiles.
- `--engine wgmma_persistent` is an H100-only SM90a backend. One 128-thread CTA
  forms one four-warp warpgroup and repeatedly issues asynchronous
  `64x{64,128,256}x16` BF16 x BF16 -> FP32 WGMMA operations selected by
  `--wgmma-instruction-n` from shared-memory-resident A/B tiles.
- `--engine wgmma_control` is an H100-only persistent resident-control
  baseline. It launches the same requested 128-thread CTA topology as
  `wgmma_persistent`, but spends nearly all steady-state time in
  `__nanosleep()` and executes no tensor instructions or operand streams.

The WGMMA source is compiled separately for `sm_90a`; the existing targets keep
their configured architecture list. CMake prints `Hopper WGMMA support: ON`
when the vendored CuTe headers and an SM90a-capable CUDA compiler are available.
If support is unavailable, the other engines still build and requesting
`wgmma_persistent` fails without falling back to another execution path.

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
- For `wgmma_persistent`, `m`, `n`, and `k` are retained for CLI compatibility
  but do not determine executed work. TFLOPS are calculated from the completed
  WGMMA count and selected `64x{64,128,256}x16` instruction shape. Phase 2
  remains BF16-only and accepts only
  `--duty-cycle 1.0`.
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

### Hopper WGMMA phase 2

`wgmma_persistent` is a synthetic Tensor Core power workload, not a GEMM shape
benchmark. At CTA startup, 128 threads initialize deterministic BF16 A and B
tiles directly in shared memory. The persistent region reuses those tiles for
SM90a WGMMA and performs no TMA transfers, no global A/B loads, no global
atomics, and no global stores. After draining all outstanding WGMMA groups, one
counter and one accumulator sample per CTA are written to global memory.

Phase 2 supports compile-time-specialized instruction N values 64, 128, and 256,
alongside one through four compile-time-specialized accumulator sets and wait
depths zero through three, with `wait_group < accumulator_sets`. Each
specialization has independent named FP32 fragments; there is no runtime-indexed
accumulator array. The `2/1` configuration is the Phase-1 baseline. The primary
new comparisons are `3/2` and `4/3`, plus shallower waits for diagnosing
register pressure versus asynchronous WGMMA concurrency.

`--wgmma-ops-per-check` controls how many operations execute between coarse,
warpgroup-uniform termination checks. These checks read PTX `%globaltimer`, a
device-wide nanosecond timebase that does not scale with SM DVFS. JSON reports
`timer_source=ptx_globaltimer_ns`, `requested_duration_ms`, and the measured
CUDA-event `actual_elapsed_ms`.

The selected kernel specialization also reports `registers_per_thread`,
`local_memory_bytes_per_thread`, `occupancy_max_active_blocks_per_sm`,
`effective_blocks_per_sm_estimate`, and
`allows_at_least_two_resident_ctas_per_sm`. A warning is emitted when resource
limits prevent two resident CTAs per SM or CUDA reports local memory. CMake
enables ptxas verbose output for the SM90a translation unit so spill loads and
stores can be checked at build time. `--active-sm-fraction` and
`--blocks-per-sm` still control grid size, but CUDA scheduling only approximates
SM coverage and does not select specific SM IDs.

JSON also reports `wgmma_instruction_m/n/k`, `wgmma_flops_per_op`, and
`wgmma_smem_operand_bytes_per_op`. For instruction N of 64, 128, and 256,
FLOPs per operation are `2 * 64 * N * 16`, while logical shared-memory operand
bytes per operation are `(64 * 16 + N * 16) * sizeof(bf16)`.

Functional smoke on H100:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wgmma_persistent --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 0.1 --blocks-per-sm 1 --wgmma-instruction-n 64 --wgmma-ops-per-check 512 --wgmma-wait-group 1 --wgmma-accumulator-sets 2 --sparsity-mode none --warmup-sec 1 --steady-sec 2
```

Full-GPU and residency comparison:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wgmma_persistent --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 1 --wgmma-ops-per-check 512 --wgmma-wait-group 1 --wgmma-accumulator-sets 2 --sparsity-mode none --warmup-sec 5 --steady-sec 20
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wgmma_persistent --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 2 --wgmma-ops-per-check 512 --wgmma-wait-group 1 --wgmma-accumulator-sets 2 --sparsity-mode none --warmup-sec 5 --steady-sec 20
```

Build with resource diagnostics and run the Phase-2 ILP matrix. Do not infer a
performance improvement until these variants are measured on H100:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh 2>&1 | tee build/wgmma_phase2_build.log
grep -Ei 'Used [0-9]+ registers|spill|local' build/wgmma_phase2_build.log

for pair in "2 1" "3 1" "3 2" "4 1" "4 2" "4 3"; do
  set -- ${pair}
  echo "=== accumulators=$1 wait_group=$2 ==="
  ./build/workloads/tensor_core_burn \
    --device 0 --dtype bf16 --engine wgmma_persistent \
    --m 64 --n 64 --k 16 \
    --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 2 \
    --wgmma-ops-per-check 2048 \
    --wgmma-accumulator-sets "$1" --wgmma-wait-group "$2" \
    --sparsity-mode none --warmup-sec 5 --steady-sec 10
done
```

Instruction-shape exploration keeps the Phase-2 synchronization model fixed.
The primary full-duty configurations are:

```bash
for config in "64 2 1" "128 1 0" "128 2 1" "256 1 0"; do
  set -- ${config}
  echo "=== instruction_n=$1 accumulators=$2 wait_group=$3 ==="
  ./build/workloads/tensor_core_burn \
    --device 0 --dtype bf16 --engine wgmma_persistent \
    --m 64 --n "$1" --k 16 \
    --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 2 \
    --wgmma-instruction-n "$1" --wgmma-ops-per-check 2048 \
    --wgmma-accumulator-sets "$2" --wgmma-wait-group "$3" \
    --sparsity-mode none --warmup-sec 5 --steady-sec 10
done
```

For each result, require real HGMMA, inspect ptxas for zero spill loads/stores,
check `local_memory_bytes_per_thread`, and confirm
`occupancy_max_active_blocks_per_sm >= 2` before comparing TFLOPS or power.

Verify the generated code after the H100 build. The CMake target embeds both
SM90a machine code and compute_90a PTX in the executable:

```bash
cuobjdump --dump-sass build/workloads/tensor_core_burn | grep -E 'HGMMA\.64x(64|128|256)x16\.F32\.BF16|WGMMA'
cuobjdump --dump-ptx build/workloads/tensor_core_burn | grep -E 'wgmma\\.mma_async'
```

Do not infer WGMMA execution from C++ type names alone. The first command should
show `HGMMA.64x64x16.F32.BF16`, `HGMMA.64x128x16.F32.BF16`, and
`HGMMA.64x256x16.F32.BF16` instructions and the second should show
`wgmma.mma_async` PTX. Absence of both is a failed WGMMA build validation.

Nsight Compute metric names vary by installed version. Discover available
metrics first, then profile comparable full-duty runs:

```bash
ncu --query-metrics | grep -Ei 'tensor|mma|wgmma|hmma|dram|l2|shared|occupancy'
ncu --set full --target-processes all -o results/ncu_wgmma ./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wgmma_persistent --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 1 --warmup-sec 1 --steady-sec 5
ncu --set full --target-processes all -o results/ncu_wmma ./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wmma_persistent --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 1 --mma-iters-per-loop 512 --warmup-sec 1 --steady-sec 5
ncu --set full --target-processes all -o results/ncu_cutlass_atom ./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 1 --synthetic-mma-ops-per-loop 512 --warmup-sec 1 --steady-sec 5
```

Compare Tensor Core/MMA utilization, achieved TFLOPS, SM utilization, DRAM and
L2 traffic, global load/store traffic, shared-memory activity, register use,
and occupancy. The intended result is higher sustained Hopper Tensor Core
utilization without meaningful HBM or TMA traffic; that claim requires H100
profiler and board-power measurements.

### Hopper persistent-control baseline

`wgmma_control` measures the power floor of the requested persistent CTA/warp
topology without WGMMA or sustained memory work. It uses 128 threads per CTA,
the same `requested_sm_count * blocks_per_sm` grid calculation as
`wgmma_persistent`, `%globaltimer` for DVFS-independent requested duration, and
`__nanosleep()` between thread-0 timer checks. `--control-sleep-ns` defaults to
100000 and must be in `[1, 1000000]`; the timer, not the nominal sleep interval,
determines termination.

The control kernel does not allocate WGMMA A/B shared-memory tiles and does not
attempt to reproduce WGMMA register pressure. It reports its actual register
count, local-memory bytes, occupancy limit, and whether the requested
`blocks_per_sm` is resource-feasible without silently clamping the grid. Its
only global store is one wakeup/check count per CTA at exit.

Use this baseline for attribution as follows:

- `P_control - P_idle` approximates persistent CTA, warp, scheduler, timer,
  sleep, and synchronization overhead.
- `P_wgmma_full - P_control` approximates incremental Tensor-compute-complex
  power, including required WGMMA register-file, shared-memory, and control
  activity. It is not pure Tensor Core transistor power.

Primary control and frozen full-compute reference:

```bash
./build/workloads/tensor_core_burn \
  --device 1 --dtype bf16 --engine wgmma_control \
  --active-sm-fraction 1.0 --blocks-per-sm 2 \
  --control-sleep-ns 100000 \
  --warmup-sec 3 --steady-sec 10

./build/workloads/tensor_core_burn \
  --device 1 --dtype bf16 --engine wgmma_persistent \
  --m 64 --n 128 --k 16 \
  --duty-cycle 1.0 --active-sm-fraction 1.0 --blocks-per-sm 2 \
  --wgmma-instruction-n 128 --wgmma-ops-per-check 2048 \
  --wgmma-accumulator-sets 2 --wgmma-wait-group 1 \
  --warmup-sec 3 --steady-sec 10
```

Isolate and reject tensor instructions in the control kernel SASS, then verify
the WGMMA reference separately:

```bash
cuobjdump --dump-sass --function wgmma_control_kernel_sm90a \
  build/workloads/tensor_core_burn | tee build/wgmma_control.sass
if grep -E 'HGMMA|WGMMA|HMMA|MMA' build/wgmma_control.sass; then
  echo "ERROR: tensor instruction found in wgmma_control" >&2
  exit 1
fi

cuobjdump --dump-sass build/workloads/tensor_core_burn |
  grep -E 'HGMMA\.64x128x16\.F32\.BF16|WGMMA'
```

On the 132-SM H100, the primary control configuration should launch 264 CTAs.
Validate approximately eight achieved active warps per SM, zero Tensor
instructions, no spills, and negligible DRAM/L2 throughput with Nsight Compute
before using its measured board power.

```bash
ncu --set full --target-processes all -o results/ncu_wgmma_control \
  ./build/workloads/tensor_core_burn \
  --device 1 --dtype bf16 --engine wgmma_control \
  --active-sm-fraction 1.0 --blocks-per-sm 2 \
  --control-sleep-ns 100000 --warmup-sec 1 --steady-sec 5
```

### Hopper WGMMA temporal duty validation

`wgmma_persistent` supports an in-kernel temporal duty schedule for the frozen
H100 `m64n128k16` BF16 SS WGMMA configuration with two accumulator sets and
`wait_group=1`. The active phase uses the same WGMMA issue primitive as the
validated full-compute workload. Before every idle phase it drains all pending
WGMMA groups, then sleeps in bounded `__nanosleep()` chunks. `%globaltimer`
modulo the requested period provides a common device-wide phase without a
global barrier or hot-loop atomics.

`--wgmma-duty-period-ns` defaults to 1 ms and
`--wgmma-duty-check-ops` defaults to 64. The latter controls active-phase timer
granularity only. At `--duty-cycle 1.0`, the original full-duty kernel remains
the selected fast path and continues to use `--wgmma-ops-per-check` (2048 in
the validated setup); it performs no duty-control nanosleep or fine timer
checks. At duty zero, the duty-controlled WGMMA kernel and its shared-memory
and accumulator resource shape remain resident, but it issues zero WGMMA ops.

Run the initial fixed-shape matrix on H100 after externally locking the selected
GPU's SM clock to 1770 MHz. The benchmark itself does not call `nvidia-smi`:

```bash
GPU=0
for duty in 0.0 0.1 0.5 0.9 1.0; do
  ./build/workloads/tensor_core_burn \
    --device "${GPU}" --dtype bf16 --engine wgmma_persistent \
    --m 64 --n 128 --k 16 \
    --duty-cycle "${duty}" --active-sm-fraction 1.0 --blocks-per-sm 2 \
    --wgmma-instruction-n 128 --wgmma-accumulator-sets 2 \
    --wgmma-wait-group 1 --wgmma-ops-per-check 2048 \
    --wgmma-duty-period-ns 1000000 --wgmma-duty-check-ops 64 \
    --sparsity-mode none --warmup-sec 3 --steady-sec 10
done
```

The JSON distinguishes requested and measured active/idle time and reports
`wall_tflops` separately from `active_window_tflops`. The legacy
`active_tflops` field retains its prior wall-time definition. Measured active
and idle nanoseconds are averages of the final per-CTA counters. A single run does
not have a same-condition full-duty reference, so
`normalized_mac_utilization` is `null`; compute it afterward as measured
`wall_tflops / full-duty wall_tflops` from the same fixed-clock experiment.

After building for H100, confirm the active path remains
`HGMMA.64x128x16.F32.BF16`, the controlled specialization contains
`NANOSLEEP`, no TMA instructions appear, occupancy still permits two CTAs/SM,
and `local_memory_bytes_per_thread=0`. Treat
`resource_validation_passed=false` as a failed resource check:

```bash
cuobjdump --dump-sass build/workloads/tensor_core_burn | \
  grep -E 'HGMMA\.64x128x16\.F32\.BF16|NANOSLEEP|TMA'

ncu --set full --target-processes all -o results/ncu_wgmma_duty50 \
  ./build/workloads/tensor_core_burn \
  --device 0 --dtype bf16 --engine wgmma_persistent \
  --m 64 --n 128 --k 16 --duty-cycle 0.5 \
  --active-sm-fraction 1.0 --blocks-per-sm 2 \
  --wgmma-instruction-n 128 --wgmma-accumulator-sets 2 \
  --wgmma-wait-group 1 --wgmma-ops-per-check 2048 \
  --wgmma-duty-period-ns 1000000 --wgmma-duty-check-ops 64 \
  --warmup-sec 1 --steady-sec 5
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
