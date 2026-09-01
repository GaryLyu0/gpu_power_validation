# GPU Power Validation

Config-driven GPU power validation framework skeleton for NVIDIA H100/B200.

The initial implementation focuses on case schema validation, command
resolution, telemetry interfaces, dry-run result generation, NVML telemetry
plumbing, CUDA microbenchmarks, clock-control orchestration, and real-workload
integration hooks. Windows laptop runs are for development and dry-run checks;
real CUDA/NVML/Nsight/power validation runs only on the Linux H100/B200 server.

## Windows to H100 Workflow

Develop on Windows, but treat GitHub as a handoff path to the target Linux
server:

1. On Windows, run only non-GPU checks:

```powershell
python -B -m pytest
python -m runner.run_suite --case-dir cases --dry-run
git status --short
```

2. Commit and push to GitHub from the laptop.

3. On the H100/B200 Linux server, clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd gpu_power_validation
git submodule update --init --recursive
```

4. Check the target server environment:

```bash
bash scripts/check_env.sh
```

5. Install Python dependencies and run dry-run validation:

```bash
python3 -m pip install -e ".[gpu,test]"
python3 -B -m pytest
python3 -m runner.run_suite --case-dir cases --dry-run
```

6. Build CUDA workloads on the H100/B200 server:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
```

7. Run smoke tests before full validation:

```bash
bash scripts/h100_smoke_test.sh
```

8. Only after the checks above pass, run real power validation on the H100/B200
server.

## Setup

Windows laptop development setup:

```powershell
python -m pip install -e ".[test]"
```

If `python` points at the Windows app alias, use your full Python path or put
your preferred Python 3.9+ installation earlier in `PATH`.

H100/B200 server setup:

```bash
python3 -m pip install -e ".[gpu,test]"
```

## Dry Run

Run one case:

```powershell
python -m runner.run_case --case cases/io/PWR-MEM-001.yaml --dry-run
```

Run all cases:

```powershell
python -m runner.run_suite --case-dir cases --dry-run
```

Dry-run mode validates the YAML, prints the resolved workload command and
telemetry configuration, creates a mock result directory, and writes:

```text
results/<timestamp>_<case_id>/
  case.yaml
  summary.json
  telemetry.csv
  stdout.log
  stderr.log
```

## NVML Telemetry

Non-dry-run execution starts telemetry before the workload process and stops it
after the workload exits. By default this uses real NVML through `pynvml`, which
must be installed and runnable on the H100/B200 server:

```bash
python -m pip install -e ".[gpu]"
python -m runner.run_case --case cases/idle/PWR-IDLE-001.yaml --mock-workload
```

For laptop-safe plumbing tests, keep both workload and telemetry mocked:

```powershell
python -m runner.run_case --case cases/idle/PWR-IDLE-001.yaml --mock-workload --mock-telemetry
```

If NVML or `pynvml` is unavailable and `--mock-telemetry` is not supplied, the
runner exits with a clear setup error. `summary.json` includes average, p95, and
max power from `telemetry.csv`.

## CUDA IO Microbenchmarks

The first independent CUDA workload covers the IO cases `PWR-MEM-001` and
`PWR-MEM-002`. `PWR-MEM-003` adds the read/write load-level CUDA microbenchmark.
The base Tensor Core and CUDA Core power cases use `tensor_core_burn` and
`cuda_core_burn`. Build them on the H100/B200 server, not on a laptop without
the target CUDA toolchain:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
```

Run the H2D one-way transfer workload:

```bash
./build/workloads/h2d_d2d_copy \
  --device 0 \
  --mode h2d \
  --bytes 268435456 \
  --warmup-sec 10 \
  --steady-sec 60
```

Run the D2D GM-to-GM copy workload:

```bash
./build/workloads/h2d_d2d_copy \
  --device 0 \
  --mode d2d \
  --bytes 268435456 \
  --warmup-sec 10 \
  --steady-sec 60
```

The workload also supports pre-sweep sizing:

```bash
./build/workloads/h2d_d2d_copy \
  --device 0 \
  --mode h2d \
  --bytes 268435456 \
  --warmup-sec 10 \
  --steady-sec 60 \
  --presweep-bytes 4096,65536,1048576,16777216,268435456
```

Each run prints JSON lines to stdout. The runner parses those lines into
`summary.json` as `workload_results` and `workload_summary`.

Run the read/write load-level workload directly:

```bash
./build/workloads/read_write_levels \
  --device 0 \
  --mode read \
  --load-factor 0.125 \
  --buffer-mb 4096 \
  --warmup-sec 10 \
  --steady-sec 60

./build/workloads/read_write_levels \
  --device 0 \
  --mode write \
  --load-factor 1.0 \
  --buffer-mb 4096 \
  --warmup-sec 10 \
  --steady-sec 60
```

Run the base AI Core workloads directly:

```bash
./build/workloads/tensor_core_burn \
  --device 0 \
  --dtype bf16 \
  --engine cublas \
  --m 8192 --n 8192 --k 8192 \
  --duty-cycle 1.0 \
  --active-sm-fraction 1.0 \
  --warmup-sec 30 \
  --steady-sec 60

./build/workloads/cuda_core_burn \
  --device 0 \
  --mode fp32_fma \
  --duty-cycle 1.0 \
  --buffer-mb 1024 \
  --warmup-sec 30 \
  --steady-sec 60

./build/workloads/cuda_core_burn \
  --device 0 \
  --mode int32_logic \
  --duty-cycle 1.0 \
  --buffer-mb 1024 \
  --warmup-sec 30 \
  --steady-sec 60
```

`tensor_core_burn` defaults to `--engine cublas`, where
`--active-sm-fraction` is a cuBLAS SM-count target hint. The optional
`--engine wmma_persistent` path uses persistent CTA coverage and device-side
`clock64` duty-cycle control. In `wmma_persistent` mode, `--period-ms` controls
active/idle switching cadence; `--blocks-per-sm`, `--mma-iters-per-loop`, and
`--accumulators-per-warp` control active compute intensity. For both engines,
validate actual SM activity and Tensor Core utilization with Nsight profiler
metrics on the H100 server.

The experimental `--engine wgmma_persistent` path is different from both of
those engines: it is an H100-only SM90a implementation with one 128-thread
warpgroup per CTA. BF16 A/B tiles are initialized once in shared memory and
reused by asynchronous `64x64x16` WGMMA operations. Phase 2 still supports only
`--duty-cycle 1.0`; it uses no TMA, performs no steady-state global A/B loads,
and derives TFLOPS from the actual completed WGMMA count. The existing
`cutlass_tile_burn` remains an SM80-style CuTe MMA atom burn, while
`wmma_persistent` remains a 32-thread warp-level WMMA path.

Phase 2 retains those workload boundaries and adds compile-time-specialized
two-, three-, and four-accumulator ILP variants with WGMMA wait depths zero
through three. The selected variant reports register count, local-memory bytes,
occupancy, and whether two CTAs per SM are expected to fit. Duration checks use
the device-wide PTX `%globaltimer` nanosecond timebase rather than converting
seconds with an SM clock rate; JSON reports both requested and CUDA-event
duration. The `2/1` accumulator/wait pair remains the Phase-1 baseline.

On H100, CMake must print `Hopper WGMMA support: ON`; the dedicated WGMMA
translation unit is compiled for `sm_90a` while other workloads retain
`CUDA_ARCHITECTURES=90`:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine wgmma_persistent --m 64 --n 64 --k 16 --duty-cycle 1.0 --active-sm-fraction 0.1 --blocks-per-sm 1 --wgmma-ops-per-check 512 --wgmma-wait-group 1 --wgmma-accumulator-sets 2 --warmup-sec 1 --steady-sec 2
cuobjdump --dump-sass build/workloads/tensor_core_burn | grep -E 'HGMMA|WGMMA'
cuobjdump --dump-ptx build/workloads/tensor_core_burn | grep -E 'wgmma\\.mma_async'
```

See [workloads/cuda_microbench/README.md](workloads/cuda_microbench/README.md)
for full-GPU, residency, WMMA comparison, and Nsight Compute commands. WGMMA
runtime behavior and power remain H100-side validation items.

Tensor Core power has separate test dimensions that should not be mixed:

- `power_gpu_op_tc_001` is active SM spatial coverage. It changes
  `--active-sm-fraction` and should keep each active SM Tensor Core saturated.
- `power_gpu_op_tc_004` is dense-zero input sparsity. It inserts zero values into
  dense operands and still uses dense Tensor Core instructions.
- `power_gpu_op_tc_005` is true 2:4 structured sparsity. It requires a sparse
  Tensor Core backend such as cuSPARSELt or CUTLASS SparseGemm.

Dense baseline:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --warmup-sec 30 --steady-sec 60
```

Active SM spatial coverage example:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 0.5 --warmup-sec 30 --steady-sec 60
```

Experimental CUTLASS/CuTe MMA atom burn prototype:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 2 --mma-iters-per-loop 256 --cutlass-tile-m 64 --cutlass-tile-n 64 --cutlass-tile-k 32 --sparsity-mode none --warmup-sec 5 --steady-sec 10
```

`cutlass_tile_burn` is for synthetic Tensor Core burn only. It calls a
CUTLASS/CuTe SM80 BF16 MMA atom directly, does not use top-level
`cutlass::gemm::device::Gemm`, does not load real A/B matrices, and does not use
shared-memory A/B tiles. `cutlass_tile_m/n/k` define synthetic atom grouping, not
real tile-local GEMM storage. Existing YAML cases should stay on the validated
engines until this prototype is profiled on H100/B200.

Cap reporting examples:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 16 --n 8 --k 16 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 1 --mma-iters-per-loop 256 --cutlass-tile-m 16 --cutlass-tile-n 8 --cutlass-tile-k 16 --sparsity-mode none --warmup-sec 1 --steady-sec 2
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cutlass_tile_burn --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --period-ms 500 --blocks-per-sm 2 --mma-iters-per-loop 256 --cutlass-tile-m 64 --cutlass-tile-n 64 --cutlass-tile-k 32 --sparsity-mode none --warmup-sec 5 --steady-sec 10
```

Dense-zero data-pattern example. Summary JSON reports
`uses_sparse_tensor_core=false` and
`dense_mma_instruction_count_unchanged=true`:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --sparsity-mode dense_zero --zero-ratio 0.5 --zero-pattern regular_k --sparse-operand A --warmup-sec 5 --steady-sec 10
```

True 2:4 structured sparse example. Summary JSON for an implemented backend
must report `uses_sparse_tensor_core=true`, `sparse_pattern="2:4"`, and the
selected `sparse_engine`. This path requires `cusparseLt.h` and `libcusparseLt`
at build time; without them, the binary fails clearly instead of falling back to
dense execution:

```bash
./build/workloads/tensor_core_burn --device 0 --dtype bf16 --engine cublas --m 8192 --n 8192 --k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 --sparsity-mode structured_2to4 --sparse-engine cusparselt --warmup-sec 5 --steady-sec 10
```

`nvbandwidth` can be built separately from `third_party/nvbandwidth` and used as
a cross-check, but the primary IO cases do not depend on it and this framework
does not modify `third_party`.

## Case Files

Cases are YAML documents with typed fields for:

- `case_id`, `title`, `category`, `duration_s`, and `gpus`
- workload command resolution
- telemetry sampler configuration
- optional clock and power-limit intent

Supported telemetry samplers are `mock_nvml` and `nvml`. Hardware-dependent
NVML execution is selected at runtime by the runner and guarded by explicit
availability checks.

## Local Validation Scope

Codex and developer laptops are not assumed to have NVIDIA GPUs, CUDA, NVML,
Nsight Systems, Nsight Compute, or the target server driver/toolchain. Local
validation is limited to non-GPU checks:

- YAML schema validation
- dry-run runner tests
- workload command generation tests
- static Python checks when the tool is already available

Hardware-dependent features must stay behind runtime checks and must support
dry-run or mock mode. CUDA/NVML/Nsight validation should be performed later on
the H100/B200 server instead of adapting behavior to a laptop environment.

Run local tests:

```powershell
python -B -m pytest
```

`--mock-workload` is only for runner and telemetry plumbing tests. Real Nsight
Systems and Nsight Compute profiling must run without `--mock-workload`, after
CUDA workloads are built on the H100/B200 host:

```bash
CUDA_ARCHITECTURES=90 bash scripts/build_workloads.sh
python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml --dry-run
nsys profile --stats=true --output results/nsys_core python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml
ncu --set full --target-processes all python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml
```

For the first CUDA IO workload on the H100/B200 server:

```bash
python -m runner.run_case --case cases/io/PWR-MEM-001.yaml
python -m runner.run_case --case cases/io/PWR-MEM-002.yaml
python -m runner.run_case --case cases/mem/PWR-MEM-003.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_001.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_004.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_005.yaml --dry-run
python -m runner.run_case --case cases/core/power_gpu_op_cc_000.yaml
python -m runner.run_case --case cases/core/power_gpu_op_cc_001.yaml
```

Clock sweep cases use `nvidia-smi` to query supported graphics clocks, lock each
planned clock with `-lgc`, and reset with `-rgc` after the run. Dry-run prints
the clock plan without applying it:

```bash
python -m runner.run_case --case cases/core/power_gpu_op_tc_002.yaml --dry-run
python -m runner.run_case --case cases/core/power_gpu_op_tc_002.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_003.yaml
```

Disable clock changes when you need workload-only validation:

```bash
python -m runner.run_case --case cases/core/power_gpu_op_tc_002.yaml --no-clock-control
```

If `nvidia-smi -lgc` fails due to insufficient privileges, the runner prints a
warning. Run on the target server with root/admin privileges or passwordless
`sudo` for clock-locking validation.

Advanced memory cases `PWR-MEM-004` through `PWR-MEM-007` are documented in
[docs/advanced_memory_cases.md](docs/advanced_memory_cases.md). TMA is guarded
and reports fallback status when true architecture-specific TMA is not enabled.

Real workload cases are command-driven and documented in
[docs/real_workload.md](docs/real_workload.md). The default stub supports an
idle mode and a PyTorch CUDA matmul mode for target-server TDP-style plumbing.

## Repository Boundaries

The third-party submodules under `third_party/` are treated as read-only inputs.
Wrappers live under `workloads/third_party_wrappers/` so integration code can
evolve without modifying vendored projects.
