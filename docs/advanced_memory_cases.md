# Advanced Memory Cases

These cases are intended for H100/B200 target-server validation. They are not
validated on developer laptops without CUDA, NVML, and the target driver stack.

## PWR-MEM-004 and PWR-MEM-005: TMA Copy

`workloads/cuda_microbench/tma_copy.cu` provides a guarded TMA entrypoint:

- `--direction gm_to_smem`
- `--direction smem_to_gm`
- `--mechanism normal`
- `--mechanism tma`

True TMA programming is architecture and toolchain specific. The initial
implementation always builds a normal shared-memory staged copy fallback. When
`--mechanism tma` is requested, the workload reports:

- `tma_supported`
- `fallback_used`

Only treat the run as a true TMA measurement when `tma_supported` is `true` and
`fallback_used` is `false`. Until an architecture-specific TMA path is enabled,
use these cases to validate runner plumbing and normal-copy comparison shape.

The build option reserved for future architecture-specific TMA code is:

```bash
cmake -S workloads/cuda_microbench -B build/workloads -DGPU_POWER_ENABLE_EXPERIMENTAL_TMA=ON
```

At this stage, enabling the option only changes reported capability gates; it
does not add a real TMA instruction sequence.

## PWR-MEM-006: L2 Hit-Rate Sweep

`workloads/cuda_microbench/l2_hit_sweep.cu` supports:

- `--target low`
- `--target medium`
- `--target high`

The target controls working-set size to bias locality. It does not guarantee a
specific hardware L2 hit rate. Validate actual hit-rate bands with Nsight
Compute counters on the H100/B200 server.

## PWR-MEM-007: SM Issue Coverage

`workloads/cuda_microbench/sm_issue_coverage.cu` supports:

- `--active-sm-fraction 0.5`
- `--active-sm-fraction 1.0`

The workload approximates half/full coverage by changing CTA count relative to
the detected SM count. Actual SM residency and scheduling coverage should be
validated with profiler counters.

## Target-Server Commands

```bash
bash scripts/build_workloads.sh

python -m runner.run_case --case cases/mem/PWR-MEM-004.yaml
python -m runner.run_case --case cases/mem/PWR-MEM-005.yaml
python -m runner.run_case --case cases/mem/PWR-MEM-006.yaml
python -m runner.run_case --case cases/mem/PWR-MEM-007.yaml
```

Optional profiler checks:

```bash
ncu --set full --target-processes all python -m runner.run_case --case cases/mem/PWR-MEM-006.yaml
ncu --set full --target-processes all python -m runner.run_case --case cases/mem/PWR-MEM-007.yaml
```
