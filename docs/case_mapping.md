# Case Mapping

This suite maps the original power validation intentions into config-driven YAML
cases. The IO H2D/D2D cases now use the first independent CUDA microbenchmark;
the remaining workload executables stay as placeholders until CUDA/NVML/Nsight
implementations are built and validated on the H100/B200 server.

| Case | Category | Framework mapping |
| --- | --- | --- |
| `PWR-IDLE-001` | idle | Driver-initialized SOC idle baseline with no GPU workload. |
| `PWR-IDLE-002` | idle | Idle baseline with clock-lock intent recorded for target-server validation. |
| `PWR-IDLE-003` | idle | Longer idle soak for power drift and stability tracking. |
| `PWR-MEM-001` | io | H2D one-way IO saturation test using `./build/workloads/h2d_d2d_copy --mode h2d` with a 4KB, 64KB, 1MB, 16MB, 256MB block-size pre-sweep and selected steady transfer size. |
| `PWR-MEM-002` | io | D2D global-memory copy bandwidth-power test using `./build/workloads/h2d_d2d_copy --mode d2d` with the same block-size pre-sweep. |
| `PWR-MEM-003` | mem | Read/write memory power sweep using `./build/workloads/read_write_levels` across read/write modes and L1-L8 load factors from 1/8 through 8/8. |
| `PWR-MEM-004` | mem | TMA copy direction test using `./build/workloads/tma_copy` for GM-to-SMEM and SMEM-to-GM, with fallback status reported. |
| `PWR-MEM-005` | mem | TMA versus normal copy mechanism comparison using `./build/workloads/tma_copy` under matched transfer shape. |
| `PWR-MEM-006` | mem | L2 locality sweep using `./build/workloads/l2_hit_sweep` with low, medium, and high target hit-rate bands. |
| `PWR-MEM-007` | mem | Half-SM versus full-SM memory issue coverage comparison using `./build/workloads/sm_issue_coverage`. |
| `power_gpu_op_tc_000` | core | Tensor Core GEMM power versus time duty cycle using `./build/workloads/tensor_core_burn`, 0% to 100% in 10% steps. |
| `power_gpu_op_tc_001` | core | Tensor Core GEMM Power @ SM Spatial Coverage using `./build/workloads/tensor_core_burn`, 10% to 100% active SM/CTA coverage in 10% steps. This is not an input-sparsity test. |
| `power_gpu_op_tc_002` | core | Tensor Core GEMM time duty-cycle sweep under `nvidia-smi` locked GPU clocks selected from `auto_low`, `auto_mid`, and `auto_high`. |
| `power_gpu_op_tc_003` | core | Tensor Core GEMM spatial coverage sweep under `nvidia-smi` locked GPU clocks selected from `auto_low`, `auto_mid`, and `auto_high`. |
| `power_gpu_op_tc_004` | core | Tensor Core GEMM Power @ Dense-Zero Input Sparsity using the dense Tensor Core path with `--sparsity-mode dense_zero`, zero-ratio sweep, and `uses_sparse_tensor_core=false`. |
| `power_gpu_op_tc_005` | core | Tensor Core Sparse GEMM Power @ 2:4 Structured Sparsity using `--sparsity-mode structured_2to4 --sparse-engine cusparselt` when cuSPARSELt is available at build/link time. This is the true sparse Tensor Core execution case; CUTLASS SparseGemm remains future work. |
| `power_gpu_op_cc_000` | core | CUDA Core floating-point elementwise power versus time duty cycle using `./build/workloads/cuda_core_burn --mode fp32_fma`. |
| `power_gpu_op_cc_001` | core | CUDA Core integer/logical elementwise power versus time duty cycle using `./build/workloads/cuda_core_burn --mode int32_logic`. |
| `power_tdp_000` | real_workload | Command-driven real workload TDP test using `workloads/python/real_workload_stub.py --mode tdp_matmul` by default. |
| `power_idle_000` | real_workload | Command-driven initialized-idle test using `workloads/python/real_workload_stub.py --mode idle` by default. |

## Runtime Boundary

Local dry-run validation checks schema parsing, command generation, and result
directory creation only. Hardware-dependent CUDA kernels, NVML sampling, clock
control, Nsight Systems, and Nsight Compute checks must run on the target
H100/B200 server behind explicit runtime checks.

## Tensor Core Taxonomy

Active SM coverage, dense-zero input sparsity, and true 2:4 structured sparsity
are related power dimensions, but they are not interchangeable:

- `power_gpu_op_tc_001` is a spatial coverage test. It changes how many SMs or
  CTAs are simultaneously active while each active SM should remain Tensor Core
  saturated.
- `power_gpu_op_tc_004` is a dense data-pattern test. It inserts zero values into
  dense operands and still reports `uses_sparse_tensor_core=false`.
- `power_gpu_op_tc_005` is the true sparse execution test. It requires a valid
  NVIDIA 2:4 pattern and cuSPARSELt build/link support, and must report
  `uses_sparse_tensor_core=true` only when `cusparseLtMatmul` executes
  successfully.

When real implementations are added, run target-server validation with commands
such as:

```bash
bash scripts/build_workloads.sh
python -m runner.run_case --case cases/io/PWR-MEM-001.yaml
python -m runner.run_case --case cases/io/PWR-MEM-002.yaml
python -m runner.run_case --case cases/mem/PWR-MEM-003.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_001.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_002.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_003.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_004.yaml
python -m runner.run_case --case cases/core/power_gpu_op_tc_005.yaml --dry-run
python -m runner.run_case --case cases/core/power_gpu_op_cc_000.yaml
python -m runner.run_case --case cases/core/power_gpu_op_cc_001.yaml
python -m runner.run_suite --case-dir cases
nsys profile --stats=true --output results/nsys_suite python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml
ncu --set full --target-processes all python -m runner.run_case --case cases/core/power_gpu_op_tc_000.yaml
```
