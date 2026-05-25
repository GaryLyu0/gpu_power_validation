# Real Workload Integration

The real workload cases are intentionally command-driven. The runner does not
need to know whether the workload is a torch script, vLLM benchmark, cuDNN
frontend benchmark, container wrapper, or custom shell command. Replace the YAML
`workload` block to point at the command you want to profile.

## Default Stub

`workloads/python/real_workload_stub.py` supports:

- `--mode idle`: sleeps through warmup and steady windows without GPU work.
- `--mode tdp_matmul`: runs a PyTorch CUDA matrix-multiply loop when PyTorch CUDA
  is available on the H100/B200 server.

Examples:

```bash
python workloads/python/real_workload_stub.py --mode idle --warmup-sec 30 --steady-sec 180

python workloads/python/real_workload_stub.py \
  --mode tdp_matmul \
  --device 0 \
  --dtype bf16 \
  --matrix-size 8192 \
  --warmup-sec 30 \
  --steady-sec 300
```

The `tdp_matmul` mode exits with a machine-readable JSON error if PyTorch or
PyTorch CUDA is unavailable. This is deliberate: do not adapt target-server
behavior to a laptop environment.

## Replacing The Stub

Edit only the case YAML `workload` section. Keep telemetry, result writing, and
summary parsing unchanged.

### Torch Model Script

```yaml
workload:
  type: command
  executable: python
  args:
    - workloads/python/my_model_power.py
    - --model
    - /models/example
    - --batch-size
    - "16"
    - --warmup-sec
    - "30"
    - --steady-sec
    - "300"
```

### vLLM Benchmark

```yaml
workload:
  type: command
  executable: python
  args:
    - -m
    - vllm.benchmarks.benchmark_throughput
    - --model
    - /models/example
    - --num-prompts
    - "1024"
```

### cuDNN Frontend SDPA Benchmark

```yaml
workload:
  type: command
  executable: ./build/workloads/run_cudnn_sdpa
  args:
    - --batch
    - "16"
    - --heads
    - "128"
    - --seq-len
    - "4096"
```

### Custom Shell Command

Prefer a small checked-in wrapper script for complex shell pipelines:

```yaml
workload:
  type: command
  executable: bash
  args:
    - scripts/run_model_power.sh
```

## Summary Integration

Any workload can print JSON lines to stdout. The runner copies those records
into `summary.json` under `workload_results`; the final steady record is exposed
as `workload_summary`.

## Target-Server Commands

```bash
python -m runner.run_case --case cases/real_workload/power_idle_000.yaml
python -m runner.run_case --case cases/real_workload/power_tdp_000.yaml
```

Laptop-safe validation:

```bash
python -m runner.run_case --case cases/real_workload/power_idle_000.yaml --dry-run
python -m runner.run_case --case cases/real_workload/power_tdp_000.yaml --dry-run
```
