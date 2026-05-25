# Agent Instructions

This repository is developed on a Windows laptop and handed off through GitHub
to a Linux NVIDIA H100/B200 validation server. Treat the laptop as a development
environment only, not as the target runtime.

- Do not assume local Windows execution validates CUDA, NVML, Nsight, clock
  control, or power behavior.
- Do not run H100-only tests locally. Use local checks only for YAML validation,
  dry-run runner tests, command generation tests, and available static Python
  checks.
- All hardware-specific values must remain configurable through case YAML,
  config files, or CLI arguments. Do not hard-code GPU index, SM count, clocks,
  warmup/steady durations, telemetry interval, output paths, or workload length
  in framework logic.
- All runners and hardware-dependent paths must preserve dry-run or mock mode.
  Dry-run may validate command construction and output layout, but must not
  launch heavy CUDA workloads or modify GPU state.
- Treat `third_party/` entries as git submodules. Do not edit submodule contents
  directly from this repository.
- Do not commit generated binaries, build directories, raw results, logs,
  telemetry dumps, Nsight Systems reports, Nsight Compute reports, or other
  profiling outputs.
- Real validation happens only after cloning/pulling this repo on the H100/B200
  Linux server, initializing submodules, checking the environment, building
  workloads there, and then running smoke/full validation.

