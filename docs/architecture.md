# Architecture

The runner is intentionally config-driven:

- Case YAML files describe GPU selection, workload command resolution,
  telemetry, and optional clock or power-limit intent.
- `runner.case_schema` validates the YAML into typed dataclasses.
- `runner.workload_launcher` resolves commands but does not execute workloads in
  dry-run mode.
- `runner.telemetry` defines the sampler interface.
- `runner.nvml_sampler` provides mock telemetry for local checks and real NVML
  telemetry behind runtime checks on the H100/B200 server.
- `runner.result_writer` owns the result directory layout.

Additional workload coverage and third-party benchmark integration are future
implementation steps.

## Hardware Boundary

The compatibility target is the H100/B200 server, not a developer laptop. Code
that touches CUDA, NVML, Nsight tools, clock controls, driver state, or GPU
hardware must be isolated behind runtime checks and explicit interfaces. The
mock sampler and dry-run runner exist so case validation and result formatting
can be tested without weakening the target-server path.

Local validation should cover schema parsing, command generation, dry-run result
writing, and available static Python checks. Hardware validation belongs on the
H100/B200 system once real implementations are added.
