#!/usr/bin/env bash
set -u

status=0

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

fail() {
  printf '[FAIL] %s\n' "$*" >&2
  status=1
}

require_cmd() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    info "${name}: $(command -v "${name}")"
  else
    fail "${name} not found in PATH"
  fi
}

optional_cmd() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    info "${name}: $(command -v "${name}")"
  else
    warn "${name} not found in PATH; related profiling checks are optional"
  fi
}

info "GPU power validation environment check"
info "Host: $(hostname)"
info "Kernel: $(uname -srmo 2>/dev/null || uname -a)"

require_cmd nvidia-smi
require_cmd nvcc
require_cmd python3
optional_cmd nsys
optional_cmd ncu

if command -v python3 >/dev/null 2>&1; then
  python3 --version || fail "python3 exists but did not run"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || fail "nvcc exists but did not run"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/tmp/gpu_power_validation_nvidia_smi_l.out 2>/tmp/gpu_power_validation_nvidia_smi_l.err; then
    info "Visible GPUs:"
    sed 's/^/  /' /tmp/gpu_power_validation_nvidia_smi_l.out
  else
    fail "nvidia-smi -L failed"
    sed 's/^/  /' /tmp/gpu_power_validation_nvidia_smi_l.err >&2
  fi

  if ! nvidia-smi >/tmp/gpu_power_validation_nvidia_smi.out 2>/tmp/gpu_power_validation_nvidia_smi.err; then
    warn "nvidia-smi summary failed; driver/NVML may not be ready"
    sed 's/^/  /' /tmp/gpu_power_validation_nvidia_smi.err >&2
  fi
fi

rm -f /tmp/gpu_power_validation_nvidia_smi_l.out \
      /tmp/gpu_power_validation_nvidia_smi_l.err \
      /tmp/gpu_power_validation_nvidia_smi.out \
      /tmp/gpu_power_validation_nvidia_smi.err

if [ "${status}" -eq 0 ]; then
  info "Required H100/B200 environment checks passed"
else
  fail "One or more required environment checks failed"
fi

exit "${status}"
