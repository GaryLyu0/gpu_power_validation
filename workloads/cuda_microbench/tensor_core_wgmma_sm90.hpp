#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gpu_power_validation {

struct WgmmaRunOptions {
  int device = 0;
  int requested_sm_count = 1;
  int blocks_per_sm = 1;
  int ops_per_check = 512;
  int wait_group = 1;
  int accumulator_sets = 2;
  double warmup_sec = 0.0;
  double steady_sec = 1.0;
};

struct WgmmaRunResult {
  std::uint64_t wgmma_ops_executed = 0;
  std::string timer_source = "ptx_globaltimer_ns";
  double requested_duration_ms = 0.0;
  double actual_elapsed_ms = 0.0;
  int grid_blocks = 0;
  int occupancy_max_active_blocks_per_sm = 0;
  int effective_blocks_per_sm_estimate = 0;
  bool occupancy_limited = false;
  int registers_per_thread = 0;
  std::size_t local_memory_bytes_per_thread = 0;
  bool allows_at_least_two_resident_ctas_per_sm = false;
  bool correctness_smoke_passed = false;
  double correctness_reference = 16.0;
  double correctness_observed = 0.0;
  double correctness_abs_error = 0.0;
  std::size_t initial_global_load_bytes = 0;
  std::size_t steady_global_load_bytes_per_loop = 0;
  std::size_t final_global_store_bytes = 0;
};

WgmmaRunResult run_wgmma_persistent_sm90a(const WgmmaRunOptions& options);

}  // namespace gpu_power_validation
