#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gpu_power_validation {

struct WgmmaControlRunOptions {
  int device = 0;
  int requested_sm_count = 1;
  int blocks_per_sm = 1;
  unsigned int sleep_ns = 100000;
  double warmup_sec = 0.0;
  double steady_sec = 1.0;
};

struct WgmmaControlRunResult {
  std::uint64_t wakeup_checks = 0;
  std::string timer_source = "ptx_globaltimer_ns";
  double requested_duration_ms = 0.0;
  double actual_elapsed_ms = 0.0;
  int block_size = 128;
  int grid_blocks = 0;
  int occupancy_max_active_blocks_per_sm = 0;
  int effective_blocks_per_sm_estimate = 0;
  bool occupancy_limited = false;
  bool blocks_per_sm_resource_feasible = false;
  int registers_per_thread = 0;
  std::size_t local_memory_bytes_per_thread = 0;
  std::size_t initial_global_load_bytes = 0;
  std::size_t steady_global_load_bytes_per_loop = 0;
  std::size_t final_global_store_bytes = 0;
};

WgmmaControlRunResult run_wgmma_control_sm90a(
    const WgmmaControlRunOptions& options);

}  // namespace gpu_power_validation
