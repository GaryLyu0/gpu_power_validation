#include "tensor_core_wgmma_control_sm90.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gpu_power_validation {
namespace {

constexpr int kControlThreads = 128;
constexpr unsigned int kMaxControlSleepNs = 1000000;

struct ControlSharedStorage {
  unsigned long long start_time_ns;
  int continue_running;
};

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << call << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

__device__ __forceinline__ unsigned long long read_globaltimer_ns() {
  unsigned long long timestamp_ns = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp_ns));
  return timestamp_ns;
}

}  // namespace

extern "C" __global__ __launch_bounds__(kControlThreads)
void wgmma_control_kernel_sm90a(
    unsigned long long duration_ns,
    unsigned int sleep_ns,
    unsigned long long* cta_wakeup_checks) {
  __shared__ ControlSharedStorage storage;

  if (threadIdx.x == 0) {
    storage.start_time_ns = read_globaltimer_ns();
    storage.continue_running = 1;
  }
  __syncthreads();

  unsigned long long wakeup_checks = 0;
  while (storage.continue_running != 0) {
    __nanosleep(sleep_ns);
    if (threadIdx.x == 0) {
      ++wakeup_checks;
      storage.continue_running =
          (read_globaltimer_ns() - storage.start_time_ns) < duration_ns ? 1 : 0;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    cta_wakeup_checks[blockIdx.x] = wakeup_checks;
  }
}

namespace {

unsigned long long duration_to_nanoseconds(double seconds) {
  const double duration_ns = seconds * 1.0e9;
  if (duration_ns <= 0.0) {
    throw std::runtime_error("WGMMA control duration must be positive");
  }
  return static_cast<unsigned long long>(duration_ns);
}

void launch_control(
    int grid_blocks,
    unsigned long long duration_ns,
    unsigned int sleep_ns,
    unsigned long long* cta_wakeup_checks) {
  wgmma_control_kernel_sm90a<<<grid_blocks, kControlThreads>>>(
      duration_ns, sleep_ns, cta_wakeup_checks);
  check_cuda(cudaGetLastError(), "wgmma_control_kernel_sm90a");
}

}  // namespace

WgmmaControlRunResult run_wgmma_control_sm90a(
    const WgmmaControlRunOptions& options) {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, options.device),
             "cudaGetDeviceProperties(wgmma_control)");
  if (properties.major != 9 || properties.minor != 0) {
    throw std::runtime_error(
        "wgmma_control requires NVIDIA Hopper compute capability 9.0 and an sm_90a build");
  }
  if (options.sleep_ns == 0 || options.sleep_ns > kMaxControlSleepNs) {
    throw std::runtime_error("--control-sleep-ns must be in [1, 1000000]");
  }

  const int grid_blocks = options.requested_sm_count * options.blocks_per_sm;
  WgmmaControlRunResult result;
  result.requested_duration_ms = options.steady_sec * 1000.0;
  result.grid_blocks = grid_blocks;

  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &result.occupancy_max_active_blocks_per_sm,
          wgmma_control_kernel_sm90a,
          kControlThreads,
          0),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(wgmma_control)");
  cudaFuncAttributes attributes{};
  check_cuda(cudaFuncGetAttributes(&attributes, wgmma_control_kernel_sm90a),
             "cudaFuncGetAttributes(wgmma_control)");
  result.registers_per_thread = attributes.numRegs;
  result.local_memory_bytes_per_thread = attributes.localSizeBytes;
  result.effective_blocks_per_sm_estimate = std::min(
      options.blocks_per_sm, result.occupancy_max_active_blocks_per_sm);
  result.occupancy_limited =
      options.blocks_per_sm > result.occupancy_max_active_blocks_per_sm;
  result.blocks_per_sm_resource_feasible = !result.occupancy_limited;

  unsigned long long* device_wakeup_checks = nullptr;
  std::vector<unsigned long long> host_wakeup_checks(
      static_cast<std::size_t>(grid_blocks));
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&device_wakeup_checks),
                 host_wakeup_checks.size() * sizeof(unsigned long long)),
             "cudaMalloc(wgmma control wakeup checks)");

  cudaEvent_t start{};
  cudaEvent_t stop{};
  try {
    if (options.warmup_sec > 0.0) {
      launch_control(
          grid_blocks,
          duration_to_nanoseconds(options.warmup_sec),
          options.sleep_ns,
          device_wakeup_checks);
      check_cuda(cudaDeviceSynchronize(),
                 "cudaDeviceSynchronize(wgmma control warmup)");
    }

    check_cuda(cudaEventCreate(&start), "cudaEventCreate(wgmma control start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(wgmma control stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(wgmma control start)");
    launch_control(
        grid_blocks,
        duration_to_nanoseconds(options.steady_sec),
        options.sleep_ns,
        device_wakeup_checks);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(wgmma control stop)");
    check_cuda(cudaEventSynchronize(stop),
               "cudaEventSynchronize(wgmma control stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime(wgmma control)");
    result.actual_elapsed_ms = static_cast<double>(elapsed_ms);

    check_cuda(cudaMemcpy(
                   host_wakeup_checks.data(),
                   device_wakeup_checks,
                   host_wakeup_checks.size() * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy(wgmma control wakeup checks)");
    for (unsigned long long count : host_wakeup_checks) {
      result.wakeup_checks += count;
    }
    result.final_global_store_bytes =
        host_wakeup_checks.size() * sizeof(unsigned long long);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_wakeup_checks);
    return result;
  } catch (...) {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    cudaFree(device_wakeup_checks);
    throw;
  }
}

}  // namespace gpu_power_validation
