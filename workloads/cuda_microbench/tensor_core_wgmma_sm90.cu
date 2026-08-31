#include "tensor_core_wgmma_sm90.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

namespace gpu_power_validation {
namespace {

using namespace cute;

constexpr int kWarpgroupThreads = 128;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using TileShape = Shape<_64, _64, _16>;
using TiledMma = decltype(make_tiled_mma(
    GMMA::ss_op_selector<
        ElementA,
        ElementB,
        float,
        TileShape,
        GMMA::Major::K,
        GMMA::Major::K>()));
static_assert(decltype(size(TiledMma{}))::value == kWarpgroupThreads,
              "The selected SM90 WGMMA atom must map to one 128-thread warpgroup");
using SmemLayoutAtomA = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K,
        ElementA,
        _64,
        _16>());
using SmemLayoutAtomB = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K,
        ElementB,
        _64,
        _16>());
using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<_64, _16>{}));
using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<_64, _16>{}));

struct alignas(128) WgmmaSharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> b;
  unsigned long long start_cycle;
  int continue_running;
};

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << call << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

template <int WaitGroup>
CUTE_DEVICE void wait_for_wgmma() {
  static_assert(WaitGroup == 0 || WaitGroup == 1, "Phase 1 supports wait groups 0 or 1");
  warpgroup_wait<WaitGroup>();
}

template <int AccumulatorSets, int WaitGroup>
__global__ __launch_bounds__(kWarpgroupThreads) void wgmma_persistent_kernel(
    unsigned long long duration_cycles,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  static_assert(AccumulatorSets == 1 || AccumulatorSets == 2,
                "Phase 1 supports one or two accumulator sets");
  static_assert(WaitGroup >= 0 && WaitGroup <= 1,
                "Phase 1 supports wait groups 0 or 1");
  static_assert(WaitGroup < AccumulatorSets,
                "The wait depth must be smaller than the accumulator-set count");

  __shared__ WgmmaSharedStorage storage;

  for (int index = threadIdx.x; index < cosize_v<SmemLayoutA>; index += blockDim.x) {
    storage.a[index] = ElementA(1.0f);
  }
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutB>; index += blockDim.x) {
    storage.b[index] = ElementB(1.0f);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(storage.b.begin()), SmemLayoutB{});

  TiledMma mma;
  ThrMMA thread_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);
  Tensor acc0 = partition_fragment_C(mma, Shape<_64, _64>{});
  Tensor acc1 = partition_fragment_C(mma, Shape<_64, _64>{});
  clear(acc0);
  if constexpr (AccumulatorSets == 2) {
    clear(acc1);
  }

  if (threadIdx.x == 0) {
    storage.start_cycle = clock64();
    storage.continue_running = 1;
  }
  __syncthreads();

  unsigned long long completed_ops = 0;
  while (storage.continue_running != 0) {
    for (int op = 0; op < ops_per_check; ++op) {
      bool use_acc1 = AccumulatorSets == 2 && ((op & 1) != 0);
      if (use_acc1) {
        warpgroup_fence_operand(acc1);
        warpgroup_arrive();
        cute::gemm(mma, tCrA, tCrB, acc1);
        warpgroup_commit_batch();
        wait_for_wgmma<WaitGroup>();
        if constexpr (WaitGroup == 0) {
          warpgroup_fence_operand(acc1);
        } else if (op > 0) {
          // wait<1> completes the older group, which used the other accumulator.
          warpgroup_fence_operand(acc0);
        }
      } else {
        warpgroup_fence_operand(acc0);
        warpgroup_arrive();
        cute::gemm(mma, tCrA, tCrB, acc0);
        warpgroup_commit_batch();
        wait_for_wgmma<WaitGroup>();
        if constexpr (WaitGroup == 0) {
          warpgroup_fence_operand(acc0);
        } else if (op > 0) {
          // wait<1> completes the older group, which used the other accumulator.
          warpgroup_fence_operand(acc1);
        }
      }
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(acc0);
    if constexpr (AccumulatorSets == 2) {
      warpgroup_fence_operand(acc1);
    }
    if (threadIdx.x == 0) {
      completed_ops += static_cast<unsigned long long>(ops_per_check);
      storage.continue_running =
          (clock64() - storage.start_cycle) < duration_cycles ? 1 : 0;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    cta_op_counts[blockIdx.x] = completed_ops;
    float output = acc0(0);
    if constexpr (AccumulatorSets == 2) {
      output += acc1(0);
    }
    cta_outputs[blockIdx.x] = output;
  }
}

template <int AccumulatorSets>
__global__ __launch_bounds__(kWarpgroupThreads) void wgmma_correctness_kernel(float* output) {
  __shared__ WgmmaSharedStorage storage;
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutA>; index += blockDim.x) {
    storage.a[index] = ElementA(1.0f);
  }
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutB>; index += blockDim.x) {
    storage.b[index] = ElementB(1.0f);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(storage.b.begin()), SmemLayoutB{});
  TiledMma mma;
  ThrMMA thread_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);
  Tensor accumulator = partition_fragment_C(mma, Shape<_64, _64>{});
  clear(accumulator);

  warpgroup_fence_operand(accumulator);
  warpgroup_arrive();
  cute::gemm(mma, tCrA, tCrB, accumulator);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(accumulator);

  if (threadIdx.x == 0) {
    output[0] = accumulator(0);
  }
}

template <int AccumulatorSets, int WaitGroup>
int query_occupancy() {
  int blocks_per_sm = 0;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          wgmma_persistent_kernel<AccumulatorSets, WaitGroup>,
          kWarpgroupThreads,
          0),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(wgmma_persistent)");
  return blocks_per_sm;
}

int query_occupancy_dispatch(int accumulator_sets, int wait_group) {
  if (accumulator_sets == 1 && wait_group == 0) {
    return query_occupancy<1, 0>();
  }
  if (accumulator_sets == 2 && wait_group == 0) {
    return query_occupancy<2, 0>();
  }
  if (accumulator_sets == 2 && wait_group == 1) {
    return query_occupancy<2, 1>();
  }
  throw std::runtime_error(
      "wgmma_persistent requires wait_group < accumulator_sets; supported pairs are 1/0, 2/0, and 2/1");
}

template <int AccumulatorSets, int WaitGroup>
void launch_persistent(
    int grid_blocks,
    unsigned long long duration_cycles,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  wgmma_persistent_kernel<AccumulatorSets, WaitGroup>
      <<<grid_blocks, kWarpgroupThreads>>>(
          duration_cycles,
          ops_per_check,
          cta_op_counts,
          cta_outputs);
  check_cuda(cudaGetLastError(), "wgmma_persistent_kernel");
}

void launch_persistent_dispatch(
    int accumulator_sets,
    int wait_group,
    int grid_blocks,
    unsigned long long duration_cycles,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  if (accumulator_sets == 1 && wait_group == 0) {
    launch_persistent<1, 0>(
        grid_blocks, duration_cycles, ops_per_check, cta_op_counts, cta_outputs);
    return;
  }
  if (accumulator_sets == 2 && wait_group == 0) {
    launch_persistent<2, 0>(
        grid_blocks, duration_cycles, ops_per_check, cta_op_counts, cta_outputs);
    return;
  }
  if (accumulator_sets == 2 && wait_group == 1) {
    launch_persistent<2, 1>(
        grid_blocks, duration_cycles, ops_per_check, cta_op_counts, cta_outputs);
    return;
  }
  throw std::runtime_error(
      "wgmma_persistent requires wait_group < accumulator_sets; supported pairs are 1/0, 2/0, and 2/1");
}

double run_correctness_smoke() {
  float* device_output = nullptr;
  float host_output = 0.0f;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(float)),
             "cudaMalloc(wgmma correctness output)");
  try {
    wgmma_correctness_kernel<1><<<1, kWarpgroupThreads>>>(device_output);
    check_cuda(cudaGetLastError(), "wgmma_correctness_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(wgmma correctness)");
    check_cuda(cudaMemcpy(
                   &host_output,
                   device_output,
                   sizeof(float),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy(wgmma correctness output)");
    cudaFree(device_output);
    return static_cast<double>(host_output);
  } catch (...) {
    cudaFree(device_output);
    throw;
  }
}

unsigned long long duration_to_cycles(int device, double seconds) {
  int clock_rate_khz = 0;
  check_cuda(
      cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device),
      "cudaDeviceGetAttribute(cudaDevAttrClockRate)");
  if (clock_rate_khz <= 0) {
    throw std::runtime_error("Invalid cudaDevAttrClockRate for wgmma_persistent");
  }
  return static_cast<unsigned long long>(
      seconds * static_cast<double>(clock_rate_khz) * 1000.0);
}

}  // namespace

WgmmaRunResult run_wgmma_persistent_sm90a(const WgmmaRunOptions& options) {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, options.device),
             "cudaGetDeviceProperties(wgmma_persistent)");
  if (properties.major != 9 || properties.minor != 0) {
    throw std::runtime_error(
        "wgmma_persistent requires NVIDIA Hopper compute capability 9.0 and an sm_90a build; no fallback is provided");
  }
  if (options.ops_per_check <= 0) {
    throw std::runtime_error("--wgmma-ops-per-check must be > 0");
  }

  const int grid_blocks = options.requested_sm_count * options.blocks_per_sm;
  const int occupancy_max_active_blocks_per_sm =
      query_occupancy_dispatch(options.accumulator_sets, options.wait_group);

  WgmmaRunResult result;
  result.grid_blocks = grid_blocks;
  result.occupancy_max_active_blocks_per_sm = occupancy_max_active_blocks_per_sm;
  result.effective_blocks_per_sm_estimate =
      std::min(options.blocks_per_sm, occupancy_max_active_blocks_per_sm);
  result.occupancy_limited = options.blocks_per_sm > occupancy_max_active_blocks_per_sm;
  result.correctness_observed = run_correctness_smoke();
  result.correctness_abs_error =
      std::abs(result.correctness_observed - result.correctness_reference);
  result.correctness_smoke_passed = result.correctness_abs_error <= 0.1;
  if (!result.correctness_smoke_passed) {
    throw std::runtime_error(
        "wgmma_persistent correctness smoke failed: expected approximately 16.0 from one BF16 WGMMA operation");
  }

  unsigned long long* device_counts = nullptr;
  float* device_outputs = nullptr;
  std::vector<unsigned long long> host_counts(static_cast<std::size_t>(grid_blocks));
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&device_counts),
                 host_counts.size() * sizeof(unsigned long long)),
             "cudaMalloc(wgmma counts)");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&device_outputs),
                 host_counts.size() * sizeof(float)),
             "cudaMalloc(wgmma outputs)");

  try {
    if (options.warmup_sec > 0.0) {
      launch_persistent_dispatch(
          options.accumulator_sets,
          options.wait_group,
          grid_blocks,
          duration_to_cycles(options.device, options.warmup_sec),
          options.ops_per_check,
          device_counts,
          device_outputs);
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(wgmma warmup)");
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(wgmma start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(wgmma stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(wgmma start)");
    launch_persistent_dispatch(
        options.accumulator_sets,
        options.wait_group,
        grid_blocks,
        duration_to_cycles(options.device, options.steady_sec),
        options.ops_per_check,
        device_counts,
        device_outputs);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(wgmma stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(wgmma stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime(wgmma)");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    check_cuda(cudaMemcpy(
                   host_counts.data(),
                   device_counts,
                   host_counts.size() * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy(wgmma counts)");
    for (unsigned long long count : host_counts) {
      result.wgmma_ops_executed += count;
    }
    result.actual_elapsed_ms = static_cast<double>(elapsed_ms);
    result.initial_global_load_bytes = 0;
    result.steady_global_load_bytes_per_loop = 0;
    result.final_global_store_bytes =
        host_counts.size() * (sizeof(unsigned long long) + sizeof(float));

    cudaFree(device_counts);
    cudaFree(device_outputs);
    return result;
  } catch (...) {
    cudaFree(device_counts);
    cudaFree(device_outputs);
    throw;
  }
}

}  // namespace gpu_power_validation
