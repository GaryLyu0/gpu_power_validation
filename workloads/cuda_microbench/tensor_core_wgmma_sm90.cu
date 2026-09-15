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

template <int InstructionN>
struct WgmmaInstructionN;

template <>
struct WgmmaInstructionN<64> {
  using Extent = _64;
  using Atom = SM90_64x64x16_F32BF16BF16_SS<
      GMMA::Major::K,
      GMMA::Major::K>;
};

template <>
struct WgmmaInstructionN<128> {
  using Extent = _128;
  using Atom = SM90_64x128x16_F32BF16BF16_SS<
      GMMA::Major::K,
      GMMA::Major::K>;
};

template <>
struct WgmmaInstructionN<256> {
  using Extent = _256;
  using Atom = SM90_64x256x16_F32BF16BF16_SS<
      GMMA::Major::K,
      GMMA::Major::K>;
};

template <int InstructionN>
using InstructionNType = typename WgmmaInstructionN<InstructionN>::Extent;

template <int InstructionN>
using WgmmaAtom = typename WgmmaInstructionN<InstructionN>::Atom;

template <int InstructionN>
using TiledMma = decltype(make_tiled_mma(WgmmaAtom<InstructionN>{}));

using SmemLayoutAtomA = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K,
        ElementA,
        _64,
        _16>());
template <int InstructionN>
using SmemLayoutAtomB = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K,
        ElementB,
        InstructionNType<InstructionN>,
        _16>());
using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<_64, _16>{}));
template <int InstructionN>
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB<InstructionN>{},
    Shape<InstructionNType<InstructionN>, _16>{}));

template <int InstructionN>
struct alignas(128) WgmmaSharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB<InstructionN>>> b;
  unsigned long long start_time_ns;
  int continue_running;
};

template <int InstructionN>
struct alignas(128) WgmmaDutySharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB<InstructionN>>> b;
  unsigned long long start_time_ns;
  unsigned long long end_time_ns;
  unsigned long long segment_start_ns;
  unsigned long long phase_deadline_ns;
  unsigned long long measured_active_ns;
  unsigned long long measured_idle_ns;
  unsigned int sleep_ns;
  int continue_running;
  int active_phase;
  int should_sleep;
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

template <int WaitGroup>
CUTE_DEVICE void wait_for_wgmma() {
  static_assert(WaitGroup >= 0 && WaitGroup <= 3,
                "Phase 2 supports wait groups from 0 through 3");
  warpgroup_wait<WaitGroup>();
}

template <int WaitGroup, class Mma, class FragmentA, class FragmentB, class Accumulator>
CUTE_DEVICE void issue_wgmma_group(
    Mma& mma,
    FragmentA& fragment_a,
    FragmentB& fragment_b,
    Accumulator& accumulator) {
  warpgroup_fence_operand(accumulator);
  warpgroup_arrive();
  cute::gemm(mma, fragment_a, fragment_b, accumulator);
  warpgroup_commit_batch();
  wait_for_wgmma<WaitGroup>();
  if constexpr (WaitGroup == 0) {
    warpgroup_fence_operand(accumulator);
  }
}

template <int InstructionN, int AccumulatorSets, int WaitGroup>
__global__ __launch_bounds__(kWarpgroupThreads) void wgmma_persistent_kernel(
    unsigned long long duration_ns,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  static_assert(AccumulatorSets >= 1 && AccumulatorSets <= 4,
                "Phase 2 supports one through four accumulator sets");
  static_assert(WaitGroup >= 0 && WaitGroup <= 3,
                "Phase 2 supports wait groups from 0 through 3");
  static_assert(WaitGroup < AccumulatorSets,
                "The wait depth must be smaller than the accumulator-set count");
  static_assert(decltype(size(TiledMma<InstructionN>{}))::value == kWarpgroupThreads,
                "The selected SM90 WGMMA atom must map to one 128-thread warpgroup");

  __shared__ WgmmaSharedStorage<InstructionN> storage;

  for (int index = threadIdx.x; index < cosize_v<SmemLayoutA>; index += blockDim.x) {
    storage.a.begin()[index] = ElementA(1.0f);
  }
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutB<InstructionN>>;
       index += blockDim.x) {
    storage.b.begin()[index] = ElementB(1.0f);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(
      make_smem_ptr(storage.b.begin()), SmemLayoutB<InstructionN>{});

  TiledMma<InstructionN> mma;
  ThrMMA thread_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);
  using OutputShape = Shape<_64, InstructionNType<InstructionN>>;
  Tensor acc0 = partition_fragment_C(mma, OutputShape{});
  Tensor acc1 = partition_fragment_C(mma, OutputShape{});
  Tensor acc2 = partition_fragment_C(mma, OutputShape{});
  Tensor acc3 = partition_fragment_C(mma, OutputShape{});
  clear(acc0);
  if constexpr (AccumulatorSets >= 2) {
    clear(acc1);
  }
  if constexpr (AccumulatorSets >= 3) {
    clear(acc2);
  }
  if constexpr (AccumulatorSets >= 4) {
    clear(acc3);
  }

  if (threadIdx.x == 0) {
    storage.start_time_ns = read_globaltimer_ns();
    storage.continue_running = 1;
  }
  __syncthreads();

  unsigned long long completed_ops = 0;
  while (storage.continue_running != 0) {
    int operations_remaining = ops_per_check;
    while (operations_remaining >= AccumulatorSets) {
      // Explicit fragments keep accumulator selection compile-time specialized;
      // no runtime-indexed accumulator array can spill into local memory.
      // Since WaitGroup < AccumulatorSets, the prior group targeting an
      // accumulator is complete before the next unrolled round reuses it.
      issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc0);
      if constexpr (AccumulatorSets >= 2) {
        issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc1);
      }
      if constexpr (AccumulatorSets >= 3) {
        issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc2);
      }
      if constexpr (AccumulatorSets >= 4) {
        issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc3);
      }
      operations_remaining -= AccumulatorSets;
    }
    if (operations_remaining >= 1) {
      issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc0);
    }
    if constexpr (AccumulatorSets >= 2) {
      if (operations_remaining >= 2) {
        issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc1);
      }
    }
    if constexpr (AccumulatorSets >= 3) {
      if (operations_remaining >= 3) {
        issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc2);
      }
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(acc0);
    if constexpr (AccumulatorSets >= 2) {
      warpgroup_fence_operand(acc1);
    }
    if constexpr (AccumulatorSets >= 3) {
      warpgroup_fence_operand(acc2);
    }
    if constexpr (AccumulatorSets >= 4) {
      warpgroup_fence_operand(acc3);
    }
    if (threadIdx.x == 0) {
      completed_ops += static_cast<unsigned long long>(ops_per_check);
      storage.continue_running =
          (read_globaltimer_ns() - storage.start_time_ns) < duration_ns ? 1 : 0;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    cta_op_counts[blockIdx.x] = completed_ops;
    float output = acc0(0);
    if constexpr (AccumulatorSets >= 2) {
      output += acc1(0);
    }
    if constexpr (AccumulatorSets >= 3) {
      output += acc2(0);
    }
    if constexpr (AccumulatorSets >= 4) {
      output += acc3(0);
    }
    cta_outputs[blockIdx.x] = output;
  }
}

template <int InstructionN, int AccumulatorSets, int WaitGroup>
__global__ __launch_bounds__(kWarpgroupThreads) void wgmma_duty_persistent_kernel(
    unsigned long long duration_ns,
    unsigned long long period_ns,
    unsigned long long active_window_ns,
    int duty_check_ops,
    unsigned long long* cta_op_counts,
    float* cta_outputs,
    unsigned long long* cta_active_ns,
    unsigned long long* cta_idle_ns) {
  static_assert(InstructionN == 128,
                "Temporal duty control is frozen to the validated N128 WGMMA atom");
  static_assert(AccumulatorSets == 2,
                "Temporal duty control is frozen to two accumulator sets");
  static_assert(WaitGroup == 1,
                "Temporal duty control is frozen to wait_group=1");
  static_assert(decltype(size(TiledMma<InstructionN>{}))::value == kWarpgroupThreads,
                "The selected SM90 WGMMA atom must map to one 128-thread warpgroup");

  __shared__ WgmmaDutySharedStorage<InstructionN> storage;

  for (int index = threadIdx.x; index < cosize_v<SmemLayoutA>; index += blockDim.x) {
    storage.a.begin()[index] = ElementA(1.0f);
  }
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutB<InstructionN>>;
       index += blockDim.x) {
    storage.b.begin()[index] = ElementB(1.0f);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(
      make_smem_ptr(storage.b.begin()), SmemLayoutB<InstructionN>{});
  TiledMma<InstructionN> mma;
  ThrMMA thread_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);
  using OutputShape = Shape<_64, InstructionNType<InstructionN>>;
  Tensor acc0 = partition_fragment_C(mma, OutputShape{});
  Tensor acc1 = partition_fragment_C(mma, OutputShape{});
  clear(acc0);
  clear(acc1);

  if (threadIdx.x == 0) {
    storage.start_time_ns = read_globaltimer_ns();
    storage.end_time_ns = storage.start_time_ns + duration_ns;
    storage.measured_active_ns = 0;
    storage.measured_idle_ns = 0;
    storage.continue_running = 1;
  }
  __syncthreads();

  unsigned long long completed_ops = 0;
  while (storage.continue_running != 0) {
    if (threadIdx.x == 0) {
      const unsigned long long now_ns = read_globaltimer_ns();
      storage.continue_running = now_ns < storage.end_time_ns ? 1 : 0;
      const unsigned long long phase_ns = now_ns % period_ns;
      storage.active_phase =
          active_window_ns > 0 && phase_ns < active_window_ns ? 1 : 0;
      storage.segment_start_ns = now_ns;
      const unsigned long long next_phase_ns = storage.active_phase != 0
                                                    ? now_ns + active_window_ns - phase_ns
                                                    : now_ns + period_ns - phase_ns;
      storage.phase_deadline_ns = next_phase_ns < storage.end_time_ns
                                      ? next_phase_ns
                                      : storage.end_time_ns;
    }
    __syncthreads();

    if (storage.continue_running == 0) {
      break;
    }

    if (storage.active_phase != 0) {
      do {
        int operations_remaining = duty_check_ops;
        while (operations_remaining >= AccumulatorSets) {
          // This is the same N128/2-accumulator/wait1 issue primitive as the
          // validated full-duty path; only the chunk boundary adds timing control.
          issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc0);
          issue_wgmma_group<WaitGroup>(mma, tCrA, tCrB, acc1);
          operations_remaining -= AccumulatorSets;
        }

        if (threadIdx.x == 0) {
          const unsigned long long now_ns = read_globaltimer_ns();
          completed_ops += static_cast<unsigned long long>(duty_check_ops);
          storage.measured_active_ns += now_ns - storage.segment_start_ns;
          storage.segment_start_ns = now_ns;
          storage.continue_running = now_ns < storage.end_time_ns ? 1 : 0;
          storage.active_phase =
              storage.continue_running != 0 && now_ns < storage.phase_deadline_ns
                  ? 1
                  : 0;
        }
        __syncthreads();
      } while (storage.active_phase != 0);

      // Timer checks preserve the wait1/two-accumulator pipeline. Fully drain
      // only after the active boundary or kernel end has actually arrived.
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc0);
      warpgroup_fence_operand(acc1);
      if (threadIdx.x == 0) {
        const unsigned long long now_ns = read_globaltimer_ns();
        storage.measured_active_ns += now_ns - storage.segment_start_ns;
        storage.continue_running = now_ns < storage.end_time_ns ? 1 : 0;
      }
      __syncthreads();
      continue;
    }

    // All lanes sleep together in bounded chunks. The device-wide globaltimer,
    // rather than the nominal nanosleep duration, determines the phase boundary.
    while (true) {
      if (threadIdx.x == 0) {
        const unsigned long long now_ns = read_globaltimer_ns();
        if (now_ns < storage.phase_deadline_ns) {
          const unsigned long long remaining_ns = storage.phase_deadline_ns - now_ns;
          unsigned long long sleep_ns = 0;
          if (remaining_ns > 20000ULL) {
            sleep_ns = remaining_ns / 2ULL < 10000ULL
                           ? remaining_ns / 2ULL
                           : 10000ULL;
          } else if (remaining_ns > 2000ULL) {
            sleep_ns = remaining_ns / 2ULL < 1000ULL
                           ? remaining_ns / 2ULL
                           : 1000ULL;
          } else {
            sleep_ns = remaining_ns / 2ULL;
            if (sleep_ns == 0) {
              sleep_ns = 1;
            }
          }
          storage.sleep_ns = static_cast<unsigned int>(sleep_ns);
          storage.should_sleep = 1;
        } else {
          storage.should_sleep = 0;
        }
      }
      __syncthreads();
      if (storage.should_sleep == 0) {
        break;
      }
      __nanosleep(storage.sleep_ns);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const unsigned long long now_ns = read_globaltimer_ns();
      storage.measured_idle_ns += now_ns - storage.segment_start_ns;
      storage.continue_running = now_ns < storage.end_time_ns ? 1 : 0;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    cta_op_counts[blockIdx.x] = completed_ops;
    cta_active_ns[blockIdx.x] = storage.measured_active_ns;
    cta_idle_ns[blockIdx.x] = storage.measured_idle_ns;
    cta_outputs[blockIdx.x] = acc0(0) + acc1(0);
  }
}

template <int InstructionN>
__global__ __launch_bounds__(kWarpgroupThreads) void wgmma_correctness_kernel(float* output) {
  static_assert(decltype(size(TiledMma<InstructionN>{}))::value == kWarpgroupThreads,
                "The selected SM90 WGMMA atom must map to one 128-thread warpgroup");
  __shared__ WgmmaSharedStorage<InstructionN> storage;
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutA>; index += blockDim.x) {
    storage.a.begin()[index] = ElementA(1.0f);
  }
  for (int index = threadIdx.x; index < cosize_v<SmemLayoutB<InstructionN>>;
       index += blockDim.x) {
    storage.b.begin()[index] = ElementB(1.0f);
  }
  __syncthreads();

  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(
      make_smem_ptr(storage.b.begin()), SmemLayoutB<InstructionN>{});
  TiledMma<InstructionN> mma;
  ThrMMA thread_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);
  Tensor accumulator = partition_fragment_C(
      mma, Shape<_64, InstructionNType<InstructionN>>{});
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

struct KernelResourceReport {
  int occupancy_max_active_blocks_per_sm = 0;
  int registers_per_thread = 0;
  std::size_t local_memory_bytes_per_thread = 0;
};

template <int InstructionN, int AccumulatorSets, int WaitGroup>
KernelResourceReport query_kernel_resources() {
  KernelResourceReport report;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &report.occupancy_max_active_blocks_per_sm,
          wgmma_persistent_kernel<InstructionN, AccumulatorSets, WaitGroup>,
          kWarpgroupThreads,
          0),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(wgmma_persistent)");
  cudaFuncAttributes attributes{};
  check_cuda(
      cudaFuncGetAttributes(
          &attributes,
          wgmma_persistent_kernel<InstructionN, AccumulatorSets, WaitGroup>),
      "cudaFuncGetAttributes(wgmma_persistent)");
  report.registers_per_thread = attributes.numRegs;
  report.local_memory_bytes_per_thread = attributes.localSizeBytes;
  return report;
}

template <int InstructionN, int AccumulatorSets>
KernelResourceReport query_wait_group_resources(int wait_group) {
  switch (wait_group) {
    case 0:
      return query_kernel_resources<InstructionN, AccumulatorSets, 0>();
    case 1:
      if constexpr (AccumulatorSets >= 2) {
        return query_kernel_resources<InstructionN, AccumulatorSets, 1>();
      }
      break;
    case 2:
      if constexpr (AccumulatorSets >= 3) {
        return query_kernel_resources<InstructionN, AccumulatorSets, 2>();
      }
      break;
    case 3:
      if constexpr (AccumulatorSets >= 4) {
        return query_kernel_resources<InstructionN, AccumulatorSets, 3>();
      }
      break;
  }
  throw std::runtime_error(
      "wgmma_persistent requires wait_group < accumulator_sets");
}

template <int InstructionN>
KernelResourceReport query_accumulator_resources_dispatch(
    int accumulator_sets,
    int wait_group) {
  switch (accumulator_sets) {
    case 1:
      return query_wait_group_resources<InstructionN, 1>(wait_group);
    case 2:
      return query_wait_group_resources<InstructionN, 2>(wait_group);
    case 3:
      return query_wait_group_resources<InstructionN, 3>(wait_group);
    case 4:
      return query_wait_group_resources<InstructionN, 4>(wait_group);
  }
  throw std::runtime_error("wgmma_persistent supports one through four accumulator sets");
}

KernelResourceReport query_kernel_resources_dispatch(
    int instruction_n,
    int accumulator_sets,
    int wait_group) {
  switch (instruction_n) {
    case 64:
      return query_accumulator_resources_dispatch<64>(accumulator_sets, wait_group);
    case 128:
      return query_accumulator_resources_dispatch<128>(accumulator_sets, wait_group);
    case 256:
      return query_accumulator_resources_dispatch<256>(accumulator_sets, wait_group);
  }
  throw std::runtime_error("--wgmma-instruction-n must be one of: 64, 128, 256");
}

KernelResourceReport query_duty_kernel_resources() {
  KernelResourceReport report;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &report.occupancy_max_active_blocks_per_sm,
          wgmma_duty_persistent_kernel<128, 2, 1>,
          kWarpgroupThreads,
          0),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(wgmma duty persistent)");
  cudaFuncAttributes attributes{};
  check_cuda(
      cudaFuncGetAttributes(
          &attributes,
          wgmma_duty_persistent_kernel<128, 2, 1>),
      "cudaFuncGetAttributes(wgmma duty persistent)");
  report.registers_per_thread = attributes.numRegs;
  report.local_memory_bytes_per_thread = attributes.localSizeBytes;
  return report;
}

template <int InstructionN, int AccumulatorSets, int WaitGroup>
void launch_persistent(
    int grid_blocks,
    unsigned long long duration_ns,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  wgmma_persistent_kernel<InstructionN, AccumulatorSets, WaitGroup>
      <<<grid_blocks, kWarpgroupThreads>>>(
          duration_ns,
          ops_per_check,
          cta_op_counts,
          cta_outputs);
  check_cuda(cudaGetLastError(), "wgmma_persistent_kernel");
}

template <int InstructionN, int AccumulatorSets>
void launch_wait_group_dispatch(
    int wait_group,
    int grid_blocks,
    unsigned long long duration_ns,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  switch (wait_group) {
    case 0:
      launch_persistent<InstructionN, AccumulatorSets, 0>(
          grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
      return;
    case 1:
      if constexpr (AccumulatorSets >= 2) {
        launch_persistent<InstructionN, AccumulatorSets, 1>(
            grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
        return;
      }
      break;
    case 2:
      if constexpr (AccumulatorSets >= 3) {
        launch_persistent<InstructionN, AccumulatorSets, 2>(
            grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
        return;
      }
      break;
    case 3:
      if constexpr (AccumulatorSets >= 4) {
        launch_persistent<InstructionN, AccumulatorSets, 3>(
            grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
        return;
      }
      break;
  }
  throw std::runtime_error(
      "wgmma_persistent requires wait_group < accumulator_sets");
}

template <int InstructionN>
void launch_accumulator_dispatch(
    int accumulator_sets,
    int wait_group,
    int grid_blocks,
    unsigned long long duration_ns,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  switch (accumulator_sets) {
    case 1:
      return launch_wait_group_dispatch<InstructionN, 1>(
          wait_group, grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
    case 2:
      return launch_wait_group_dispatch<InstructionN, 2>(
          wait_group, grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
    case 3:
      return launch_wait_group_dispatch<InstructionN, 3>(
          wait_group, grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
    case 4:
      return launch_wait_group_dispatch<InstructionN, 4>(
          wait_group, grid_blocks, duration_ns, ops_per_check, cta_op_counts, cta_outputs);
  }
  throw std::runtime_error("wgmma_persistent supports one through four accumulator sets");
}

void launch_persistent_dispatch(
    int instruction_n,
    int accumulator_sets,
    int wait_group,
    int grid_blocks,
    unsigned long long duration_ns,
    int ops_per_check,
    unsigned long long* cta_op_counts,
    float* cta_outputs) {
  switch (instruction_n) {
    case 64:
      return launch_accumulator_dispatch<64>(
          accumulator_sets, wait_group, grid_blocks, duration_ns, ops_per_check,
          cta_op_counts, cta_outputs);
    case 128:
      return launch_accumulator_dispatch<128>(
          accumulator_sets, wait_group, grid_blocks, duration_ns, ops_per_check,
          cta_op_counts, cta_outputs);
    case 256:
      return launch_accumulator_dispatch<256>(
          accumulator_sets, wait_group, grid_blocks, duration_ns, ops_per_check,
          cta_op_counts, cta_outputs);
  }
  throw std::runtime_error("--wgmma-instruction-n must be one of: 64, 128, 256");
}

void launch_duty_persistent(
    int grid_blocks,
    unsigned long long duration_ns,
    unsigned long long period_ns,
    unsigned long long active_window_ns,
    int duty_check_ops,
    unsigned long long* cta_op_counts,
    float* cta_outputs,
    unsigned long long* cta_active_ns,
    unsigned long long* cta_idle_ns) {
  wgmma_duty_persistent_kernel<128, 2, 1><<<grid_blocks, kWarpgroupThreads>>>(
      duration_ns,
      period_ns,
      active_window_ns,
      duty_check_ops,
      cta_op_counts,
      cta_outputs,
      cta_active_ns,
      cta_idle_ns);
  check_cuda(cudaGetLastError(), "wgmma_duty_persistent_kernel");
}

template <int InstructionN>
void launch_correctness_kernel(float* device_output) {
  wgmma_correctness_kernel<InstructionN><<<1, kWarpgroupThreads>>>(device_output);
}

double run_correctness_smoke(int instruction_n) {
  float* device_output = nullptr;
  float host_output = 0.0f;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(float)),
             "cudaMalloc(wgmma correctness output)");
  try {
    switch (instruction_n) {
      case 64:
        launch_correctness_kernel<64>(device_output);
        break;
      case 128:
        launch_correctness_kernel<128>(device_output);
        break;
      case 256:
        launch_correctness_kernel<256>(device_output);
        break;
      default:
        throw std::runtime_error("--wgmma-instruction-n must be one of: 64, 128, 256");
    }
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

unsigned long long duration_to_nanoseconds(double seconds) {
  const double duration_ns = seconds * 1.0e9;
  if (duration_ns <= 0.0) {
    throw std::runtime_error("WGMMA duration must be positive");
  }
  return static_cast<unsigned long long>(duration_ns);
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
  if (options.duty_cycle < 0.0 || options.duty_cycle > 1.0) {
    throw std::runtime_error("--duty-cycle must be in [0, 1]");
  }
  if (options.duty_period_ns == 0) {
    throw std::runtime_error("--wgmma-duty-period-ns must be > 0");
  }
  if (options.duty_check_ops <= 0) {
    throw std::runtime_error("--wgmma-duty-check-ops must be > 0");
  }
  if (options.instruction_n != 64 && options.instruction_n != 128 &&
      options.instruction_n != 256) {
    throw std::runtime_error("--wgmma-instruction-n must be one of: 64, 128, 256");
  }

  const int grid_blocks = options.requested_sm_count * options.blocks_per_sm;
  const bool full_duty = options.duty_cycle == 1.0;
  if (!full_duty &&
      (options.instruction_n != 128 || options.accumulator_sets != 2 ||
       options.wait_group != 1)) {
    throw std::runtime_error(
        "WGMMA temporal duty control currently requires the validated "
        "--wgmma-instruction-n 128 --wgmma-accumulator-sets 2 "
        "--wgmma-wait-group 1 configuration");
  }
  if (!full_duty && options.duty_cycle > 0.0 &&
      options.duty_check_ops % 2 != 0) {
    throw std::runtime_error(
        "--wgmma-duty-check-ops must be even for the validated two-accumulator "
        "pipeline so accumulator rotation remains legal across active chunks");
  }
  const KernelResourceReport kernel_resources = full_duty
                                                    ? query_kernel_resources_dispatch(
                                                          options.instruction_n,
                                                          options.accumulator_sets,
                                                          options.wait_group)
                                                    : query_duty_kernel_resources();

  unsigned long long requested_active_window_ns = static_cast<unsigned long long>(
      std::llround(static_cast<double>(options.duty_period_ns) * options.duty_cycle));
  requested_active_window_ns =
      std::min(requested_active_window_ns,
               static_cast<unsigned long long>(options.duty_period_ns));
  if (!full_duty && options.duty_cycle > 0.0 &&
      requested_active_window_ns >= options.duty_period_ns) {
    requested_active_window_ns = options.duty_period_ns - 1;
  }

  WgmmaRunResult result;
  result.instruction_n = options.instruction_n;
  result.requested_period_ns = options.duty_period_ns;
  result.requested_active_window_ns = requested_active_window_ns;
  result.requested_idle_window_ns =
      options.duty_period_ns - requested_active_window_ns;
  result.requested_duration_ms = options.steady_sec * 1000.0;
  result.grid_blocks = grid_blocks;
  result.occupancy_max_active_blocks_per_sm =
      kernel_resources.occupancy_max_active_blocks_per_sm;
  result.effective_blocks_per_sm_estimate =
      std::min(options.blocks_per_sm, result.occupancy_max_active_blocks_per_sm);
  result.occupancy_limited =
      options.blocks_per_sm > result.occupancy_max_active_blocks_per_sm;
  result.registers_per_thread = kernel_resources.registers_per_thread;
  result.local_memory_bytes_per_thread =
      kernel_resources.local_memory_bytes_per_thread;
  result.allows_at_least_two_resident_ctas_per_sm =
      result.occupancy_max_active_blocks_per_sm >= 2;
  result.correctness_observed = run_correctness_smoke(options.instruction_n);
  result.correctness_abs_error =
      std::abs(result.correctness_observed - result.correctness_reference);
  result.correctness_smoke_passed = result.correctness_abs_error <= 0.1;
  if (!result.correctness_smoke_passed) {
    throw std::runtime_error(
        "wgmma_persistent correctness smoke failed: expected approximately 16.0 from one BF16 WGMMA operation");
  }

  unsigned long long* device_counts = nullptr;
  float* device_outputs = nullptr;
  unsigned long long* device_active_ns = nullptr;
  unsigned long long* device_idle_ns = nullptr;
  std::vector<unsigned long long> host_counts(static_cast<std::size_t>(grid_blocks));
  std::vector<unsigned long long> host_active_ns;
  std::vector<unsigned long long> host_idle_ns;
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&device_counts),
                 host_counts.size() * sizeof(unsigned long long)),
             "cudaMalloc(wgmma counts)");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&device_outputs),
                 host_counts.size() * sizeof(float)),
             "cudaMalloc(wgmma outputs)");
  if (!full_duty) {
    host_active_ns.resize(host_counts.size());
    host_idle_ns.resize(host_counts.size());
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&device_active_ns),
                   host_counts.size() * sizeof(unsigned long long)),
               "cudaMalloc(wgmma active time)");
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&device_idle_ns),
                   host_counts.size() * sizeof(unsigned long long)),
               "cudaMalloc(wgmma idle time)");
  }

  try {
    if (options.warmup_sec > 0.0) {
      if (full_duty) {
        launch_persistent_dispatch(
            options.instruction_n,
            options.accumulator_sets,
            options.wait_group,
            grid_blocks,
            duration_to_nanoseconds(options.warmup_sec),
            options.ops_per_check,
            device_counts,
            device_outputs);
      } else {
        launch_duty_persistent(
            grid_blocks,
            duration_to_nanoseconds(options.warmup_sec),
            options.duty_period_ns,
            requested_active_window_ns,
            options.duty_check_ops,
            device_counts,
            device_outputs,
            device_active_ns,
            device_idle_ns);
      }
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(wgmma warmup)");
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(wgmma start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(wgmma stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(wgmma start)");
    if (full_duty) {
      launch_persistent_dispatch(
          options.instruction_n,
          options.accumulator_sets,
          options.wait_group,
          grid_blocks,
          duration_to_nanoseconds(options.steady_sec),
          options.ops_per_check,
          device_counts,
          device_outputs);
    } else {
      launch_duty_persistent(
          grid_blocks,
          duration_to_nanoseconds(options.steady_sec),
          options.duty_period_ns,
          requested_active_window_ns,
          options.duty_check_ops,
          device_counts,
          device_outputs,
          device_active_ns,
          device_idle_ns);
    }
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
    if (full_duty) {
      result.measured_active_ns = static_cast<unsigned long long>(
          result.actual_elapsed_ms * 1.0e6);
      result.measured_idle_ns = 0;
    } else {
      check_cuda(cudaMemcpy(
                     host_active_ns.data(),
                     device_active_ns,
                     host_active_ns.size() * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost),
                 "cudaMemcpy(wgmma active time)");
      check_cuda(cudaMemcpy(
                     host_idle_ns.data(),
                     device_idle_ns,
                     host_idle_ns.size() * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost),
                 "cudaMemcpy(wgmma idle time)");
      unsigned long long active_sum_ns = 0;
      unsigned long long idle_sum_ns = 0;
      for (std::size_t index = 0; index < host_active_ns.size(); ++index) {
        active_sum_ns += host_active_ns[index];
        idle_sum_ns += host_idle_ns[index];
      }
      result.measured_active_ns = active_sum_ns / host_active_ns.size();
      result.measured_idle_ns = idle_sum_ns / host_idle_ns.size();
    }
    result.initial_global_load_bytes = 0;
    result.steady_global_load_bytes_per_loop = 0;
    result.final_global_store_bytes = host_counts.size() *
        (sizeof(unsigned long long) + sizeof(float) +
         (full_duty ? 0 : 2 * sizeof(unsigned long long)));

    cudaFree(device_counts);
    cudaFree(device_outputs);
    cudaFree(device_active_ns);
    cudaFree(device_idle_ns);
    return result;
  } catch (...) {
    cudaFree(device_counts);
    cudaFree(device_outputs);
    cudaFree(device_active_ns);
    cudaFree(device_idle_ns);
    throw;
  }
}

}  // namespace gpu_power_validation
