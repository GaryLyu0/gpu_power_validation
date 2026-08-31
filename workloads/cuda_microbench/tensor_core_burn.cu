#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#if defined(HAVE_CUTLASS_CUTE)
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#endif
#if defined(HAVE_CUSPARSELT)
#include <cusparseLt.h>
#endif
#if defined(HAVE_WGMMA_SM90A)
#include "tensor_core_wgmma_sm90.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWmmaThreadsPerBlock = 128;
constexpr int kWmmaDynamicSmemBytes = 0;

struct Options {
  int device = 0;
  std::string dtype = "bf16";
  std::string engine = "cublas";
  int m = 8192;
  int n = 8192;
  int k = 8192;
  double duty_cycle = 1.0;
  double active_sm_fraction = 1.0;
  double warmup_sec = 30.0;
  double steady_sec = 60.0;
  double period_ms = 1000.0;
  int blocks_per_sm = 1;
  int mma_iters_per_loop = 64;
  int accumulators_per_warp = 1;
  int atomic_period = 1024;
  int batch_count = 1;
  int cutlass_tile_m = 16;
  int cutlass_tile_n = 8;
  int cutlass_tile_k = 16;
  int synthetic_m = 0;
  int synthetic_n = 0;
  int synthetic_k = 0;
  int synthetic_mma_ops_per_loop_override = 0;
  int wgmma_ops_per_check = 512;
  int wgmma_wait_group = 1;
  int wgmma_accumulator_sets = 2;
  std::string sparsity_mode = "none";
  double zero_ratio = 0.0;
  std::string zero_pattern = "regular_k";
  std::string sparse_operand = "A";
  std::string sparse_engine = "cusparselt";
};

struct Result {
  std::string dtype;
  std::string engine;
  int m = 0;
  int n = 0;
  int k = 0;
  double duty_cycle = 0.0;
  double active_sm_fraction = 0.0;
  int requested_sm_count = 0;
  double steady_sec = 0.0;
  double active_elapsed_ms = 0.0;
  std::uint64_t iterations = 0;
  double active_tflops = 0.0;
  double scheduled_tflops = 0.0;
  double period_ms = 0.0;
  double actual_elapsed_ms = 0.0;
  std::string duty_control_mode;
  std::string spatial_control_mode;
  bool sm_count_target_applied = false;
  int grid_blocks = 0;
  int blocks_per_sm = 0;
  int mma_iters_per_loop = 0;
  int accumulators_per_warp = 0;
  int atomic_period = 0;
  int batch_count = 1;
  double per_gemm_flops = 0.0;
  double total_flops_per_call = 0.0;
  std::string gemm_semantics = "";
  std::string expected_use_case = "";
  std::vector<std::string> warnings;
  int occupancy_max_active_blocks_per_sm = 0;
  int effective_blocks_per_sm_estimate = 0;
  bool occupancy_limited = false;
  double spatial_coverage_fraction = 0.0;
  std::string sparsity_test_dimension = "none";
  std::string tensor_core_execution_path = "dense_tensor_core";
  std::string sparsity_mode = "none";
  double zero_ratio = 0.0;
  std::string zero_pattern = "regular_k";
  std::string sparse_operand = "A";
  std::string sparse_engine = "";
  std::string sparse_pattern = "";
  bool uses_sparse_tensor_core = false;
  bool dense_mma_instruction_count_unchanged = true;
  bool correctness_smoke_passed = false;
  double correctness_reference = 0.0;
  double correctness_observed = 0.0;
  double correctness_abs_error = 0.0;
  int logical_m = 0;
  int logical_n = 0;
  int logical_k = 0;
  double dense_equivalent_flops = 0.0;
  double measured_runtime_ms = 0.0;
  double dense_baseline_tflops = 0.0;
  double sparse_tflops_or_dense_equivalent_tflops = 0.0;
  double speedup_vs_dense = 0.0;
  double joules_per_dense_equivalent_flop = 0.0;
  bool compression_time_excluded = false;
  bool setup_time_excluded = false;
  bool matrix_shape_is_real = true;
  bool synthetic_mnk_controls_mma_count = false;
  bool memory_traffic_minimized = false;
  std::string uses_global_ab = "";
  bool uses_shared_memory_tiles = false;
  int cutlass_tile_m = 0;
  int cutlass_tile_n = 0;
  int cutlass_tile_k = 0;
  int synthetic_m_tiles = 0;
  int synthetic_n_tiles = 0;
  int synthetic_k_tiles = 0;
  std::uint64_t synthetic_mma_ops_per_loop = 0;
  std::string synthetic_mma_ops_per_loop_source = "";
  int synthetic_mma_ops_per_loop_override = 0;
  std::uint64_t requested_synthetic_tile_ops = 0;
  std::uint64_t distributed_synthetic_ops_per_block = 0;
  std::uint64_t synthetic_mma_ops_cap = 0;
  bool synthetic_mma_ops_cap_applied = false;
  int cutlass_atom_shape_m = 0;
  int cutlass_atom_shape_n = 0;
  int cutlass_atom_shape_k = 0;
  std::string cutlass_atom_arch = "";
  bool uses_cutlass_tiled_mma_object = false;
  bool uses_cutlass_mma_atom_direct = false;
  int warpgroup_threads = 0;
  int warps_per_warpgroup = 0;
  int wgmma_instruction_m = 0;
  int wgmma_instruction_n = 0;
  int wgmma_instruction_k = 0;
  std::uint64_t wgmma_ops_executed = 0;
  double wgmma_flops_per_op = 0.0;
  int wgmma_wait_group = 0;
  int wgmma_accumulator_sets = 0;
  int wgmma_ops_per_check = 0;
  bool uses_tma = false;
  bool uses_tma_known = false;
  std::uint64_t wgmma_smem_operand_bytes_per_op = 0;
  std::uint64_t initial_global_load_bytes = 0;
  std::uint64_t steady_global_load_bytes_per_loop = 0;
  std::uint64_t final_global_store_bytes = 0;
  std::string note;
};

Result make_base_result(const Options& options, int requested_sm_count) {
  Result result;
  result.dtype = options.dtype;
  result.engine = options.engine;
  result.m = options.m;
  result.n = options.n;
  result.k = options.k;
  result.duty_cycle = options.duty_cycle;
  result.active_sm_fraction = options.active_sm_fraction;
  result.requested_sm_count = requested_sm_count;
  result.steady_sec = options.steady_sec;
  result.period_ms = options.period_ms;
  result.blocks_per_sm = options.engine == "wmma_persistent" ? options.blocks_per_sm : 0;
  result.mma_iters_per_loop =
      options.engine == "wmma_persistent" ? options.mma_iters_per_loop : 0;
  result.accumulators_per_warp =
      options.engine == "wmma_persistent" ? options.accumulators_per_warp : 0;
  result.atomic_period = options.engine == "wmma_persistent" ? options.atomic_period : 0;
  result.batch_count = options.batch_count;
  result.per_gemm_flops =
      2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
      static_cast<double>(options.k);
  result.total_flops_per_call =
      result.per_gemm_flops * static_cast<double>(options.batch_count);
  result.sparsity_mode = options.sparsity_mode;
  result.zero_ratio = options.zero_ratio;
  result.zero_pattern = options.zero_pattern;
  result.sparse_operand = options.sparse_operand;
  result.sparse_engine =
      options.sparsity_mode == "structured_2to4" ? options.sparse_engine : "";
  result.sparse_pattern = options.sparsity_mode == "structured_2to4" ? "2:4" : "";
  result.uses_sparse_tensor_core = false;
  result.dense_mma_instruction_count_unchanged = options.sparsity_mode != "structured_2to4";
  result.spatial_coverage_fraction = options.active_sm_fraction;
  if (options.sparsity_mode == "dense_zero") {
    result.sparsity_test_dimension = "dense_zero_input_data_pattern";
  } else if (options.sparsity_mode == "structured_2to4") {
    result.sparsity_test_dimension = "structured_2to4_sparse_tensor_core";
    result.tensor_core_execution_path = "sparse_tensor_core";
  }
  result.logical_m = options.m;
  result.logical_n = options.n;
  result.logical_k = options.k;
  result.dense_equivalent_flops =
      2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
      static_cast<double>(options.k);
  return result;
}

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << call << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

void check_cublas(cublasStatus_t status, const char* call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream message;
    message << call << " failed with cublasStatus_t=" << static_cast<int>(status);
    throw std::runtime_error(message.str());
  }
}

#if defined(HAVE_CUSPARSELT)
void check_cusparselt(cusparseStatus_t status, const char* call) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::ostringstream message;
    message << call << " failed with cusparseStatus_t=" << static_cast<int>(status)
            << " (" << cusparseLtGetErrorName(status) << ": "
            << cusparseLtGetErrorString(status) << ")";
    throw std::runtime_error(message.str());
  }
}
#endif

double parse_double(const std::string& value, const std::string& name) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    throw std::runtime_error("Invalid " + name + ": " + value);
  }
  return parsed;
}

int parse_int(const std::string& value, const std::string& name) {
  char* end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed <= 0) {
    throw std::runtime_error("Invalid " + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

int parse_non_negative_int(const std::string& value, const std::string& name) {
  char* end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed < 0) {
    throw std::runtime_error("Invalid " + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

int parse_device(const std::string& value) {
  char* end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed < 0) {
    throw std::runtime_error("Invalid --device: " + value);
  }
  return static_cast<int>(parsed);
}

int ceil_div_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    std::string arg = argv[index];
    auto require_value = [&](const std::string& name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++index];
    };

    if (arg == "--device") {
      options.device = parse_device(require_value(arg));
    } else if (arg == "--dtype") {
      options.dtype = require_value(arg);
    } else if (arg == "--engine") {
      options.engine = require_value(arg);
    } else if (arg == "--m") {
      options.m = parse_int(require_value(arg), arg);
    } else if (arg == "--n") {
      options.n = parse_int(require_value(arg), arg);
    } else if (arg == "--k") {
      options.k = parse_int(require_value(arg), arg);
    } else if (arg == "--duty-cycle") {
      options.duty_cycle = parse_double(require_value(arg), arg);
    } else if (arg == "--active-sm-fraction") {
      options.active_sm_fraction = parse_double(require_value(arg), arg);
    } else if (arg == "--warmup-sec") {
      options.warmup_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--steady-sec") {
      options.steady_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--period-ms") {
      options.period_ms = parse_double(require_value(arg), arg);
    } else if (arg == "--blocks-per-sm") {
      options.blocks_per_sm = parse_int(require_value(arg), arg);
    } else if (arg == "--mma-iters-per-loop") {
      options.mma_iters_per_loop = parse_int(require_value(arg), arg);
    } else if (arg == "--accumulators-per-warp") {
      options.accumulators_per_warp = parse_int(require_value(arg), arg);
    } else if (arg == "--atomic-period") {
      options.atomic_period = parse_int(require_value(arg), arg);
    } else if (arg == "--batch-count") {
      options.batch_count = parse_int(require_value(arg), arg);
    } else if (arg == "--cutlass-tile-m") {
      options.cutlass_tile_m = parse_int(require_value(arg), arg);
    } else if (arg == "--cutlass-tile-n") {
      options.cutlass_tile_n = parse_int(require_value(arg), arg);
    } else if (arg == "--cutlass-tile-k") {
      options.cutlass_tile_k = parse_int(require_value(arg), arg);
    } else if (arg == "--synthetic-m") {
      options.synthetic_m = parse_int(require_value(arg), arg);
    } else if (arg == "--synthetic-n") {
      options.synthetic_n = parse_int(require_value(arg), arg);
    } else if (arg == "--synthetic-k") {
      options.synthetic_k = parse_int(require_value(arg), arg);
    } else if (arg == "--synthetic-mma-ops-per-loop") {
      options.synthetic_mma_ops_per_loop_override = parse_int(require_value(arg), arg);
    } else if (arg == "--wgmma-ops-per-check") {
      options.wgmma_ops_per_check = parse_int(require_value(arg), arg);
    } else if (arg == "--wgmma-wait-group") {
      options.wgmma_wait_group = parse_non_negative_int(require_value(arg), arg);
    } else if (arg == "--wgmma-accumulator-sets") {
      options.wgmma_accumulator_sets = parse_int(require_value(arg), arg);
    } else if (arg == "--sparsity-mode") {
      options.sparsity_mode = require_value(arg);
    } else if (arg == "--zero-ratio") {
      options.zero_ratio = parse_double(require_value(arg), arg);
    } else if (arg == "--zero-pattern") {
      options.zero_pattern = require_value(arg);
    } else if (arg == "--sparse-operand") {
      options.sparse_operand = require_value(arg);
    } else if (arg == "--sparse-engine") {
      options.sparse_engine = require_value(arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage: tensor_core_burn --device 0 --dtype bf16 "
          << "--engine cublas|cublas_strided_batched|wmma_persistent|cutlass_tile_burn|wgmma_persistent "
          << "--m 8192 --n 8192 "
          << "--k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 "
          << "--period-ms 1000 --blocks-per-sm 1 "
          << "--mma-iters-per-loop 64 --accumulators-per-warp 1 "
          << "--atomic-period 1024 --batch-count 1 "
          << "--cutlass-tile-m 16 --cutlass-tile-n 8 --cutlass-tile-k 16 "
          << "--synthetic-m 8192 --synthetic-n 8192 --synthetic-k 8192 "
          << "--synthetic-mma-ops-per-loop 256 "
          << "--wgmma-ops-per-check 512 --wgmma-wait-group 1 "
          << "--wgmma-accumulator-sets 2 "
          << "--sparsity-mode none|dense_zero|structured_2to4 "
          << "--zero-ratio 0.0 --zero-pattern regular_k "
          << "--sparse-operand A --sparse-engine cusparselt "
          << "--warmup-sec 30 --steady-sec 60\n\n"
          << "engine=cublas keeps the original cuBLAS GEMM active/idle windows.\n"
          << "engine=wmma_persistent launches persistent CTAs and controls "
          << "active/idle phases inside the CUDA kernel with clock64. "
          << "For wmma_persistent, --period-ms controls switching cadence while "
          << "--blocks-per-sm, --mma-iters-per-loop, and "
          << "--accumulators-per-warp control active intensity.\n"
          << "engine=cutlass_tile_burn is experimental: it is a CUTLASS/CuTe "
          << "MMA atom based synthetic Tensor Core burn. It does not use "
          << "top-level CUTLASS device::Gemm, does not load real A/B matrices, "
          << "does not use shared-memory A/B tiles, and cutlass-tile-m/n/k "
          << "define synthetic atom grouping rather than real tile-local GEMM "
          << "storage. --synthetic-mma-ops-per-loop manually overrides the "
          << "actual CuteMmaAtom::fma calls per active loop when small synthetic "
          << "shapes distribute to too few operations per block.\n"
          << "engine=wgmma_persistent is Hopper H100/SM90a only. It launches "
          << "one 128-thread warpgroup per CTA and executes asynchronous BF16 "
          << "SM90a WGMMA from A/B tiles initialized once in shared memory. "
          << "Phase 1 requires --duty-cycle 1.0, uses no TMA, performs no "
          << "steady-state global A/B loads, and counts FLOPs from actual "
          << "64x64x16 WGMMA operations. --wgmma-ops-per-check sets the "
          << "coarse timer-check batch; supported wait-group/accumulator-set "
          << "pairs are 0/1, 0/2, and 1/2.\n"
          << "sparsity-mode=dense_zero inserts zero values into dense cuBLAS operands "
          << "but does not use hardware sparse Tensor Cores.\n"
          << "sparsity-mode=structured_2to4 requires a real sparse backend such as "
          << "cuSPARSELt and fails if that backend is unavailable.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.dtype != "bf16") {
    throw std::runtime_error("Only --dtype bf16 is supported in this workload");
  }
  if (options.engine != "cublas" && options.engine != "cublas_strided_batched" &&
      options.engine != "wmma_persistent" && options.engine != "cutlass_tile_burn" &&
      options.engine != "wgmma_persistent") {
    throw std::runtime_error(
        "--engine must be cublas, cublas_strided_batched, wmma_persistent, cutlass_tile_burn, or wgmma_persistent");
  }
  if (options.duty_cycle < 0.0 || options.duty_cycle > 1.0) {
    throw std::runtime_error("--duty-cycle must be in [0, 1]");
  }
  if (options.active_sm_fraction < 0.1 || options.active_sm_fraction > 1.0) {
    throw std::runtime_error("--active-sm-fraction must be in [0.1, 1.0]");
  }
  if (options.warmup_sec < 0.0 || options.steady_sec <= 0.0) {
    throw std::runtime_error("--warmup-sec must be >= 0 and --steady-sec must be > 0");
  }
  if (options.period_ms <= 0.0) {
    throw std::runtime_error("--period-ms must be > 0");
  }
  if (options.blocks_per_sm <= 0) {
    throw std::runtime_error("--blocks-per-sm must be > 0");
  }
  if (options.mma_iters_per_loop <= 0) {
    throw std::runtime_error("--mma-iters-per-loop must be > 0");
  }
  if (options.accumulators_per_warp != 1 && options.accumulators_per_warp != 2 &&
      options.accumulators_per_warp != 4 && options.accumulators_per_warp != 8) {
    throw std::runtime_error("--accumulators-per-warp must be one of: 1, 2, 4, 8");
  }
  if (options.atomic_period <= 0) {
    throw std::runtime_error("--atomic-period must be > 0");
  }
  if (options.batch_count <= 0) {
    throw std::runtime_error("--batch-count must be > 0");
  }
  if (options.wgmma_wait_group != 0 && options.wgmma_wait_group != 1) {
    throw std::runtime_error("--wgmma-wait-group must be 0 or 1 in phase 1");
  }
  if (options.wgmma_accumulator_sets != 1 && options.wgmma_accumulator_sets != 2) {
    throw std::runtime_error("--wgmma-accumulator-sets must be 1 or 2 in phase 1");
  }
  if (options.wgmma_wait_group >= options.wgmma_accumulator_sets) {
    throw std::runtime_error(
        "--wgmma-wait-group must be smaller than --wgmma-accumulator-sets");
  }
  if (options.cutlass_tile_m <= 0 || options.cutlass_tile_n <= 0 ||
      options.cutlass_tile_k <= 0) {
    throw std::runtime_error("--cutlass-tile-m/n/k must be > 0");
  }
  if (options.sparsity_mode != "none" && options.sparsity_mode != "dense_zero" &&
      options.sparsity_mode != "structured_2to4") {
    throw std::runtime_error("--sparsity-mode must be none, dense_zero, or structured_2to4");
  }
  if (options.zero_ratio < 0.0 || options.zero_ratio >= 1.0) {
    throw std::runtime_error("--zero-ratio must be in [0.0, 1.0)");
  }
  if (options.zero_pattern != "random" && options.zero_pattern != "regular_k" &&
      options.zero_pattern != "block") {
    throw std::runtime_error("--zero-pattern must be random, regular_k, or block");
  }
  if (options.sparse_operand != "A" && options.sparse_operand != "B" &&
      options.sparse_operand != "both") {
    throw std::runtime_error("--sparse-operand must be A, B, or both");
  }
  if (options.sparse_engine != "cusparselt" && options.sparse_engine != "cutlass_spgemm") {
    throw std::runtime_error("--sparse-engine must be cusparselt or cutlass_spgemm");
  }
  if (options.engine == "wgmma_persistent" && options.sparsity_mode != "none") {
    throw std::runtime_error(
        "engine=wgmma_persistent phase 1 supports --sparsity-mode none only");
  }
  if (options.engine == "wgmma_persistent" && options.duty_cycle != 1.0) {
    throw std::runtime_error(
        "wgmma_persistent phase 1 currently supports duty_cycle=1.0 only; "
        "warpgroup-uniform duty-cycle control will be implemented separately");
  }
  return options;
}

class EventPair {
 public:
  EventPair() {
    check_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
  }

  ~EventPair() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  cudaEvent_t start() const { return start_; }
  cudaEvent_t stop() const { return stop_; }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

enum ZeroPattern {
  kZeroPatternRandom = 0,
  kZeroPatternRegularK = 1,
  kZeroPatternBlock = 2,
};

int zero_pattern_id(const std::string& pattern) {
  if (pattern == "random") {
    return kZeroPatternRandom;
  }
  if (pattern == "regular_k") {
    return kZeroPatternRegularK;
  }
  if (pattern == "block") {
    return kZeroPatternBlock;
  }
  throw std::runtime_error("Unknown zero pattern: " + pattern);
}

bool operand_selected(const Options& options, const std::string& operand) {
  return options.sparsity_mode == "dense_zero" &&
         (options.sparse_operand == operand || options.sparse_operand == "both");
}

__device__ unsigned int hash_u32(unsigned int value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__device__ bool should_zero_value(
    int k_index,
    int other_index,
    int k_extent,
    double zero_ratio,
    int zero_pattern) {
  if (zero_ratio <= 0.0) {
    return false;
  }

  if (zero_pattern == kZeroPatternBlock) {
    int zero_count = static_cast<int>(zero_ratio * static_cast<double>(k_extent));
    return k_index < zero_count;
  }

  if (zero_pattern == kZeroPatternRegularK) {
    constexpr int period = 1024;
    int zero_count = static_cast<int>(zero_ratio * static_cast<double>(period));
    return (k_index % period) < zero_count;
  }

  unsigned int mixed = hash_u32(
      static_cast<unsigned int>(k_index) * 1315423911U ^
      static_cast<unsigned int>(other_index) * 2654435761U);
  double unit = static_cast<double>(mixed) / static_cast<double>(0xffffffffU);
  return unit < zero_ratio;
}

bool should_zero_host(
    int k_index,
    int other_index,
    int k_extent,
    double zero_ratio,
    int zero_pattern) {
  if (zero_ratio <= 0.0) {
    return false;
  }
  if (zero_pattern == kZeroPatternBlock) {
    int zero_count = static_cast<int>(zero_ratio * static_cast<double>(k_extent));
    return k_index < zero_count;
  }
  if (zero_pattern == kZeroPatternRegularK) {
    constexpr int period = 1024;
    int zero_count = static_cast<int>(zero_ratio * static_cast<double>(period));
    return (k_index % period) < zero_count;
  }
  unsigned int mixed =
      static_cast<unsigned int>(k_index) * 1315423911U ^
      static_cast<unsigned int>(other_index) * 2654435761U;
  mixed ^= mixed >> 16;
  mixed *= 0x7feb352dU;
  mixed ^= mixed >> 15;
  mixed *= 0x846ca68bU;
  mixed ^= mixed >> 16;
  double unit = static_cast<double>(mixed) / static_cast<double>(0xffffffffU);
  return unit < zero_ratio;
}

__global__ void fill_bf16_kernel(
    __nv_bfloat16* data,
    int rows,
    int cols,
    int k_extent,
    int k_axis_is_row,
    double zero_ratio,
    int zero_pattern,
    float value) {
  std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  __nv_bfloat16 converted = __float2bfloat16(value);
  while (index < count) {
    int row = static_cast<int>(index % static_cast<std::size_t>(rows));
    int col = static_cast<int>(index / static_cast<std::size_t>(rows));
    int k_index = k_axis_is_row ? row : col;
    int other_index = k_axis_is_row ? col : row;
    data[index] = should_zero_value(k_index, other_index, k_extent, zero_ratio, zero_pattern)
                      ? __float2bfloat16(0.0f)
                      : converted;
    index += stride;
  }
}

void fill_bf16(
    __nv_bfloat16* data,
    int rows,
    int cols,
    int k_extent,
    bool k_axis_is_row,
    double zero_ratio,
    int zero_pattern,
    float value) {
  int threads = 256;
  std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  int blocks = static_cast<int>(std::min<std::size_t>((count + threads - 1) / threads, 65535));
  fill_bf16_kernel<<<blocks, threads>>>(
      data,
      rows,
      cols,
      k_extent,
      k_axis_is_row ? 1 : 0,
      zero_ratio,
      zero_pattern,
      value);
  check_cuda(cudaGetLastError(), "fill_bf16_kernel");
}

void run_gemm(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
      cublasGemmEx(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          options.m,
          options.n,
          options.k,
          &alpha,
          a,
          CUDA_R_16BF,
          options.m,
          b,
          CUDA_R_16BF,
          options.k,
          &beta,
          c,
          CUDA_R_32F,
          options.m,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx");
}

void run_strided_batched_gemm(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const long long int stride_a =
      static_cast<long long int>(options.m) * static_cast<long long int>(options.k);
  const long long int stride_b =
      static_cast<long long int>(options.k) * static_cast<long long int>(options.n);
  const long long int stride_c =
      static_cast<long long int>(options.m) * static_cast<long long int>(options.n);
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          options.m,
          options.n,
          options.k,
          &alpha,
          a,
          CUDA_R_16BF,
          options.m,
          stride_a,
          b,
          CUDA_R_16BF,
          options.k,
          stride_b,
          &beta,
          c,
          CUDA_R_32F,
          options.m,
          stride_c,
          options.batch_count,
          CUBLAS_COMPUTE_32F_FAST_16BF,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmStridedBatchedEx");
}

void run_active_window(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c,
    double seconds,
    std::uint64_t& iterations) {
  if (seconds <= 0.0) {
    return;
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    run_gemm(handle, options, a, b, c);
    ++iterations;
  }
}

void run_strided_batched_active_window(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c,
    double seconds,
    std::uint64_t& iterations) {
  if (seconds <= 0.0) {
    return;
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    run_strided_batched_gemm(handle, options, a, b, c);
    ++iterations;
  }
}

struct CorrectnessSmoke {
  bool passed = false;
  double reference = 0.0;
  double observed = 0.0;
  double abs_error = 0.0;
};

float dense_zero_operand_value(
    const Options& options,
    const std::string& operand,
    int k_index,
    int other_index) {
  if (!operand_selected(options, operand)) {
    return 1.0f;
  }
  return should_zero_host(
             k_index,
             other_index,
             options.k,
             options.zero_ratio,
             zero_pattern_id(options.zero_pattern))
             ? 0.0f
             : 1.0f;
}

CorrectnessSmoke run_correctness_smoke(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c) {
  CorrectnessSmoke smoke;
  if (options.sparsity_mode != "dense_zero") {
    return smoke;
  }

  run_gemm(handle, options, a, b, c);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(correctness smoke)");
  check_cuda(cudaMemcpy(&smoke.observed, c, sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy(correctness smoke)");

  for (int kk = 0; kk < options.k; ++kk) {
    smoke.reference += static_cast<double>(dense_zero_operand_value(options, "A", kk, 0)) *
                       static_cast<double>(dense_zero_operand_value(options, "B", kk, 0));
  }
  smoke.abs_error = std::abs(smoke.observed - smoke.reference);
  smoke.passed = smoke.abs_error <= std::max(1.0, std::abs(smoke.reference)) * 1.0e-2;
  return smoke;
}

Result measure_cublas(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c,
    int requested_sm_count,
    bool sm_count_target_applied,
    const CorrectnessSmoke& correctness_smoke) {
  std::uint64_t warmup_iterations = 0;
  run_active_window(handle, options, a, b, c, options.warmup_sec, warmup_iterations);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  if (options.duty_cycle == 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(options.steady_sec));
    Result result = make_base_result(options, requested_sm_count);
    result.actual_elapsed_ms = options.steady_sec * 1000.0;
    result.duty_control_mode = "cpu_windowed_cublas";
    result.spatial_control_mode = "cublas_sm_count_target";
    result.sm_count_target_applied = sm_count_target_applied;
    result.correctness_smoke_passed = correctness_smoke.passed;
    result.correctness_reference = correctness_smoke.reference;
    result.correctness_observed = correctness_smoke.observed;
    result.correctness_abs_error = correctness_smoke.abs_error;
    result.note = "cuBLAS engine preserves the original CPU-controlled active/idle window behavior.";
    if (options.sparsity_mode == "dense_zero") {
      result.note +=
          " dense_zero inserts zero values into dense operands and does not use hardware sparse Tensor Cores.";
    }
    return result;
  }

  EventPair events;
  std::uint64_t iterations = 0;
  const double period_sec = options.period_ms / 1000.0;
  const double active_sec = period_sec * options.duty_cycle;
  const double idle_sec = period_sec - active_sec;
  auto steady_deadline = std::chrono::steady_clock::now() +
                         std::chrono::duration<double>(options.steady_sec);

  check_cuda(cudaEventRecord(events.start()), "cudaEventRecord(start)");
  while (std::chrono::steady_clock::now() < steady_deadline) {
    run_active_window(handle, options, a, b, c, active_sec, iterations);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(active)");
    if (idle_sec > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(idle_sec));
    }
  }
  check_cuda(cudaEventRecord(events.stop()), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(stop)");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
             "cudaEventElapsedTime");

  const double gemm_ops =
      2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
      static_cast<double>(options.k);
  const double active_elapsed_s = std::max(static_cast<double>(elapsed_ms) / 1000.0, 1e-9);
  const double total_ops = gemm_ops * static_cast<double>(iterations);
  Result result = make_base_result(options, requested_sm_count);
  result.active_elapsed_ms = static_cast<double>(elapsed_ms);
  result.iterations = iterations;
  result.active_tflops = total_ops / active_elapsed_s / 1.0e12;
  result.scheduled_tflops = total_ops / options.steady_sec / 1.0e12;
  result.actual_elapsed_ms = static_cast<double>(elapsed_ms);
  result.measured_runtime_ms = static_cast<double>(elapsed_ms);
  result.dense_baseline_tflops = result.scheduled_tflops;
  result.sparse_tflops_or_dense_equivalent_tflops = result.scheduled_tflops;
  result.speedup_vs_dense = 1.0;
  result.duty_control_mode = "cpu_windowed_cublas";
  result.spatial_control_mode = "cublas_sm_count_target";
  result.sm_count_target_applied = sm_count_target_applied;
  result.correctness_smoke_passed = correctness_smoke.passed;
  result.correctness_reference = correctness_smoke.reference;
  result.correctness_observed = correctness_smoke.observed;
  result.correctness_abs_error = correctness_smoke.abs_error;
  result.note =
      "cuBLAS active_sm_fraction is a cuBLAS SM-count target hint; validate actual activity with profiler metrics.";
  if (options.sparsity_mode == "dense_zero") {
    result.note +=
        " dense_zero inserts zero values into dense operands and does not use hardware sparse Tensor Cores; dense MMA instruction count is unchanged.";
  }
  return result;
}

std::vector<std::string> strided_batched_warnings(const Options& options) {
  std::vector<std::string> warnings;
  if (options.batch_count == 1 &&
      (options.m < 256 || options.n < 256 || options.k < 64)) {
    warnings.push_back(
        "batch_count is 1 and the GEMM shape is small; a single tiny GEMM may under-utilize SMs");
  }
  if ((options.m % 8) != 0 || (options.n % 8) != 0 || (options.k % 8) != 0 ||
      options.k < 16) {
    warnings.push_back(
        "M/N/K are not all multiples of 8 or K is less than 16; this tiny/non-aligned GEMM includes tail, padding, or library behavior");
  }
  return warnings;
}

Result measure_cublas_strided_batched(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c,
    int requested_sm_count,
    bool sm_count_target_applied) {
  std::uint64_t warmup_iterations = 0;
  run_strided_batched_active_window(
      handle, options, a, b, c, options.warmup_sec, warmup_iterations);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(strided_batched warmup)");

  if (options.duty_cycle == 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(options.steady_sec));
    Result result = make_base_result(options, requested_sm_count);
    result.actual_elapsed_ms = options.steady_sec * 1000.0;
    result.duty_control_mode = "cpu_windowed_cublas_strided_batched";
    result.spatial_control_mode = "cublas_sm_count_target";
    result.sm_count_target_applied = sm_count_target_applied;
    result.gemm_semantics = "strided_batched_tiny_gemm";
    result.matrix_shape_is_real = true;
    result.expected_use_case = "fills GPU by batching many independent tiny GEMMs";
    result.warnings = strided_batched_warnings(options);
    result.note =
        "cublas_strided_batched uses cublasGemmStridedBatchedEx with BF16 inputs, FP32 output, and Tensor Core eligible BF16 fast compute.";
    return result;
  }

  EventPair events;
  std::uint64_t iterations = 0;
  const double period_sec = options.period_ms / 1000.0;
  const double active_sec = period_sec * options.duty_cycle;
  const double idle_sec = period_sec - active_sec;
  auto steady_deadline = std::chrono::steady_clock::now() +
                         std::chrono::duration<double>(options.steady_sec);

  check_cuda(cudaEventRecord(events.start()), "cudaEventRecord(strided_batched start)");
  while (std::chrono::steady_clock::now() < steady_deadline) {
    run_strided_batched_active_window(handle, options, a, b, c, active_sec, iterations);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(strided_batched active)");
    if (idle_sec > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(idle_sec));
    }
  }
  check_cuda(cudaEventRecord(events.stop()), "cudaEventRecord(strided_batched stop)");
  check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(strided_batched stop)");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
             "cudaEventElapsedTime(strided_batched)");

  const double total_flops_per_call =
      2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
      static_cast<double>(options.k) * static_cast<double>(options.batch_count);
  const double total_ops = total_flops_per_call * static_cast<double>(iterations);
  const double active_elapsed_s = std::max(static_cast<double>(elapsed_ms) / 1000.0, 1e-9);
  Result result = make_base_result(options, requested_sm_count);
  result.active_elapsed_ms = static_cast<double>(elapsed_ms);
  result.iterations = iterations;
  result.active_tflops = total_ops / active_elapsed_s / 1.0e12;
  result.scheduled_tflops = total_ops / options.steady_sec / 1.0e12;
  result.actual_elapsed_ms = static_cast<double>(elapsed_ms);
  result.measured_runtime_ms = static_cast<double>(elapsed_ms);
  result.dense_baseline_tflops = result.scheduled_tflops;
  result.sparse_tflops_or_dense_equivalent_tflops = result.scheduled_tflops;
  result.speedup_vs_dense = 1.0;
  result.duty_control_mode = "cpu_windowed_cublas_strided_batched";
  result.spatial_control_mode = "cublas_sm_count_target";
  result.sm_count_target_applied = sm_count_target_applied;
  result.gemm_semantics = "strided_batched_tiny_gemm";
  result.matrix_shape_is_real = true;
  result.expected_use_case = "fills GPU by batching many independent tiny GEMMs";
  result.warnings = strided_batched_warnings(options);
  result.note =
      "cublas_strided_batched uses cublasGemmStridedBatchedEx with BF16 inputs, FP32 output, and Tensor Core eligible BF16 fast compute. Use batch_count to fill the GPU with many independent tiny GEMMs.";
  return result;
}

#if defined(HAVE_CUSPARSELT)
struct SparseCusparseLtState {
  cusparseLtHandle_t handle{};
  cusparseLtMatDescriptor_t mat_a{};
  cusparseLtMatDescriptor_t mat_b{};
  cusparseLtMatDescriptor_t mat_c{};
  cusparseLtMatDescriptor_t mat_d{};
  cusparseLtMatmulDescriptor_t matmul{};
  cusparseLtMatmulAlgSelection_t alg_selection{};
  cusparseLtMatmulPlan_t plan{};
  cudaStream_t stream{};
  void* workspace = nullptr;
  void* compressed_a = nullptr;
  void* compressed_buffer = nullptr;
  __nv_bfloat16* pruned_a = nullptr;
  __nv_bfloat16* sparse_c = nullptr;
  int* d_valid = nullptr;
  std::size_t workspace_size = 0;
  std::size_t compressed_size = 0;
  std::size_t compressed_buffer_size = 0;
  bool handle_initialized = false;
  bool stream_initialized = false;
  bool mat_a_initialized = false;
  bool mat_b_initialized = false;
  bool mat_c_initialized = false;
  bool mat_d_initialized = false;
  bool plan_initialized = false;
};

void destroy_sparse_cusparselt_state(SparseCusparseLtState& state) {
  cudaFree(state.workspace);
  cudaFree(state.compressed_a);
  cudaFree(state.compressed_buffer);
  cudaFree(state.pruned_a);
  cudaFree(state.sparse_c);
  cudaFree(state.d_valid);
  if (state.stream_initialized) {
    cudaStreamDestroy(state.stream);
  }
  if (state.plan_initialized) {
    cusparseLtMatmulPlanDestroy(&state.plan);
  }
  if (state.mat_a_initialized) {
    cusparseLtMatDescriptorDestroy(&state.mat_a);
  }
  if (state.mat_b_initialized) {
    cusparseLtMatDescriptorDestroy(&state.mat_b);
  }
  if (state.mat_c_initialized) {
    cusparseLtMatDescriptorDestroy(&state.mat_c);
  }
  if (state.mat_d_initialized) {
    cusparseLtMatDescriptorDestroy(&state.mat_d);
  }
  if (state.handle_initialized) {
    cusparseLtDestroy(&state.handle);
  }
}

SparseCusparseLtState prepare_sparse_cusparselt_state(
    const Options& options,
    const __nv_bfloat16* dense_a,
    const __nv_bfloat16* dense_b) {
  (void)dense_b;
  SparseCusparseLtState state;
  try {
    if (options.m % 16 != 0 || options.n % 16 != 0 || options.k % 16 != 0) {
      throw std::runtime_error(
          "structured_2to4 BF16 cuSPARSELt path requires m, n, and k to be multiples of 16");
    }

    check_cuda(cudaStreamCreate(&state.stream), "cudaStreamCreate(cusparselt)");
    state.stream_initialized = true;
    check_cusparselt(cusparseLtInit(&state.handle), "cusparseLtInit");
    state.handle_initialized = true;

    constexpr int kAlignment = 16;
    check_cusparselt(
        cusparseLtStructuredDescriptorInit(
            &state.handle,
            &state.mat_a,
            options.m,
            options.k,
            options.m,
            kAlignment,
            CUDA_R_16BF,
            CUSPARSE_ORDER_COL,
            CUSPARSELT_SPARSITY_50_PERCENT),
        "cusparseLtStructuredDescriptorInit(A)");
    state.mat_a_initialized = true;
    check_cusparselt(
        cusparseLtDenseDescriptorInit(
            &state.handle,
            &state.mat_b,
            options.k,
            options.n,
            options.k,
            kAlignment,
            CUDA_R_16BF,
            CUSPARSE_ORDER_COL),
        "cusparseLtDenseDescriptorInit(B)");
    state.mat_b_initialized = true;
    check_cusparselt(
        cusparseLtDenseDescriptorInit(
            &state.handle,
            &state.mat_c,
            options.m,
            options.n,
            options.m,
            kAlignment,
            CUDA_R_16BF,
            CUSPARSE_ORDER_COL),
        "cusparseLtDenseDescriptorInit(C)");
    state.mat_c_initialized = true;
    check_cusparselt(
        cusparseLtDenseDescriptorInit(
            &state.handle,
            &state.mat_d,
            options.m,
            options.n,
            options.m,
            kAlignment,
            CUDA_R_16BF,
            CUSPARSE_ORDER_COL),
        "cusparseLtDenseDescriptorInit(D)");
    state.mat_d_initialized = true;
    check_cusparselt(
        cusparseLtMatmulDescriptorInit(
            &state.handle,
            &state.matmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &state.mat_a,
            &state.mat_b,
            &state.mat_c,
            &state.mat_d,
            CUSPARSE_COMPUTE_32F),
        "cusparseLtMatmulDescriptorInit");
    check_cusparselt(
        cusparseLtMatmulAlgSelectionInit(
            &state.handle,
            &state.alg_selection,
            &state.matmul,
            CUSPARSELT_MATMUL_ALG_DEFAULT),
        "cusparseLtMatmulAlgSelectionInit");
    check_cusparselt(
        cusparseLtMatmulPlanInit(&state.handle, &state.plan, &state.matmul, &state.alg_selection),
        "cusparseLtMatmulPlanInit");
    state.plan_initialized = true;
    check_cusparselt(
        cusparseLtMatmulGetWorkspace(&state.handle, &state.plan, &state.workspace_size),
        "cusparseLtMatmulGetWorkspace");
    check_cusparselt(
        cusparseLtSpMMACompressedSize(
            &state.handle,
            &state.plan,
            &state.compressed_size,
            &state.compressed_buffer_size),
        "cusparseLtSpMMACompressedSize");

    if (state.workspace_size > 0) {
      check_cuda(cudaMalloc(&state.workspace, state.workspace_size), "cudaMalloc(workspace)");
    }
    check_cuda(cudaMalloc(&state.compressed_a, state.compressed_size),
               "cudaMalloc(compressed_a)");
    if (state.compressed_buffer_size > 0) {
      check_cuda(cudaMalloc(&state.compressed_buffer, state.compressed_buffer_size),
                 "cudaMalloc(compressed_buffer)");
    }

    const std::size_t a_count = static_cast<std::size_t>(options.m) * options.k;
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&state.pruned_a), a_count * sizeof(__nv_bfloat16)),
        "cudaMalloc(pruned_a)");
    const std::size_t c_count = static_cast<std::size_t>(options.m) * options.n;
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&state.sparse_c), c_count * sizeof(__nv_bfloat16)),
        "cudaMalloc(sparse_c)");
    check_cuda(cudaMemsetAsync(
                   state.sparse_c,
                   0,
                   c_count * sizeof(__nv_bfloat16),
                   state.stream),
               "cudaMemsetAsync(sparse_c)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&state.d_valid), sizeof(int)),
               "cudaMalloc(d_valid)");

    check_cusparselt(
        cusparseLtSpMMAPrune(
            &state.handle,
            &state.matmul,
            dense_a,
            state.pruned_a,
            CUSPARSELT_PRUNE_SPMMA_TILE,
            state.stream),
        "cusparseLtSpMMAPrune(A)");
    check_cusparselt(
        cusparseLtSpMMAPruneCheck(
            &state.handle,
            &state.matmul,
            state.pruned_a,
            state.d_valid,
            state.stream),
        "cusparseLtSpMMAPruneCheck(A)");
    int h_valid = 0;
    check_cuda(
        cudaMemcpyAsync(&h_valid, state.d_valid, sizeof(int), cudaMemcpyDeviceToHost, state.stream),
        "cudaMemcpyAsync(prune check)");
    check_cuda(cudaStreamSynchronize(state.stream), "cudaStreamSynchronize(prune)");
    if (h_valid != 0) {
      throw std::runtime_error("cuSPARSELt prune check failed for A 2:4 pattern");
    }

    check_cusparselt(
        cusparseLtSpMMACompress(
            &state.handle,
            &state.plan,
            state.pruned_a,
            state.compressed_a,
            state.compressed_buffer,
            state.stream),
        "cusparseLtSpMMACompress(A)");
    check_cuda(cudaStreamSynchronize(state.stream), "cudaStreamSynchronize(compress)");
    return state;
  } catch (...) {
    destroy_sparse_cusparselt_state(state);
    throw;
  }
}

void run_sparse_cusparselt_gemm(
    SparseCusparseLtState& state,
    const __nv_bfloat16* b) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaStream_t streams[] = {state.stream};
  check_cusparselt(
      cusparseLtMatmul(
          &state.handle,
          &state.plan,
          &alpha,
          state.compressed_a,
          b,
          &beta,
          state.sparse_c,
          state.sparse_c,
          state.workspace,
          streams,
          1),
      "cusparseLtMatmul");
}

void run_sparse_active_window(
    SparseCusparseLtState& state,
    const Options& options,
    const __nv_bfloat16* b,
    double seconds,
    std::uint64_t& iterations) {
  if (seconds <= 0.0) {
    return;
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    run_sparse_cusparselt_gemm(state, b);
    ++iterations;
  }
}

double measure_dense_baseline_tflops(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* pruned_a,
    const __nv_bfloat16* b,
    float* c) {
  constexpr int kBaselineIterations = 3;
  run_gemm(handle, options, pruned_a, b, c);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(dense baseline warmup)");

  EventPair events;
  check_cuda(cudaEventRecord(events.start()), "cudaEventRecord(dense baseline start)");
  for (int index = 0; index < kBaselineIterations; ++index) {
    run_gemm(handle, options, pruned_a, b, c);
  }
  check_cuda(cudaEventRecord(events.stop()), "cudaEventRecord(dense baseline stop)");
  check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(dense baseline stop)");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
             "cudaEventElapsedTime(dense baseline)");
  const double dense_ops =
      2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
      static_cast<double>(options.k) * static_cast<double>(kBaselineIterations);
  return dense_ops / std::max(static_cast<double>(elapsed_ms) / 1000.0, 1.0e-9) /
         1.0e12;
}

Result measure_cusparselt_sparse(
    cublasHandle_t dense_handle,
    const Options& options,
    const __nv_bfloat16* dense_a,
    const __nv_bfloat16* b,
    float* c,
    int requested_sm_count,
    bool sm_count_target_applied) {
  SparseCusparseLtState sparse_state =
      prepare_sparse_cusparselt_state(options, dense_a, b);
  try {
    const double dense_baseline_tflops =
        measure_dense_baseline_tflops(dense_handle, options, sparse_state.pruned_a, b, c);

    std::uint64_t warmup_iterations = 0;
    run_sparse_active_window(
        sparse_state, options, b, options.warmup_sec, warmup_iterations);
    check_cuda(cudaStreamSynchronize(sparse_state.stream),
               "cudaStreamSynchronize(sparse warmup)");

    EventPair events;
    std::uint64_t iterations = 0;
    const double period_sec = options.period_ms / 1000.0;
    const double active_sec = period_sec * options.duty_cycle;
    const double idle_sec = period_sec - active_sec;
    auto steady_deadline = std::chrono::steady_clock::now() +
                           std::chrono::duration<double>(options.steady_sec);

    check_cuda(cudaEventRecord(events.start(), sparse_state.stream),
               "cudaEventRecord(sparse start)");
    if (options.duty_cycle == 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(options.steady_sec));
    } else {
      while (std::chrono::steady_clock::now() < steady_deadline) {
        run_sparse_active_window(sparse_state, options, b, active_sec, iterations);
        check_cuda(cudaStreamSynchronize(sparse_state.stream),
                   "cudaStreamSynchronize(sparse active)");
        if (idle_sec > 0.0) {
          std::this_thread::sleep_for(std::chrono::duration<double>(idle_sec));
        }
      }
    }
    check_cuda(cudaEventRecord(events.stop(), sparse_state.stream),
               "cudaEventRecord(sparse stop)");
    check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(sparse stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
               "cudaEventElapsedTime(sparse)");

    const double dense_equivalent_ops =
        2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) *
        static_cast<double>(options.k) * static_cast<double>(iterations);
    Result result = make_base_result(options, requested_sm_count);
    result.uses_sparse_tensor_core = (iterations > 0 || warmup_iterations > 0);
    result.dense_mma_instruction_count_unchanged = false;
    result.active_elapsed_ms = static_cast<double>(elapsed_ms);
    result.actual_elapsed_ms = static_cast<double>(elapsed_ms);
    result.measured_runtime_ms = static_cast<double>(elapsed_ms);
    result.iterations = iterations;
    result.active_tflops =
        dense_equivalent_ops / std::max(static_cast<double>(elapsed_ms) / 1000.0, 1.0e-9) /
        1.0e12;
    result.scheduled_tflops = dense_equivalent_ops / options.steady_sec / 1.0e12;
    result.dense_baseline_tflops = dense_baseline_tflops;
    result.sparse_tflops_or_dense_equivalent_tflops = result.scheduled_tflops;
    result.speedup_vs_dense =
        dense_baseline_tflops > 0.0 ? result.sparse_tflops_or_dense_equivalent_tflops /
                                          dense_baseline_tflops
                                    : 0.0;
    result.duty_control_mode = "cpu_windowed_cusparselt";
    result.spatial_control_mode = "sparse_cusparselt_full_requested_coverage";
    result.sm_count_target_applied = sm_count_target_applied;
    result.compression_time_excluded = true;
    result.setup_time_excluded = true;
    result.note =
        "structured_2to4 uses cuSPARSELt SpMMA with operand A pruned and compressed before the steady sparse window; compression/setup time is excluded from sparse measured_runtime_ms. Dense baseline uses the expanded pruned A values and the same B values. Validate sparse Tensor Core kernels with Nsight Systems/Compute.";
    destroy_sparse_cusparselt_state(sparse_state);
    return result;
  } catch (...) {
    destroy_sparse_cusparselt_state(sparse_state);
    throw;
  }
}
#endif

// Accumulator count is compile-time specialized so variants with 1, 2, or 4
// accumulators do not pay register pressure for unused fragments.
template <int AccumulatorsPerWarp>
__global__ void wmma_persistent_bf16_kernel(
    float* output,
    unsigned long long* iterations,
    unsigned long long duration_cycles,
    unsigned long long period_cycles,
    unsigned long long active_cycles,
    int mma_iters_per_loop,
    int atomic_period) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kTileK = 16;
  constexpr int kWarpsPerBlock = 4;

  const unsigned long long start = clock64();
  const int warp_id = threadIdx.x / 32;
  unsigned long long pending_iterations = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float>
      c_frags[AccumulatorsPerWarp];

  wmma::fill_fragment(a_frag, __float2bfloat16(1.0f));
  wmma::fill_fragment(b_frag, __float2bfloat16(1.0f));
#pragma unroll
  for (int acc = 0; acc < AccumulatorsPerWarp; ++acc) {
    wmma::fill_fragment(c_frags[acc], 0.0f);
  }
#endif

  while (true) {
    unsigned long long elapsed = clock64() - start;
    if (elapsed >= duration_cycles) {
      break;
    }

    const bool active =
        active_cycles > 0 &&
        (active_cycles >= period_cycles || (elapsed % period_cycles) < active_cycles);

    if (active) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      for (int i = 0; i < mma_iters_per_loop; ++i) {
#pragma unroll
        for (int acc = 0; acc < AccumulatorsPerWarp; ++acc) {
          wmma::mma_sync(c_frags[acc], a_frag, b_frag, c_frags[acc]);
        }
      }
      if ((threadIdx.x % 32) == 0) {
        pending_iterations += static_cast<unsigned long long>(mma_iters_per_loop) *
                              static_cast<unsigned long long>(AccumulatorsPerWarp);
        if (pending_iterations >= static_cast<unsigned long long>(atomic_period)) {
          atomicAdd(iterations, pending_iterations);
          pending_iterations = 0;
        }
      }
#else
      if ((threadIdx.x % 32) == 0) {
        pending_iterations += static_cast<unsigned long long>(AccumulatorsPerWarp);
        if (pending_iterations >= static_cast<unsigned long long>(atomic_period)) {
          atomicAdd(iterations, pending_iterations);
          pending_iterations = 0;
        }
      }
#endif
    } else {
      __nanosleep(1000);
    }
  }

  if ((threadIdx.x % 32) == 0 && pending_iterations > 0) {
    atomicAdd(iterations, pending_iterations);
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#pragma unroll
  for (int acc = 0; acc < AccumulatorsPerWarp; ++acc) {
    wmma::store_matrix_sync(
        output +
            ((static_cast<unsigned int>(blockIdx.x) * kWarpsPerBlock +
              static_cast<unsigned int>(warp_id)) *
                 AccumulatorsPerWarp +
             static_cast<unsigned int>(acc)) *
                kTileM * kTileN,
        c_frags[acc],
        kTileN,
        wmma::mem_row_major);
  }
#else
  if (threadIdx.x == 0) {
    output[blockIdx.x] = 0.0f;
  }
#endif
}

unsigned long long cycles_for_ms(double ms, int clock_rate_khz) {
  const double cycles = std::max(1.0, ms * static_cast<double>(clock_rate_khz));
  return static_cast<unsigned long long>(cycles);
}

int get_device_clock_rate_khz(int device) {
  int clock_rate_khz = 0;
  check_cuda(cudaDeviceGetAttribute(
                 &clock_rate_khz,
                 cudaDevAttrClockRate,
                 device),
             "cudaDeviceGetAttribute(cudaDevAttrClockRate)");
  if (clock_rate_khz <= 0) {
    throw std::runtime_error("Invalid cudaDevAttrClockRate");
  }
  return clock_rate_khz;
}

template <int AccumulatorsPerWarp>
int query_wmma_occupancy_max_active_blocks_per_sm() {
  int max_active_blocks = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &max_active_blocks,
                 wmma_persistent_bf16_kernel<AccumulatorsPerWarp>,
                 kWmmaThreadsPerBlock,
                 kWmmaDynamicSmemBytes),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor(wmma_persistent_bf16_kernel)");
  return max_active_blocks;
}

int query_wmma_occupancy_dispatch(int accumulators_per_warp) {
  switch (accumulators_per_warp) {
    case 1:
      return query_wmma_occupancy_max_active_blocks_per_sm<1>();
    case 2:
      return query_wmma_occupancy_max_active_blocks_per_sm<2>();
    case 4:
      return query_wmma_occupancy_max_active_blocks_per_sm<4>();
    case 8:
      return query_wmma_occupancy_max_active_blocks_per_sm<8>();
    default:
      throw std::runtime_error("--accumulators-per-warp must be one of: 1, 2, 4, 8");
  }
}

template <int AccumulatorsPerWarp>
void launch_wmma_window(
    const Options& options,
    int clock_rate_khz,
    float* output,
    unsigned long long* iteration_counter,
    int grid_blocks,
    double seconds,
    bool reset_counter) {
  if (seconds <= 0.0) {
    return;
  }
  if (reset_counter) {
    check_cuda(cudaMemset(iteration_counter, 0, sizeof(unsigned long long)),
               "cudaMemset(iteration_counter)");
  }

  const unsigned long long duration_cycles =
      cycles_for_ms(seconds * 1000.0, clock_rate_khz);
  const unsigned long long period_cycles =
      cycles_for_ms(options.period_ms, clock_rate_khz);
  const unsigned long long active_cycles =
      static_cast<unsigned long long>(std::floor(period_cycles * options.duty_cycle));

  wmma_persistent_bf16_kernel<AccumulatorsPerWarp><<<grid_blocks, kWmmaThreadsPerBlock>>>(
      output,
      iteration_counter,
      duration_cycles,
      period_cycles,
      active_cycles,
      options.mma_iters_per_loop,
      options.atomic_period);
  check_cuda(cudaGetLastError(), "wmma_persistent_bf16_kernel");
}

void launch_wmma_window_dispatch(
    const Options& options,
    int clock_rate_khz,
    float* output,
    unsigned long long* iteration_counter,
    int grid_blocks,
    double seconds,
    bool reset_counter) {
  switch (options.accumulators_per_warp) {
    case 1:
      launch_wmma_window<1>(
          options, clock_rate_khz, output, iteration_counter, grid_blocks, seconds, reset_counter);
      return;
    case 2:
      launch_wmma_window<2>(
          options, clock_rate_khz, output, iteration_counter, grid_blocks, seconds, reset_counter);
      return;
    case 4:
      launch_wmma_window<4>(
          options, clock_rate_khz, output, iteration_counter, grid_blocks, seconds, reset_counter);
      return;
    case 8:
      launch_wmma_window<8>(
          options, clock_rate_khz, output, iteration_counter, grid_blocks, seconds, reset_counter);
      return;
    default:
      throw std::runtime_error("--accumulators-per-warp must be one of: 1, 2, 4, 8");
  }
}

Result measure_wmma_persistent(
    const Options& options,
    const cudaDeviceProp& prop,
    int requested_sm_count) {
  if (prop.major < 8) {
    throw std::runtime_error("engine=wmma_persistent requires SM80 or newer for BF16 WMMA");
  }

  const int clock_rate_khz = get_device_clock_rate_khz(options.device);
  const int blocks_per_sm = options.blocks_per_sm;
  const int grid_blocks = requested_sm_count * blocks_per_sm;
  const int occupancy_max_active_blocks_per_sm =
      query_wmma_occupancy_dispatch(options.accumulators_per_warp);
  const int effective_blocks_per_sm_estimate =
      std::min(blocks_per_sm, occupancy_max_active_blocks_per_sm);
  const bool occupancy_limited = blocks_per_sm > occupancy_max_active_blocks_per_sm;
  constexpr int warps_per_block = 4;
  const int output_values_per_block = warps_per_block * options.accumulators_per_warp * 16 * 16;
  float* output = nullptr;
  unsigned long long* iteration_counter = nullptr;
  unsigned long long host_iterations = 0;

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&output),
                        static_cast<std::size_t>(grid_blocks) * output_values_per_block *
                            sizeof(float)),
             "cudaMalloc(output)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&iteration_counter),
                        sizeof(unsigned long long)),
             "cudaMalloc(iteration_counter)");

  try {
    launch_wmma_window_dispatch(
        options, clock_rate_khz, output, iteration_counter, grid_blocks, options.warmup_sec, true);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(wmma warmup)");

    EventPair events;
    check_cuda(cudaEventRecord(events.start()), "cudaEventRecord(wmma start)");
    launch_wmma_window_dispatch(
        options, clock_rate_khz, output, iteration_counter, grid_blocks, options.steady_sec, true);
    check_cuda(cudaEventRecord(events.stop()), "cudaEventRecord(wmma stop)");
    check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(wmma stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
               "cudaEventElapsedTime(wmma)");
    check_cuda(cudaMemcpy(&host_iterations,
                          iteration_counter,
                          sizeof(host_iterations),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(iteration_counter)");

    const double ops_per_mma = 2.0 * 16.0 * 16.0 * 16.0;
    const double total_ops = ops_per_mma * static_cast<double>(host_iterations);
    const double active_elapsed_ms =
        std::max(static_cast<double>(elapsed_ms) * options.duty_cycle, 0.0);
    const double active_elapsed_s = std::max(active_elapsed_ms / 1000.0, 1e-9);

    cudaFree(output);
    cudaFree(iteration_counter);

    Result result = make_base_result(options, requested_sm_count);
    result.active_elapsed_ms = active_elapsed_ms;
    result.iterations = host_iterations;
    result.active_tflops = total_ops / active_elapsed_s / 1.0e12;
    result.scheduled_tflops = total_ops / options.steady_sec / 1.0e12;
    result.actual_elapsed_ms = static_cast<double>(elapsed_ms);
    result.measured_runtime_ms = static_cast<double>(elapsed_ms);
    result.sparse_tflops_or_dense_equivalent_tflops = result.scheduled_tflops;
    result.duty_control_mode = "device_clock64_persistent_kernel";
    result.spatial_control_mode = "persistent_cta_count";
    result.grid_blocks = grid_blocks;
    result.blocks_per_sm = blocks_per_sm;
    result.mma_iters_per_loop = options.mma_iters_per_loop;
    result.accumulators_per_warp = options.accumulators_per_warp;
    result.atomic_period = options.atomic_period;
    result.occupancy_max_active_blocks_per_sm = occupancy_max_active_blocks_per_sm;
    result.effective_blocks_per_sm_estimate = effective_blocks_per_sm_estimate;
    result.occupancy_limited = occupancy_limited;
    result.note =
        "wmma_persistent uses m/n/k as nominal reporting parameters; actual MAC pressure is controlled by blocks_per_sm, mma_iters_per_loop, accumulators_per_warp, atomic_period, active_sm_fraction, duty_cycle, and period_ms. blocks_per_sm is requested launch density; actual resident CTA count is bounded by occupancy. Validate Tensor Core utilization with Nsight profiler metrics.";
    return result;
  } catch (...) {
    cudaFree(output);
    cudaFree(iteration_counter);
    throw;
  }
}

#if defined(HAVE_CUTLASS_CUTE)
__global__ void cutlass_tile_burn_kernel(
    float* output,
    unsigned long long* iterations,
    unsigned long long duration_cycles,
    unsigned long long period_cycles,
    unsigned long long active_cycles,
    int synthetic_mma_ops_per_loop,
    int atomic_period) {
  constexpr int kWarpsPerBlock = 4;
  constexpr std::uint32_t kBf16OnePair = 0x3f803f80U;

  // The prototype identifies the middle-level CuTe TiledMMA/atom shape here,
  // then calls the SM80 BF16 atom directly in the tight loop to avoid global
  // A/B traffic and top-level device::Gemm scheduling.
  using CuteMmaAtom = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  [[maybe_unused]] auto tiled_mma = cute::make_tiled_mma(CuteMmaAtom{});

  const unsigned long long start = clock64();
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  float c0 = 0.0f;
  float c1 = 0.0f;
  float c2 = 0.0f;
  float c3 = 0.0f;
  unsigned long long pending_iterations = 0;

  while (true) {
    unsigned long long elapsed = clock64() - start;
    if (elapsed >= duration_cycles) {
      break;
    }

    const bool active =
        active_cycles > 0 &&
        (active_cycles >= period_cycles || (elapsed % period_cycles) < active_cycles);
    if (active) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      for (int op = 0; op < synthetic_mma_ops_per_loop; ++op) {
        CuteMmaAtom::fma(
            c0,
            c1,
            c2,
            c3,
            kBf16OnePair,
            kBf16OnePair,
            kBf16OnePair,
            kBf16OnePair,
            kBf16OnePair,
            kBf16OnePair,
            c0,
            c1,
            c2,
            c3);
      }
#endif
      if (lane_id == 0) {
        pending_iterations += static_cast<unsigned long long>(synthetic_mma_ops_per_loop);
        if (pending_iterations >= static_cast<unsigned long long>(atomic_period)) {
          atomicAdd(iterations, pending_iterations);
          pending_iterations = 0;
        }
      }
    } else {
      __nanosleep(1000);
    }
  }

  if (lane_id == 0) {
    if (pending_iterations > 0) {
      atomicAdd(iterations, pending_iterations);
    }
    std::size_t base =
        (static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp_id) * 4;
    output[base + 0] = c0;
    output[base + 1] = c1;
    output[base + 2] = c2;
    output[base + 3] = c3;
  }
}

int query_cutlass_tile_burn_occupancy() {
  int blocks_per_sm = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks_per_sm, cutlass_tile_burn_kernel, kWmmaThreadsPerBlock, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor(cutlass_tile_burn)");
  return blocks_per_sm;
}

void launch_cutlass_tile_burn_window(
    const Options& options,
    int clock_rate_khz,
    float* output,
    unsigned long long* iteration_counter,
    int grid_blocks,
    int synthetic_mma_ops_per_loop,
    double seconds,
    bool reset_counter) {
  if (reset_counter) {
    check_cuda(cudaMemset(iteration_counter, 0, sizeof(unsigned long long)),
               "cudaMemset(cutlass iteration_counter)");
  }
  if (seconds <= 0.0) {
    return;
  }
  unsigned long long duration_cycles =
      static_cast<unsigned long long>(seconds * static_cast<double>(clock_rate_khz) * 1000.0);
  unsigned long long period_cycles =
      static_cast<unsigned long long>((options.period_ms / 1000.0) *
                                      static_cast<double>(clock_rate_khz) * 1000.0);
  period_cycles = std::max<unsigned long long>(period_cycles, 1);
  unsigned long long active_cycles =
      static_cast<unsigned long long>(static_cast<double>(period_cycles) * options.duty_cycle);

  cutlass_tile_burn_kernel<<<grid_blocks, kWmmaThreadsPerBlock>>>(
      output,
      iteration_counter,
      duration_cycles,
      period_cycles,
      active_cycles,
      synthetic_mma_ops_per_loop,
      options.atomic_period);
  check_cuda(cudaGetLastError(), "cutlass_tile_burn_kernel");
}

Result measure_cutlass_tile_burn(
    const Options& options,
    const cudaDeviceProp& prop,
    int requested_sm_count) {
  if (prop.major < 8) {
    throw std::runtime_error("engine=cutlass_tile_burn requires SM80 or newer");
  }
  if (options.sparsity_mode != "none") {
    throw std::runtime_error("engine=cutlass_tile_burn supports --sparsity-mode none only");
  }
  constexpr int kAtomM = 16;
  constexpr int kAtomN = 8;
  constexpr int kAtomK = 16;
  if (options.cutlass_tile_m % kAtomM != 0 || options.cutlass_tile_n % kAtomN != 0 ||
      options.cutlass_tile_k % kAtomK != 0) {
    throw std::runtime_error(
        "cutlass_tile_burn prototype requires tile M/N/K to be multiples of 16/8/16");
  }

  const int synthetic_m = options.synthetic_m > 0 ? options.synthetic_m : options.m;
  const int synthetic_n = options.synthetic_n > 0 ? options.synthetic_n : options.n;
  const int synthetic_k = options.synthetic_k > 0 ? options.synthetic_k : options.k;
  const int synthetic_m_tiles = ceil_div_int(synthetic_m, options.cutlass_tile_m);
  const int synthetic_n_tiles = ceil_div_int(synthetic_n, options.cutlass_tile_n);
  const int synthetic_k_tiles = ceil_div_int(synthetic_k, options.cutlass_tile_k);
  const std::uint64_t atom_ops_per_cutlass_tile =
      static_cast<std::uint64_t>(options.cutlass_tile_m / kAtomM) *
      static_cast<std::uint64_t>(options.cutlass_tile_n / kAtomN) *
      static_cast<std::uint64_t>(options.cutlass_tile_k / kAtomK);
  const std::uint64_t synthetic_tile_ops =
      static_cast<std::uint64_t>(synthetic_m_tiles) *
      static_cast<std::uint64_t>(synthetic_n_tiles) *
      static_cast<std::uint64_t>(synthetic_k_tiles) * atom_ops_per_cutlass_tile;

  const int clock_rate_khz = get_device_clock_rate_khz(options.device);
  const int blocks_per_sm = options.blocks_per_sm;
  const int grid_blocks = requested_sm_count * blocks_per_sm;
  const int occupancy_max_active_blocks_per_sm = query_cutlass_tile_burn_occupancy();
  const int effective_blocks_per_sm_estimate =
      std::min(blocks_per_sm, occupancy_max_active_blocks_per_sm);
  const bool occupancy_limited = blocks_per_sm > occupancy_max_active_blocks_per_sm;
  const std::uint64_t distributed_ops =
      std::max<std::uint64_t>(1, (synthetic_tile_ops + grid_blocks - 1) / grid_blocks);
  const std::uint64_t synthetic_mma_ops_cap =
      static_cast<std::uint64_t>(options.mma_iters_per_loop);
  const bool auto_synthetic_mma_ops_cap_applied = distributed_ops > synthetic_mma_ops_cap;
  bool synthetic_mma_ops_cap_applied = auto_synthetic_mma_ops_cap_applied;
  int synthetic_mma_ops_per_loop = static_cast<int>(std::min<std::uint64_t>(
      std::max<std::uint64_t>(1, distributed_ops),
      synthetic_mma_ops_cap));
  std::string synthetic_mma_ops_per_loop_source = "auto_distributed";
  if (options.synthetic_mma_ops_per_loop_override > 0) {
    synthetic_mma_ops_per_loop = options.synthetic_mma_ops_per_loop_override;
    synthetic_mma_ops_per_loop_source = "manual_override";
    synthetic_mma_ops_cap_applied = false;
  }

  constexpr int kWarpsPerBlock = 4;
  constexpr int kValuesPerWarp = 4;
  float* output = nullptr;
  unsigned long long* iteration_counter = nullptr;
  unsigned long long host_iterations = 0;
  const std::size_t output_values =
      static_cast<std::size_t>(grid_blocks) * kWarpsPerBlock * kValuesPerWarp;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&output), output_values * sizeof(float)),
             "cudaMalloc(cutlass output)");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&iteration_counter),
                        sizeof(unsigned long long)),
             "cudaMalloc(cutlass iteration_counter)");

  try {
    launch_cutlass_tile_burn_window(
        options,
        clock_rate_khz,
        output,
        iteration_counter,
        grid_blocks,
        synthetic_mma_ops_per_loop,
        options.warmup_sec,
        true);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cutlass warmup)");

    EventPair events;
    check_cuda(cudaEventRecord(events.start()), "cudaEventRecord(cutlass start)");
    launch_cutlass_tile_burn_window(
        options,
        clock_rate_khz,
        output,
        iteration_counter,
        grid_blocks,
        synthetic_mma_ops_per_loop,
        options.steady_sec,
        true);
    check_cuda(cudaEventRecord(events.stop()), "cudaEventRecord(cutlass stop)");
    check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(cutlass stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
               "cudaEventElapsedTime(cutlass)");
    check_cuda(cudaMemcpy(&host_iterations,
                          iteration_counter,
                          sizeof(host_iterations),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(cutlass iteration_counter)");

    const double ops_per_atom = 2.0 * static_cast<double>(kAtomM) *
                                static_cast<double>(kAtomN) *
                                static_cast<double>(kAtomK);
    const double total_ops = ops_per_atom * static_cast<double>(host_iterations);
    const double active_elapsed_ms =
        std::max(static_cast<double>(elapsed_ms) * options.duty_cycle, 0.0);
    const double active_elapsed_s = std::max(active_elapsed_ms / 1000.0, 1.0e-9);

    cudaFree(output);
    cudaFree(iteration_counter);

    Result result = make_base_result(options, requested_sm_count);
    result.active_elapsed_ms = active_elapsed_ms;
    result.iterations = host_iterations;
    result.active_tflops = total_ops / active_elapsed_s / 1.0e12;
    result.scheduled_tflops = total_ops / options.steady_sec / 1.0e12;
    result.actual_elapsed_ms = static_cast<double>(elapsed_ms);
    result.measured_runtime_ms = static_cast<double>(elapsed_ms);
    result.sparse_tflops_or_dense_equivalent_tflops = result.scheduled_tflops;
    result.duty_control_mode = "device_clock64_persistent_cutlass_tile_kernel";
    result.spatial_control_mode = "persistent_cta_count";
    result.grid_blocks = grid_blocks;
    result.blocks_per_sm = blocks_per_sm;
    result.mma_iters_per_loop = options.mma_iters_per_loop;
    result.atomic_period = options.atomic_period;
    result.occupancy_max_active_blocks_per_sm = occupancy_max_active_blocks_per_sm;
    result.effective_blocks_per_sm_estimate = effective_blocks_per_sm_estimate;
    result.occupancy_limited = occupancy_limited;
    result.matrix_shape_is_real = false;
    result.synthetic_mnk_controls_mma_count = true;
    result.memory_traffic_minimized = true;
    result.uses_global_ab = "false";
    result.uses_shared_memory_tiles = false;
    result.cutlass_tile_m = options.cutlass_tile_m;
    result.cutlass_tile_n = options.cutlass_tile_n;
    result.cutlass_tile_k = options.cutlass_tile_k;
    result.synthetic_m_tiles = synthetic_m_tiles;
    result.synthetic_n_tiles = synthetic_n_tiles;
    result.synthetic_k_tiles = synthetic_k_tiles;
    result.synthetic_mma_ops_per_loop = synthetic_mma_ops_per_loop;
    result.synthetic_mma_ops_per_loop_source = synthetic_mma_ops_per_loop_source;
    result.synthetic_mma_ops_per_loop_override =
        options.synthetic_mma_ops_per_loop_override;
    result.requested_synthetic_tile_ops = synthetic_tile_ops;
    result.distributed_synthetic_ops_per_block = distributed_ops;
    result.synthetic_mma_ops_cap = synthetic_mma_ops_cap;
    result.synthetic_mma_ops_cap_applied = synthetic_mma_ops_cap_applied;
    result.cutlass_atom_shape_m = kAtomM;
    result.cutlass_atom_shape_n = kAtomN;
    result.cutlass_atom_shape_k = kAtomK;
    result.cutlass_atom_arch = "SM80";
    result.uses_cutlass_tiled_mma_object = true;
    result.uses_cutlass_mma_atom_direct = true;
    result.initial_global_load_bytes = 0;
    result.steady_global_load_bytes_per_loop = 0;
    result.final_global_store_bytes = output_values * sizeof(float);
    std::ostringstream note;
    note
        << "cutlass_tile_burn is an experimental CUTLASS/CuTe MMA atom based synthetic Tensor Core burn. It uses a CUTLASS/CuTe SM80 BF16 MMA atom directly in a persistent clock64-controlled kernel, does not use top-level CUTLASS device::Gemm, does not load real A/B matrices, does not use shared-memory A/B tiles, and does not run a real full GEMM shape benchmark. cutlass_tile_m/n/k define synthetic atom grouping, not real tile-local GEMM storage. Current version is register-constant atom burn; a later implementation may add a true tile-local shared-memory CUTLASS/CuTe burn that loads A/B tiles once per CTA and reuses them.";
    if (synthetic_mma_ops_per_loop_source == "manual_override") {
      note << " synthetic_mma_ops_per_loop_source=manual_override, so "
           << "--synthetic-mma-ops-per-loop directly controls the actual number of "
           << "CuteMmaAtom::fma calls per active loop.";
    } else if (synthetic_mma_ops_cap_applied) {
      note << " synthetic_mma_ops_per_loop_source=auto_distributed and "
           << "synthetic_mma_ops_cap_applied=true because the requested synthetic M/N/K maps to "
           << distributed_ops << " atom operations per block per loop, exceeding the cap of "
           << synthetic_mma_ops_cap
           << ". Raise --mma-iters-per-loop if you want the synthetic M/N/K to issue more atoms per loop. "
           << "Small synthetic shapes may map to only 1 atom per block per loop after distribution across grid_blocks.";
    } else {
      note << " synthetic_mma_ops_per_loop_source=auto_distributed and "
           << "synthetic_mma_ops_cap_applied=false because the requested synthetic M/N/K maps to "
           << distributed_ops << " atom operations per block per loop within the cap of "
           << synthetic_mma_ops_cap
           << ". Small synthetic shapes may map to only 1 atom per block per loop after distribution across grid_blocks.";
    }
    result.note = note.str();
    return result;
  } catch (...) {
    cudaFree(output);
    cudaFree(iteration_counter);
    throw;
  }
}
#else
Result measure_cutlass_tile_burn(
    const Options&,
    const cudaDeviceProp&,
    int) {
  throw std::runtime_error(
      "engine=cutlass_tile_burn requires CUTLASS/CuTe headers from third_party/cutlass "
      "and a build with HAVE_CUTLASS_CUTE");
}
#endif

#if defined(HAVE_WGMMA_SM90A)
Result measure_wgmma_persistent(const Options& options, int requested_sm_count) {
  gpu_power_validation::WgmmaRunOptions run_options;
  run_options.device = options.device;
  run_options.requested_sm_count = requested_sm_count;
  run_options.blocks_per_sm = options.blocks_per_sm;
  run_options.ops_per_check = options.wgmma_ops_per_check;
  run_options.wait_group = options.wgmma_wait_group;
  run_options.accumulator_sets = options.wgmma_accumulator_sets;
  run_options.warmup_sec = options.warmup_sec;
  run_options.steady_sec = options.steady_sec;

  const gpu_power_validation::WgmmaRunResult run_result =
      gpu_power_validation::run_wgmma_persistent_sm90a(run_options);
  constexpr double kWgmmaFlopsPerOp = 2.0 * 64.0 * 64.0 * 16.0;
  const double total_flops =
      static_cast<double>(run_result.wgmma_ops_executed) * kWgmmaFlopsPerOp;

  Result result = make_base_result(options, requested_sm_count);
  result.active_elapsed_ms = run_result.actual_elapsed_ms;
  result.actual_elapsed_ms = run_result.actual_elapsed_ms;
  result.measured_runtime_ms = run_result.actual_elapsed_ms;
  result.iterations = run_result.wgmma_ops_executed;
  result.active_tflops = run_result.actual_elapsed_ms > 0.0
                              ? total_flops / (run_result.actual_elapsed_ms * 1.0e9)
                              : 0.0;
  result.scheduled_tflops =
      options.steady_sec > 0.0 ? total_flops / (options.steady_sec * 1.0e12) : 0.0;
  result.duty_control_mode = "persistent_warpgroup_full_duty";
  result.spatial_control_mode = "persistent_cta_count";
  result.grid_blocks = run_result.grid_blocks;
  result.blocks_per_sm = options.blocks_per_sm;
  result.occupancy_max_active_blocks_per_sm =
      run_result.occupancy_max_active_blocks_per_sm;
  result.effective_blocks_per_sm_estimate =
      run_result.effective_blocks_per_sm_estimate;
  result.occupancy_limited = run_result.occupancy_limited;
  result.gemm_semantics = "persistent_sm90a_wgmma_instruction_burn";
  result.expected_use_case =
      "isolates Hopper Tensor Core pressure with shared-memory-resident operands";
  result.per_gemm_flops = 0.0;
  result.total_flops_per_call = 0.0;
  result.dense_equivalent_flops = 0.0;
  result.matrix_shape_is_real = false;
  result.synthetic_mnk_controls_mma_count = false;
  result.memory_traffic_minimized = true;
  result.uses_global_ab = "false";
  result.uses_shared_memory_tiles = true;
  result.tensor_core_execution_path = "sm90a_wgmma";
  result.cutlass_atom_arch = "SM90a_WGMMA";
  result.uses_cutlass_tiled_mma_object = true;
  result.uses_cutlass_mma_atom_direct = false;
  result.warpgroup_threads = 128;
  result.warps_per_warpgroup = 4;
  result.wgmma_instruction_m = 64;
  result.wgmma_instruction_n = 64;
  result.wgmma_instruction_k = 16;
  result.wgmma_ops_executed = run_result.wgmma_ops_executed;
  result.wgmma_flops_per_op = kWgmmaFlopsPerOp;
  result.wgmma_wait_group = options.wgmma_wait_group;
  result.wgmma_accumulator_sets = options.wgmma_accumulator_sets;
  result.wgmma_ops_per_check = options.wgmma_ops_per_check;
  result.wgmma_smem_operand_bytes_per_op =
      (64ULL * 16ULL + 64ULL * 16ULL) * sizeof(__nv_bfloat16);
  result.uses_tma = false;
  result.uses_tma_known = true;
  result.initial_global_load_bytes = run_result.initial_global_load_bytes;
  result.steady_global_load_bytes_per_loop =
      run_result.steady_global_load_bytes_per_loop;
  result.final_global_store_bytes = run_result.final_global_store_bytes;
  result.correctness_smoke_passed = run_result.correctness_smoke_passed;
  result.correctness_reference = run_result.correctness_reference;
  result.correctness_observed = run_result.correctness_observed;
  result.correctness_abs_error = run_result.correctness_abs_error;
  result.note =
      "wgmma_persistent phase 1 uses one 128-thread warpgroup per CTA, BF16 A/B tiles initialized once in shared memory, FP32 accumulation, no TMA, no steady-state global A/B loads, no hot-loop atomics, and a coarse warpgroup-uniform timer check. The requested duration begins after in-kernel shared-memory and descriptor setup, while actual_elapsed_ms conservatively includes that small one-time startup. m/n/k are retained only for CLI compatibility and do not determine executed work. blocks_per_sm is requested launch density; CUDA block scheduling approximates active-SM coverage and does not guarantee specific SM IDs.";
  return result;
}
#else
Result measure_wgmma_persistent(const Options&, int) {
  throw std::runtime_error(
      "engine=wgmma_persistent requires vendored CUTLASS/CuTe SM90 GMMA support and an sm_90a-capable CUDA compiler; this binary was built without HAVE_WGMMA_SM90A and no WMMA or cuBLAS fallback is allowed");
}
#endif

void print_json(const Result& result) {
  std::cout << "{"
            << "\"workload\":\"tensor_core_burn\","
            << "\"dtype\":\"" << result.dtype << "\","
            << "\"m\":" << result.m << ","
            << "\"n\":" << result.n << ","
            << "\"k\":" << result.k << ","
            << "\"duty_cycle\":" << result.duty_cycle << ","
            << "\"active_sm_fraction\":" << result.active_sm_fraction << ","
            << "\"requested_sm_count\":" << result.requested_sm_count << ","
            << "\"steady_sec\":" << result.steady_sec << ","
            << "\"active_elapsed_ms\":" << result.active_elapsed_ms << ","
            << "\"iterations\":" << result.iterations << ","
            << "\"active_tflops\":" << result.active_tflops << ","
            << "\"scheduled_tflops\":" << result.scheduled_tflops << ","
            << "\"engine\":\"" << result.engine << "\","
            << "\"period_ms\":" << result.period_ms << ","
            << "\"actual_elapsed_ms\":" << result.actual_elapsed_ms << ","
            << "\"duty_control_mode\":\"" << result.duty_control_mode << "\","
            << "\"spatial_control_mode\":\"" << result.spatial_control_mode << "\","
            << "\"sm_count_target_applied\":"
            << (result.sm_count_target_applied ? "true" : "false") << ","
            << "\"grid_blocks\":" << result.grid_blocks << ","
            << "\"blocks_per_sm\":" << result.blocks_per_sm << ","
            << "\"mma_iters_per_loop\":" << result.mma_iters_per_loop << ","
            << "\"accumulators_per_warp\":" << result.accumulators_per_warp << ","
            << "\"atomic_period\":" << result.atomic_period << ","
            << "\"batch_count\":" << result.batch_count << ","
            << "\"per_gemm_flops\":" << result.per_gemm_flops << ","
            << "\"total_flops_per_call\":" << result.total_flops_per_call << ","
            << "\"gemm_semantics\":\"" << result.gemm_semantics << "\","
            << "\"expected_use_case\":\"" << result.expected_use_case << "\","
            << "\"warnings\":[";
  for (std::size_t index = 0; index < result.warnings.size(); ++index) {
    if (index > 0) {
      std::cout << ",";
    }
    std::cout << "\"" << result.warnings[index] << "\"";
  }
  std::cout << "],"
            << "\"occupancy_max_active_blocks_per_sm\":"
            << result.occupancy_max_active_blocks_per_sm << ","
            << "\"effective_blocks_per_sm_estimate\":"
            << result.effective_blocks_per_sm_estimate << ","
            << "\"occupancy_limited\":"
            << (result.occupancy_limited ? "true" : "false") << ","
            << "\"spatial_coverage_fraction\":" << result.spatial_coverage_fraction << ","
            << "\"sparsity_test_dimension\":\"" << result.sparsity_test_dimension << "\","
            << "\"tensor_core_execution_path\":\"" << result.tensor_core_execution_path
            << "\","
            << "\"sparsity_mode\":\"" << result.sparsity_mode << "\","
            << "\"zero_ratio\":" << result.zero_ratio << ","
            << "\"zero_pattern\":\"" << result.zero_pattern << "\","
            << "\"sparse_operand\":\"" << result.sparse_operand << "\","
            << "\"sparse_engine\":\"" << result.sparse_engine << "\","
            << "\"sparse_pattern\":\"" << result.sparse_pattern << "\","
            << "\"uses_sparse_tensor_core\":"
            << (result.uses_sparse_tensor_core ? "true" : "false") << ","
            << "\"dense_mma_instruction_count_unchanged\":"
            << (result.dense_mma_instruction_count_unchanged ? "true" : "false") << ","
            << "\"correctness_smoke_passed\":"
            << (result.correctness_smoke_passed ? "true" : "false") << ","
            << "\"correctness_reference\":" << result.correctness_reference << ","
            << "\"correctness_observed\":" << result.correctness_observed << ","
            << "\"correctness_abs_error\":" << result.correctness_abs_error << ","
            << "\"logical_m\":" << result.logical_m << ","
            << "\"logical_n\":" << result.logical_n << ","
            << "\"logical_k\":" << result.logical_k << ","
            << "\"dense_equivalent_flops\":" << result.dense_equivalent_flops << ","
            << "\"measured_runtime_ms\":" << result.measured_runtime_ms << ","
            << "\"dense_baseline_tflops\":" << result.dense_baseline_tflops << ","
            << "\"sparse_tflops_or_dense_equivalent_tflops\":"
            << result.sparse_tflops_or_dense_equivalent_tflops << ","
            << "\"speedup_vs_dense\":" << result.speedup_vs_dense << ","
            << "\"joules_per_dense_equivalent_flop\":"
            << result.joules_per_dense_equivalent_flop << ","
            << "\"compression_time_excluded\":"
            << (result.compression_time_excluded ? "true" : "false") << ","
            << "\"setup_time_excluded\":"
            << (result.setup_time_excluded ? "true" : "false") << ","
            << "\"matrix_shape_is_real\":"
            << (result.matrix_shape_is_real ? "true" : "false") << ","
            << "\"synthetic_mnk_controls_mma_count\":"
            << (result.synthetic_mnk_controls_mma_count ? "true" : "false") << ","
            << "\"memory_traffic_minimized\":"
            << (result.memory_traffic_minimized ? "true" : "false") << ","
            << "\"uses_global_ab\":\"" << result.uses_global_ab << "\","
            << "\"uses_shared_memory_tiles\":"
            << (result.uses_shared_memory_tiles ? "true" : "false") << ","
            << "\"cutlass_tile_m\":" << result.cutlass_tile_m << ","
            << "\"cutlass_tile_n\":" << result.cutlass_tile_n << ","
            << "\"cutlass_tile_k\":" << result.cutlass_tile_k << ","
            << "\"synthetic_m_tiles\":" << result.synthetic_m_tiles << ","
            << "\"synthetic_n_tiles\":" << result.synthetic_n_tiles << ","
            << "\"synthetic_k_tiles\":" << result.synthetic_k_tiles << ","
            << "\"synthetic_mma_ops_per_loop\":" << result.synthetic_mma_ops_per_loop << ","
            << "\"synthetic_mma_ops_per_loop_source\":\""
            << result.synthetic_mma_ops_per_loop_source << "\","
            << "\"synthetic_mma_ops_per_loop_override\":"
            << result.synthetic_mma_ops_per_loop_override << ","
            << "\"requested_synthetic_tile_ops\":"
            << result.requested_synthetic_tile_ops << ","
            << "\"distributed_synthetic_ops_per_block\":"
            << result.distributed_synthetic_ops_per_block << ","
            << "\"synthetic_mma_ops_cap\":" << result.synthetic_mma_ops_cap << ","
            << "\"synthetic_mma_ops_cap_applied\":"
            << (result.synthetic_mma_ops_cap_applied ? "true" : "false") << ","
            << "\"cutlass_atom_shape_m\":" << result.cutlass_atom_shape_m << ","
            << "\"cutlass_atom_shape_n\":" << result.cutlass_atom_shape_n << ","
            << "\"cutlass_atom_shape_k\":" << result.cutlass_atom_shape_k << ","
            << "\"cutlass_atom_arch\":\"" << result.cutlass_atom_arch << "\","
            << "\"uses_cutlass_tiled_mma_object\":"
            << (result.uses_cutlass_tiled_mma_object ? "true" : "false") << ","
            << "\"uses_cutlass_mma_atom_direct\":"
            << (result.uses_cutlass_mma_atom_direct ? "true" : "false") << ","
            << "\"warpgroup_threads\":" << result.warpgroup_threads << ","
            << "\"warps_per_warpgroup\":" << result.warps_per_warpgroup << ","
            << "\"wgmma_instruction_m\":" << result.wgmma_instruction_m << ","
            << "\"wgmma_instruction_n\":" << result.wgmma_instruction_n << ","
            << "\"wgmma_instruction_k\":" << result.wgmma_instruction_k << ","
            << "\"wgmma_ops_executed\":" << result.wgmma_ops_executed << ","
            << "\"wgmma_flops_per_op\":" << result.wgmma_flops_per_op << ","
            << "\"wgmma_wait_group\":" << result.wgmma_wait_group << ","
            << "\"wgmma_accumulator_sets\":" << result.wgmma_accumulator_sets << ","
            << "\"wgmma_ops_per_check\":" << result.wgmma_ops_per_check << ","
            << "\"wgmma_smem_operand_bytes_per_op\":"
            << result.wgmma_smem_operand_bytes_per_op << ","
            << "\"uses_tma\":"
            << (result.uses_tma_known
                    ? (result.uses_tma ? "true" : "false")
                    : "null")
            << ","
            << "\"initial_global_load_bytes\":" << result.initial_global_load_bytes << ","
            << "\"steady_global_load_bytes_per_loop\":"
            << result.steady_global_load_bytes_per_loop << ","
            << "\"final_global_store_bytes\":" << result.final_global_store_bytes << ","
            << "\"note\":\"" << result.note << "\""
            << "}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  __nv_bfloat16* a = nullptr;
  __nv_bfloat16* b = nullptr;
  float* c = nullptr;
  cublasHandle_t handle = nullptr;

  try {
    Options options = parse_args(argc, argv);
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    const int requested_sm_count = std::max(
        1, static_cast<int>(std::ceil(prop.multiProcessorCount * options.active_sm_fraction)));

    if (options.engine == "wmma_persistent" && options.sparsity_mode == "dense_zero") {
      throw std::runtime_error(
          "sparsity-mode=dense_zero is implemented for engine=cublas dense operands; "
          "wmma_persistent uses synthetic WMMA fragments and does not consume initialized A/B operands");
    }
    if (options.engine == "wmma_persistent" && options.sparsity_mode == "structured_2to4") {
      throw std::runtime_error(
          "sparsity-mode=structured_2to4 is implemented for engine=cublas with a sparse GEMM backend; "
          "wmma_persistent remains a dense synthetic WMMA engine");
    }
    if (options.engine == "cutlass_tile_burn" && options.sparsity_mode != "none") {
      throw std::runtime_error(
          "engine=cutlass_tile_burn is a dense synthetic tile burn and supports --sparsity-mode none only");
    }
    if (options.engine == "cublas_strided_batched" && options.sparsity_mode != "none") {
      throw std::runtime_error(
          "engine=cublas_strided_batched supports --sparsity-mode none only");
    }
    if (options.engine == "wmma_persistent") {
      Result result = measure_wmma_persistent(options, prop, requested_sm_count);
      print_json(result);
      return 0;
    }
    if (options.engine == "wgmma_persistent") {
      Result result = measure_wgmma_persistent(options, requested_sm_count);
      print_json(result);
      return 0;
    }
    if (options.engine == "cutlass_tile_burn") {
      Result result = measure_cutlass_tile_burn(options, prop, requested_sm_count);
      print_json(result);
      return 0;
    }

    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
    bool sm_count_target_applied = false;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    check_cublas(cublasSetSmCountTarget(handle, requested_sm_count),
                 "cublasSetSmCountTarget");
    sm_count_target_applied = true;
#endif

    const std::size_t batch_multiplier =
        options.engine == "cublas_strided_batched"
            ? static_cast<std::size_t>(options.batch_count)
            : 1;
    const std::size_t a_count =
        static_cast<std::size_t>(options.m) * options.k * batch_multiplier;
    const std::size_t b_count =
        static_cast<std::size_t>(options.k) * options.n * batch_multiplier;
    const std::size_t c_count =
        static_cast<std::size_t>(options.m) * options.n * batch_multiplier;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&a), a_count * sizeof(__nv_bfloat16)),
               "cudaMalloc(a)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&b), b_count * sizeof(__nv_bfloat16)),
               "cudaMalloc(b)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c), c_count * sizeof(float)),
               "cudaMalloc(c)");
    const int pattern_id = zero_pattern_id(options.zero_pattern);
    fill_bf16(
        a,
        options.m,
        options.k * static_cast<int>(batch_multiplier),
        options.k,
        false,
        operand_selected(options, "A") ? options.zero_ratio : 0.0,
        pattern_id,
        1.0f);
    fill_bf16(
        b,
        options.k,
        options.n * static_cast<int>(batch_multiplier),
        options.k,
        true,
        operand_selected(options, "B") ? options.zero_ratio : 0.0,
        pattern_id,
        1.0f);
    check_cuda(cudaMemset(c, 0, c_count * sizeof(float)), "cudaMemset(c)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    if (options.sparsity_mode == "structured_2to4") {
      if (options.sparse_engine == "cutlass_spgemm") {
        throw std::runtime_error(
            "sparse-engine=cutlass_spgemm is not enabled in this build; use "
            "--sparse-engine cusparselt when cuSPARSELt is available");
      }
      if (options.sparse_engine != "cusparselt") {
        throw std::runtime_error("structured_2to4 currently supports --sparse-engine cusparselt");
      }
      if (options.sparse_operand != "A") {
        throw std::runtime_error(
            "structured_2to4 currently supports operand A first; use --sparse-operand A");
      }
#if defined(HAVE_CUSPARSELT)
      Result result = measure_cusparselt_sparse(
          handle, options, a, b, c, requested_sm_count, sm_count_target_applied);
      print_json(result);
#else
      throw std::runtime_error(
          "sparsity-mode=structured_2to4 --sparse-engine cusparselt requires "
          "cuSPARSELt headers and libcusparseLt at build time; this binary was "
          "built without HAVE_CUSPARSELT and refuses to fall back to dense_zero "
          "or dense GEMM");
#endif
      cublasDestroy(handle);
      cudaFree(a);
      cudaFree(b);
      cudaFree(c);
      return 0;
    }

    if (options.engine == "cublas_strided_batched") {
      Result result = measure_cublas_strided_batched(
          handle, options, a, b, c, requested_sm_count, sm_count_target_applied);
      print_json(result);

      cublasDestroy(handle);
      cudaFree(a);
      cudaFree(b);
      cudaFree(c);
      return 0;
    }

    CorrectnessSmoke correctness_smoke = run_correctness_smoke(handle, options, a, b, c);
    Result result = measure_cublas(
        handle, options, a, b, c, requested_sm_count, sm_count_target_applied, correctness_smoke);
    print_json(result);

    cublasDestroy(handle);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"error\":\"" << error.what() << "\"}" << std::endl;
    if (handle) {
      cublasDestroy(handle);
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 1;
  }
}
