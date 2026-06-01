#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

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

namespace {

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
  std::string note;
};

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

int parse_device(const std::string& value) {
  char* end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed < 0) {
    throw std::runtime_error("Invalid --device: " + value);
  }
  return static_cast<int>(parsed);
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
    } else if (arg == "--help") {
      std::cout
          << "Usage: tensor_core_burn --device 0 --dtype bf16 "
          << "--engine cublas|wmma_persistent --m 8192 --n 8192 "
          << "--k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 "
          << "--period-ms 1000 --warmup-sec 30 --steady-sec 60\n\n"
          << "engine=cublas keeps the original cuBLAS GEMM active/idle windows.\n"
          << "engine=wmma_persistent launches persistent CTAs and controls "
          << "active/idle phases inside the CUDA kernel with clock64.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.dtype != "bf16") {
    throw std::runtime_error("Only --dtype bf16 is supported in this workload");
  }
  if (options.engine != "cublas" && options.engine != "wmma_persistent") {
    throw std::runtime_error("--engine must be cublas or wmma_persistent");
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

__global__ void fill_bf16_kernel(__nv_bfloat16* data, std::size_t count, float value) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  __nv_bfloat16 converted = __float2bfloat16(value);
  while (index < count) {
    data[index] = converted;
    index += stride;
  }
}

void fill_bf16(__nv_bfloat16* data, std::size_t count, float value) {
  int threads = 256;
  int blocks = static_cast<int>(std::min<std::size_t>((count + threads - 1) / threads, 65535));
  fill_bf16_kernel<<<blocks, threads>>>(data, count, value);
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

Result measure_cublas(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c,
    int requested_sm_count,
    bool sm_count_target_applied) {
  std::uint64_t warmup_iterations = 0;
  run_active_window(handle, options, a, b, c, options.warmup_sec, warmup_iterations);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  if (options.duty_cycle == 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(options.steady_sec));
    return Result{
        options.dtype,
        options.engine,
        options.m,
        options.n,
        options.k,
        options.duty_cycle,
        options.active_sm_fraction,
        requested_sm_count,
        options.steady_sec,
        0.0,
        0,
        0.0,
        0.0,
        options.period_ms,
        options.steady_sec * 1000.0,
        "cpu_windowed_cublas",
        "cublas_sm_count_target",
        sm_count_target_applied,
        0,
        0,
        "cuBLAS engine preserves the original CPU-controlled active/idle window behavior.",
    };
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
  return Result{
      options.dtype,
      options.engine,
      options.m,
      options.n,
      options.k,
      options.duty_cycle,
      options.active_sm_fraction,
      requested_sm_count,
      options.steady_sec,
      static_cast<double>(elapsed_ms),
      iterations,
      total_ops / active_elapsed_s / 1.0e12,
      total_ops / options.steady_sec / 1.0e12,
      options.period_ms,
      static_cast<double>(elapsed_ms),
      "cpu_windowed_cublas",
      "cublas_sm_count_target",
      sm_count_target_applied,
      0,
      0,
      "cuBLAS active_sm_fraction is a cuBLAS SM-count target hint; validate actual activity with profiler metrics.",
  };
}

__global__ void wmma_persistent_bf16_kernel(
    float* output,
    unsigned long long* iterations,
    unsigned long long duration_cycles,
    unsigned long long period_cycles,
    unsigned long long active_cycles) {
  constexpr int kTileM = 16;
  constexpr int kTileN = 16;
  constexpr int kTileK = 16;
  constexpr int kMmaPerLoop = 64;
  constexpr int kWarpsPerBlock = 4;

  const unsigned long long start = clock64();
  const int warp_id = threadIdx.x / 32;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileK, __nv_bfloat16, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileK, float> c_frag;

  wmma::fill_fragment(a_frag, __float2bfloat16(1.0f));
  wmma::fill_fragment(b_frag, __float2bfloat16(1.0f));
  wmma::fill_fragment(c_frag, 0.0f);
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
#pragma unroll 4
      for (int i = 0; i < kMmaPerLoop; ++i) {
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      if ((threadIdx.x % 32) == 0) {
        atomicAdd(iterations, static_cast<unsigned long long>(kMmaPerLoop));
      }
#else
      if ((threadIdx.x % 32) == 0) {
        atomicAdd(iterations, 1ULL);
      }
#endif
    } else {
      __nanosleep(1000);
    }
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  wmma::store_matrix_sync(
      output +
          (static_cast<unsigned int>(blockIdx.x) * kWarpsPerBlock +
           static_cast<unsigned int>(warp_id)) *
              kTileM * kTileN,
      c_frag,
      kTileN,
      wmma::mem_row_major);
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

  constexpr int threads_per_block = 128;
  wmma_persistent_bf16_kernel<<<grid_blocks, threads_per_block>>>(
      output,
      iteration_counter,
      duration_cycles,
      period_cycles,
      active_cycles);
  check_cuda(cudaGetLastError(), "wmma_persistent_bf16_kernel");
}

Result measure_wmma_persistent(
    const Options& options,
    const cudaDeviceProp& prop,
    int requested_sm_count) {
  if (prop.major < 8) {
    throw std::runtime_error("engine=wmma_persistent requires SM80 or newer for BF16 WMMA");
  }

  const int clock_rate_khz = get_device_clock_rate_khz(options.device);
  constexpr int blocks_per_sm = 1;
  const int grid_blocks = requested_sm_count * blocks_per_sm;
  constexpr int warps_per_block = 4;
  constexpr int output_values_per_block = warps_per_block * 16 * 16;
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
    launch_wmma_window(
        options, clock_rate_khz, output, iteration_counter, grid_blocks, options.warmup_sec, true);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(wmma warmup)");

    EventPair events;
    check_cuda(cudaEventRecord(events.start()), "cudaEventRecord(wmma start)");
    launch_wmma_window(
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

    return Result{
        options.dtype,
        options.engine,
        options.m,
        options.n,
        options.k,
        options.duty_cycle,
        options.active_sm_fraction,
        requested_sm_count,
        options.steady_sec,
        active_elapsed_ms,
        host_iterations,
        total_ops / active_elapsed_s / 1.0e12,
        total_ops / options.steady_sec / 1.0e12,
        options.period_ms,
        static_cast<double>(elapsed_ms),
        "device_clock64_persistent_kernel",
        "persistent_cta_count",
        false,
        grid_blocks,
        blocks_per_sm,
        "wmma_persistent uses m/n/k as nominal reporting parameters; MAC pressure is controlled by persistent CTAs and mma_sync loop intensity. Validate Tensor Core utilization with Nsight profiler metrics.",
    };
  } catch (...) {
    cudaFree(output);
    cudaFree(iteration_counter);
    throw;
  }
}

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

    if (options.engine == "wmma_persistent") {
      Result result = measure_wmma_persistent(options, prop, requested_sm_count);
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

    const std::size_t a_count = static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_count = static_cast<std::size_t>(options.k) * options.n;
    const std::size_t c_count = static_cast<std::size_t>(options.m) * options.n;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&a), a_count * sizeof(__nv_bfloat16)),
               "cudaMalloc(a)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&b), b_count * sizeof(__nv_bfloat16)),
               "cudaMalloc(b)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c), c_count * sizeof(float)),
               "cudaMalloc(c)");
    fill_bf16(a, a_count, 1.0f);
    fill_bf16(b, b_count, 1.0f);
    check_cuda(cudaMemset(c, 0, c_count * sizeof(float)), "cudaMemset(c)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    Result result = measure_cublas(
        handle, options, a, b, c, requested_sm_count, sm_count_target_applied);
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
