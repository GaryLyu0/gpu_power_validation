#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
  int m = 8192;
  int n = 8192;
  int k = 8192;
  double duty_cycle = 1.0;
  double active_sm_fraction = 1.0;
  double warmup_sec = 30.0;
  double steady_sec = 60.0;
};

struct Result {
  std::string dtype;
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
    } else if (arg == "--help") {
      std::cout
          << "Usage: tensor_core_burn --device 0 --dtype bf16 --m 8192 --n 8192 "
          << "--k 8192 --duty-cycle 1.0 --active-sm-fraction 1.0 "
          << "--warmup-sec 30 --steady-sec 60\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.dtype != "bf16") {
    throw std::runtime_error("Only --dtype bf16 is supported in this initial workload");
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

Result measure(
    cublasHandle_t handle,
    const Options& options,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* c,
    int requested_sm_count) {
  std::uint64_t warmup_iterations = 0;
  run_active_window(handle, options, a, b, c, options.warmup_sec, warmup_iterations);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  if (options.duty_cycle == 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(options.steady_sec));
    return Result{
        options.dtype,
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
    };
  }

  EventPair events;
  std::uint64_t iterations = 0;
  constexpr double period_sec = 1.0;
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
  };
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
            << "\"scheduled_tflops\":" << result.scheduled_tflops
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

    check_cublas(cublasCreate(&handle), "cublasCreate");
    check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
    check_cublas(cublasSetSmCountTarget(handle, requested_sm_count),
                 "cublasSetSmCountTarget");
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

    Result result = measure(handle, options, a, b, c, requested_sm_count);
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
