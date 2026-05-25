#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
  std::string mode = "fp32_fma";
  double duty_cycle = 1.0;
  std::size_t buffer_mb = 1024;
  double warmup_sec = 30.0;
  double steady_sec = 60.0;
};

struct Result {
  std::string mode;
  double duty_cycle = 0.0;
  std::size_t buffer_mb = 0;
  double steady_sec = 0.0;
  double active_elapsed_ms = 0.0;
  std::uint64_t iterations = 0;
  double active_gops = 0.0;
  double scheduled_gops = 0.0;
};

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << call << " failed: " << cudaGetErrorString(status);
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

std::size_t parse_size(const std::string& value, const std::string& name) {
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed == 0) {
    throw std::runtime_error("Invalid " + name + ": " + value);
  }
  return static_cast<std::size_t>(parsed);
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
    } else if (arg == "--mode") {
      options.mode = require_value(arg);
    } else if (arg == "--duty-cycle") {
      options.duty_cycle = parse_double(require_value(arg), arg);
    } else if (arg == "--buffer-mb") {
      options.buffer_mb = parse_size(require_value(arg), arg);
    } else if (arg == "--warmup-sec") {
      options.warmup_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--steady-sec") {
      options.steady_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage: cuda_core_burn --device 0 --mode fp32_fma|int32_logic "
          << "--duty-cycle 1.0 --buffer-mb 1024 --warmup-sec 30 --steady-sec 60\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.mode != "fp32_fma" && options.mode != "int32_logic") {
    throw std::runtime_error("--mode must be fp32_fma or int32_logic");
  }
  if (options.duty_cycle < 0.0 || options.duty_cycle > 1.0) {
    throw std::runtime_error("--duty-cycle must be in [0, 1]");
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

constexpr int kThreadsPerBlock = 256;
constexpr int kBlocksPerSm = 8;
constexpr int kInnerOps = 256;
constexpr int kBatchKernels = 8;

float* fp_buffer = nullptr;
std::uint32_t* int_buffer = nullptr;
std::size_t element_count = 0;
int grid_blocks = 0;

__global__ void fp32_fma_kernel(float* data, std::size_t count) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  while (index < count) {
    float x = data[index];
#pragma unroll 16
    for (int op = 0; op < kInnerOps; ++op) {
      x = fmaf(x, 1.000001f, 0.000001f);
    }
    data[index] = x;
    index += stride;
  }
}

__global__ void int32_logic_kernel(std::uint32_t* data, std::size_t count) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  while (index < count) {
    std::uint32_t x = data[index] ^ 0x9e3779b9U;
#pragma unroll 16
    for (int op = 0; op < kInnerOps; ++op) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      x += 0x7f4a7c15U;
    }
    data[index] = x;
    index += stride;
  }
}

using LaunchFn = void (*)();

void launch_fp32() {
  fp32_fma_kernel<<<grid_blocks, kThreadsPerBlock>>>(fp_buffer, element_count);
  check_cuda(cudaGetLastError(), "fp32_fma_kernel");
}

void launch_int32() {
  int32_logic_kernel<<<grid_blocks, kThreadsPerBlock>>>(int_buffer, element_count);
  check_cuda(cudaGetLastError(), "int32_logic_kernel");
}

void run_active_window(LaunchFn launch, double seconds, std::uint64_t& iterations) {
  if (seconds <= 0.0) {
    return;
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    for (int index = 0; index < kBatchKernels; ++index) {
      launch();
      ++iterations;
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(active batch)");
  }
}

Result measure(const Options& options, LaunchFn launch) {
  std::uint64_t warmup_iterations = 0;
  run_active_window(launch, options.warmup_sec, warmup_iterations);

  if (options.duty_cycle == 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(options.steady_sec));
    return Result{
        options.mode,
        options.duty_cycle,
        options.buffer_mb,
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
    run_active_window(launch, active_sec, iterations);
    if (idle_sec > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(idle_sec));
    }
  }
  check_cuda(cudaEventRecord(events.stop()), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(stop)");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
             "cudaEventElapsedTime");

  const double ops_per_element = options.mode == "fp32_fma" ? (2.0 * kInnerOps)
                                                            : (4.0 * kInnerOps);
  const double ops_per_kernel = static_cast<double>(element_count) * ops_per_element;
  const double total_ops = ops_per_kernel * static_cast<double>(iterations);
  const double active_elapsed_s = std::max(static_cast<double>(elapsed_ms) / 1000.0, 1e-9);
  return Result{
      options.mode,
      options.duty_cycle,
      options.buffer_mb,
      options.steady_sec,
      static_cast<double>(elapsed_ms),
      iterations,
      total_ops / active_elapsed_s / 1.0e9,
      total_ops / options.steady_sec / 1.0e9,
  };
}

void print_json(const Result& result) {
  std::cout << "{"
            << "\"workload\":\"cuda_core_burn\","
            << "\"mode\":\"" << result.mode << "\","
            << "\"duty_cycle\":" << result.duty_cycle << ","
            << "\"buffer_mb\":" << result.buffer_mb << ","
            << "\"steady_sec\":" << result.steady_sec << ","
            << "\"active_elapsed_ms\":" << result.active_elapsed_ms << ","
            << "\"iterations\":" << result.iterations << ","
            << "\"active_gops\":" << result.active_gops << ","
            << "\"scheduled_gops\":" << result.scheduled_gops
            << "}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    grid_blocks = std::max(1, prop.multiProcessorCount * kBlocksPerSm);

    const std::uint64_t buffer_bytes =
        static_cast<std::uint64_t>(options.buffer_mb) * 1024ULL * 1024ULL;
    element_count = buffer_bytes / sizeof(float);

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&fp_buffer), buffer_bytes),
               "cudaMalloc(fp_buffer)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&int_buffer), buffer_bytes),
               "cudaMalloc(int_buffer)");
    check_cuda(cudaMemset(fp_buffer, 0x3f, buffer_bytes), "cudaMemset(fp_buffer)");
    check_cuda(cudaMemset(int_buffer, 0x5a, buffer_bytes), "cudaMemset(int_buffer)");

    Result result = measure(
        options, options.mode == "fp32_fma" ? launch_fp32 : launch_int32);
    print_json(result);

    cudaFree(fp_buffer);
    cudaFree(int_buffer);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"error\":\"" << error.what() << "\"}" << std::endl;
    cudaFree(fp_buffer);
    cudaFree(int_buffer);
    return 1;
  }
}
