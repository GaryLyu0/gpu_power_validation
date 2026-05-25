#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  int device = 0;
  std::string target = "medium";
  std::size_t buffer_mb = 1024;
  double warmup_sec = 10.0;
  double steady_sec = 60.0;
};

struct Result {
  std::string target;
  std::size_t buffer_mb = 0;
  std::size_t working_set_mb = 0;
  double steady_sec = 0.0;
  double elapsed_ms = 0.0;
  std::uint64_t iterations = 0;
  double bandwidth_gbps = 0.0;
};

std::uint32_t* data_buffer = nullptr;
std::uint32_t* sink_buffer = nullptr;
std::size_t active_elements = 0;
int grid_blocks = 0;

void check(cudaError_t status, const char* call) {
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
    } else if (arg == "--target") {
      options.target = require_value(arg);
    } else if (arg == "--buffer-mb") {
      options.buffer_mb = parse_size(require_value(arg), arg);
    } else if (arg == "--warmup-sec") {
      options.warmup_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--steady-sec") {
      options.steady_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage: l2_hit_sweep --device 0 --target low|medium|high "
          << "--buffer-mb 1024 --warmup-sec 10 --steady-sec 60\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (options.target != "low" && options.target != "medium" && options.target != "high") {
    throw std::runtime_error("--target must be low, medium, or high");
  }
  if (options.warmup_sec < 0.0 || options.steady_sec <= 0.0) {
    throw std::runtime_error("--warmup-sec must be >= 0 and --steady-sec must be > 0");
  }
  return options;
}

class EventPair {
 public:
  EventPair() {
    check(cudaEventCreate(&start_), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
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

__global__ void l2_sweep_kernel(
    const std::uint32_t* data,
    std::uint32_t* sink,
    std::size_t count,
    std::uint32_t salt) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t stride = blockDim.x * gridDim.x;
  std::uint32_t sum = salt + static_cast<std::uint32_t>(tid);
  for (std::size_t index = tid; index < count; index += stride) {
    std::size_t mixed = (index * 1315423911ULL + sum) % count;
    sum += data[mixed];
  }
  sink[tid] = sum;
}

void launch(std::uint32_t salt) {
  l2_sweep_kernel<<<grid_blocks, 256>>>(data_buffer, sink_buffer, active_elements, salt);
  check(cudaGetLastError(), "l2_sweep_kernel");
}

void run_for_seconds(double seconds, std::uint64_t& iterations) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  std::uint32_t salt = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    launch(salt++);
    ++iterations;
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(batch)");
  }
}

std::size_t working_set_mb(const Options& options) {
  if (options.target == "high") {
    return 4;
  }
  if (options.target == "medium") {
    return std::min<std::size_t>(128, options.buffer_mb);
  }
  return options.buffer_mb;
}

Result measure(const Options& options, std::size_t active_mb) {
  std::uint64_t warmup_iterations = 0;
  run_for_seconds(options.warmup_sec, warmup_iterations);

  EventPair events;
  std::uint64_t iterations = 0;
  check(cudaEventRecord(events.start()), "cudaEventRecord(start)");
  run_for_seconds(options.steady_sec, iterations);
  check(cudaEventRecord(events.stop()), "cudaEventRecord(stop)");
  check(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(stop)");

  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
        "cudaEventElapsedTime");

  double elapsed_s = std::max(static_cast<double>(elapsed_ms) / 1000.0, 1e-9);
  double bytes = static_cast<double>(active_elements) * sizeof(std::uint32_t) *
                 static_cast<double>(iterations);
  return Result{
      options.target,
      options.buffer_mb,
      active_mb,
      options.steady_sec,
      static_cast<double>(elapsed_ms),
      iterations,
      bytes / elapsed_s / 1.0e9,
  };
}

void print_json(const Result& result) {
  std::cout << "{"
            << "\"workload\":\"l2_hit_sweep\","
            << "\"target\":\"" << result.target << "\","
            << "\"buffer_mb\":" << result.buffer_mb << ","
            << "\"working_set_mb\":" << result.working_set_mb << ","
            << "\"steady_sec\":" << result.steady_sec << ","
            << "\"elapsed_ms\":" << result.elapsed_ms << ","
            << "\"iterations\":" << result.iterations << ","
            << "\"bandwidth_gbps\":" << result.bandwidth_gbps << ","
            << "\"note\":\"target is workload-shape intent; verify actual L2 hit rate with profiler counters\""
            << "}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    check(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    grid_blocks = std::max(1, prop.multiProcessorCount * 8);

    std::size_t active_mb = working_set_mb(options);
    active_elements = (active_mb * 1024ULL * 1024ULL) / sizeof(std::uint32_t);
    std::size_t buffer_bytes = options.buffer_mb * 1024ULL * 1024ULL;
    std::size_t sink_bytes = static_cast<std::size_t>(grid_blocks) * 256 *
                             sizeof(std::uint32_t);
    check(cudaMalloc(reinterpret_cast<void**>(&data_buffer), buffer_bytes),
          "cudaMalloc(data_buffer)");
    check(cudaMalloc(reinterpret_cast<void**>(&sink_buffer), sink_bytes),
          "cudaMalloc(sink_buffer)");
    check(cudaMemset(data_buffer, 0x5a, buffer_bytes), "cudaMemset(data_buffer)");
    check(cudaMemset(sink_buffer, 0, sink_bytes), "cudaMemset(sink_buffer)");

    print_json(measure(options, active_mb));

    cudaFree(data_buffer);
    cudaFree(sink_buffer);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"error\":\"" << error.what() << "\"}" << std::endl;
    cudaFree(data_buffer);
    cudaFree(sink_buffer);
    return 1;
  }
}

