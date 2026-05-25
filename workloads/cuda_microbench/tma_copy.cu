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
  std::string direction = "gm_to_smem";
  std::string mechanism = "normal";
  std::size_t bytes = 16777216;
  double warmup_sec = 10.0;
  double steady_sec = 60.0;
};

std::uint32_t* src_buffer = nullptr;
std::uint32_t* dst_buffer = nullptr;
std::size_t element_count = 0;
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
    } else if (arg == "--direction") {
      options.direction = require_value(arg);
    } else if (arg == "--mechanism") {
      options.mechanism = require_value(arg);
    } else if (arg == "--bytes") {
      options.bytes = parse_size(require_value(arg), arg);
    } else if (arg == "--warmup-sec") {
      options.warmup_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--steady-sec") {
      options.steady_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage: tma_copy --device 0 --direction gm_to_smem|smem_to_gm "
          << "--mechanism normal|tma --bytes 16777216 --warmup-sec 10 --steady-sec 60\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (options.direction != "gm_to_smem" && options.direction != "smem_to_gm") {
    throw std::runtime_error("--direction must be gm_to_smem or smem_to_gm");
  }
  if (options.mechanism != "normal" && options.mechanism != "tma") {
    throw std::runtime_error("--mechanism must be normal or tma");
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

__global__ void staged_copy_kernel(
    const std::uint32_t* src,
    std::uint32_t* dst,
    std::size_t count,
    bool reverse) {
  extern __shared__ std::uint32_t tile[];
  std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x;
  std::size_t index = base + threadIdx.x;
  if (index < count) {
    tile[threadIdx.x] = src[index];
  }
  __syncthreads();
  if (index < count) {
    std::size_t out = reverse ? (count - 1 - index) : index;
    dst[out] = tile[threadIdx.x];
  }
}

void launch(const Options& options) {
  bool reverse = options.direction == "smem_to_gm";
  staged_copy_kernel<<<grid_blocks, 256, 256 * sizeof(std::uint32_t)>>>(
      src_buffer, dst_buffer, element_count, reverse);
  check(cudaGetLastError(), "staged_copy_kernel");
}

void run_for_seconds(const Options& options, double seconds, std::uint64_t& iterations) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    launch(options);
    ++iterations;
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(batch)");
  }
}

void print_json(
    const Options& options,
    bool tma_supported,
    double elapsed_ms,
    std::uint64_t iterations) {
  double elapsed_s = std::max(elapsed_ms / 1000.0, 1e-9);
  double bytes = static_cast<double>(options.bytes) * static_cast<double>(iterations);
  bool fallback_used = options.mechanism == "tma" && !tma_supported;
  std::cout << "{"
            << "\"workload\":\"tma_copy\","
            << "\"direction\":\"" << options.direction << "\","
            << "\"mechanism\":\"" << options.mechanism << "\","
            << "\"tma_supported\":" << (tma_supported ? "true" : "false") << ","
            << "\"fallback_used\":" << (fallback_used ? "true" : "false") << ","
            << "\"bytes\":" << options.bytes << ","
            << "\"steady_sec\":" << options.steady_sec << ","
            << "\"elapsed_ms\":" << elapsed_ms << ","
            << "\"iterations\":" << iterations << ","
            << "\"bandwidth_gbps\":" << bytes / elapsed_s / 1.0e9
            << "}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    check(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    element_count = options.bytes / sizeof(std::uint32_t);
    grid_blocks = static_cast<int>((element_count + 255) / 256);

    check(cudaMalloc(reinterpret_cast<void**>(&src_buffer), options.bytes),
          "cudaMalloc(src_buffer)");
    check(cudaMalloc(reinterpret_cast<void**>(&dst_buffer), options.bytes),
          "cudaMalloc(dst_buffer)");
    check(cudaMemset(src_buffer, 0x5a, options.bytes), "cudaMemset(src_buffer)");
    check(cudaMemset(dst_buffer, 0, options.bytes), "cudaMemset(dst_buffer)");

#if defined(GPU_POWER_ENABLE_EXPERIMENTAL_TMA)
    bool tma_supported = prop.major >= 9;
#else
    bool tma_supported = false;
#endif

    std::uint64_t warmup_iterations = 0;
    run_for_seconds(options, options.warmup_sec, warmup_iterations);

    EventPair events;
    std::uint64_t iterations = 0;
    check(cudaEventRecord(events.start()), "cudaEventRecord(start)");
    run_for_seconds(options, options.steady_sec, iterations);
    check(cudaEventRecord(events.stop()), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
          "cudaEventElapsedTime");
    print_json(options, tma_supported, elapsed_ms, iterations);

    cudaFree(src_buffer);
    cudaFree(dst_buffer);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"error\":\"" << error.what() << "\"}" << std::endl;
    cudaFree(src_buffer);
    cudaFree(dst_buffer);
    return 1;
  }
}

