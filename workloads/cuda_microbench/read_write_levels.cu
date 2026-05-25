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
  std::string mode = "read";
  double load_factor = 1.0;
  std::size_t buffer_mb = 4096;
  double warmup_sec = 10.0;
  double steady_sec = 60.0;
};

struct Result {
  std::string mode;
  double load_factor = 0.0;
  std::size_t buffer_mb = 0;
  std::uint64_t active_bytes = 0;
  double steady_sec = 0.0;
  double elapsed_ms = 0.0;
  std::uint64_t iterations = 0;
  double bandwidth_gbps = 0.0;
};

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
    } else if (arg == "--mode") {
      options.mode = require_value(arg);
    } else if (arg == "--load-factor") {
      options.load_factor = parse_double(require_value(arg), arg);
    } else if (arg == "--buffer-mb") {
      options.buffer_mb = parse_size(require_value(arg), arg);
    } else if (arg == "--warmup-sec") {
      options.warmup_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--steady-sec") {
      options.steady_sec = parse_double(require_value(arg), arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage: read_write_levels --device 0 --mode read|write "
          << "--load-factor F --buffer-mb MB --warmup-sec S --steady-sec S\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.mode != "read" && options.mode != "write") {
    throw std::runtime_error("--mode must be read or write");
  }
  if (options.load_factor <= 0.0 || options.load_factor > 1.0) {
    throw std::runtime_error("--load-factor must be in (0, 1]");
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

__global__ void read_kernel(
    const std::uint32_t* data,
    std::uint32_t* sink,
    std::size_t active_elements) {
  const std::size_t stride = blockDim.x * gridDim.x;
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t sum = 0;
  while (index < active_elements) {
    sum += data[index];
    index += stride;
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void write_kernel(
    std::uint32_t* data,
    std::size_t active_elements,
    std::uint32_t seed) {
  const std::size_t stride = blockDim.x * gridDim.x;
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < active_elements) {
    data[index] = seed + static_cast<std::uint32_t>(index);
    index += stride;
  }
}

using LaunchFn = void (*)(cudaStream_t, std::uint32_t);

constexpr int kThreadsPerBlock = 256;
constexpr int kBlocksPerSm = 8;
constexpr int kBatchKernels = 16;

std::uint32_t* device_buffer = nullptr;
std::uint32_t* sink_buffer = nullptr;
std::size_t active_elements = 0;
int grid_blocks = 0;

void launch_read(cudaStream_t stream, std::uint32_t seed) {
  (void)seed;
  read_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      device_buffer, sink_buffer, active_elements);
  check(cudaGetLastError(), "read_kernel");
}

void launch_write(cudaStream_t stream, std::uint32_t seed) {
  write_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      device_buffer, active_elements, seed);
  check(cudaGetLastError(), "write_kernel");
}

void run_for_seconds(cudaStream_t stream, LaunchFn launch, double seconds) {
  if (seconds <= 0.0) {
    return;
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  std::uint32_t seed = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    for (int index = 0; index < kBatchKernels; ++index) {
      launch(stream, seed++);
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup batch)");
  }
}

Result measure(const Options& options, LaunchFn launch, std::uint64_t active_bytes) {
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

  run_for_seconds(stream, launch, options.warmup_sec);

  EventPair events;
  std::uint64_t iterations = 0;
  std::uint32_t seed = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(options.steady_sec);

  check(cudaEventRecord(events.start(), stream), "cudaEventRecord(start)");
  while (std::chrono::steady_clock::now() < deadline) {
    for (int index = 0; index < kBatchKernels; ++index) {
      launch(stream, seed++);
      ++iterations;
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(steady batch)");
  }
  check(cudaEventRecord(events.stop(), stream), "cudaEventRecord(stop)");
  check(cudaEventSynchronize(events.stop()), "cudaEventSynchronize(stop)");

  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()),
        "cudaEventElapsedTime");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");

  const double elapsed_s = std::max(static_cast<double>(elapsed_ms) / 1000.0, 1e-9);
  const double total_bytes =
      static_cast<double>(active_bytes) * static_cast<double>(iterations);
  return Result{
      options.mode,
      options.load_factor,
      options.buffer_mb,
      active_bytes,
      options.steady_sec,
      static_cast<double>(elapsed_ms),
      iterations,
      total_bytes / elapsed_s / 1.0e9,
  };
}

void print_json(const Result& result) {
  std::cout << "{"
            << "\"mode\":\"" << result.mode << "\","
            << "\"load_factor\":" << result.load_factor << ","
            << "\"buffer_mb\":" << result.buffer_mb << ","
            << "\"active_bytes\":" << result.active_bytes << ","
            << "\"steady_sec\":" << result.steady_sec << ","
            << "\"elapsed_ms\":" << result.elapsed_ms << ","
            << "\"iterations\":" << result.iterations << ","
            << "\"bandwidth_gbps\":" << result.bandwidth_gbps
            << "}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    check(cudaSetDevice(options.device), "cudaSetDevice");

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    grid_blocks = std::max(1, prop.multiProcessorCount * kBlocksPerSm);

    const std::uint64_t buffer_bytes =
        static_cast<std::uint64_t>(options.buffer_mb) * 1024ULL * 1024ULL;
    const std::size_t total_elements = buffer_bytes / sizeof(std::uint32_t);
    active_elements = std::max<std::size_t>(
        1, static_cast<std::size_t>(static_cast<double>(total_elements) * options.load_factor));
    const std::uint64_t active_bytes =
        static_cast<std::uint64_t>(active_elements) * sizeof(std::uint32_t);

    check(cudaMalloc(reinterpret_cast<void**>(&device_buffer), buffer_bytes),
          "cudaMalloc(device_buffer)");
    check(cudaMalloc(
              reinterpret_cast<void**>(&sink_buffer),
              static_cast<std::size_t>(grid_blocks) * kThreadsPerBlock *
                  sizeof(std::uint32_t)),
          "cudaMalloc(sink_buffer)");
    check(cudaMemset(device_buffer, 0x5a, buffer_bytes), "cudaMemset(device_buffer)");
    check(cudaMemset(
              sink_buffer,
              0,
              static_cast<std::size_t>(grid_blocks) * kThreadsPerBlock *
                  sizeof(std::uint32_t)),
          "cudaMemset(sink_buffer)");

    print_json(measure(
        options,
        options.mode == "read" ? launch_read : launch_write,
        active_bytes));

    cudaFree(device_buffer);
    cudaFree(sink_buffer);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"error\":\"" << error.what() << "\"}" << std::endl;
    return 1;
  }
}
