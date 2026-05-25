#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int device = 0;
  std::string mode = "h2d";
  std::size_t bytes = 268435456;
  double warmup_sec = 10.0;
  double steady_sec = 60.0;
  double presweep_steady_sec = 2.0;
  std::vector<std::size_t> presweep_bytes;
};

struct Result {
  std::string mode;
  std::string phase;
  std::size_t bytes = 0;
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

std::size_t parse_size(const std::string& value) {
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed == 0) {
    throw std::runtime_error("Invalid byte count: " + value);
  }
  return static_cast<std::size_t>(parsed);
}

double parse_seconds(const std::string& value, const std::string& name) {
  char* end = nullptr;
  double parsed = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || parsed < 0.0) {
    throw std::runtime_error("Invalid " + name + ": " + value);
  }
  return parsed;
}

std::vector<std::size_t> parse_size_list(const std::string& value) {
  std::vector<std::size_t> sizes;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      sizes.push_back(parse_size(item));
    }
  }
  return sizes;
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
      options.device = std::stoi(require_value(arg));
    } else if (arg == "--mode") {
      options.mode = require_value(arg);
    } else if (arg == "--bytes") {
      options.bytes = parse_size(require_value(arg));
    } else if (arg == "--warmup-sec") {
      options.warmup_sec = parse_seconds(require_value(arg), arg);
    } else if (arg == "--steady-sec") {
      options.steady_sec = parse_seconds(require_value(arg), arg);
    } else if (arg == "--presweep-bytes") {
      options.presweep_bytes = parse_size_list(require_value(arg));
    } else if (arg == "--presweep-steady-sec") {
      options.presweep_steady_sec = parse_seconds(require_value(arg), arg);
    } else if (arg == "--help") {
      std::cout
          << "Usage: h2d_d2d_copy --device 0 --mode h2d|d2d --bytes N "
          << "--warmup-sec S --steady-sec S [--presweep-bytes a,b,c]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.mode != "h2d" && options.mode != "d2d") {
    throw std::runtime_error("--mode must be h2d or d2d");
  }
  if (options.steady_sec <= 0.0) {
    throw std::runtime_error("--steady-sec must be greater than zero");
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

using CopyFn = void (*)(cudaStream_t, std::size_t);
constexpr int kBatchCopies = 16;

void run_for_seconds(cudaStream_t stream, CopyFn copy, std::size_t bytes, double seconds) {
  if (seconds <= 0.0) {
    return;
  }
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    for (int index = 0; index < kBatchCopies; ++index) {
      copy(stream, bytes);
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup batch)");
  }
}

Result measure_copy(
    const std::string& mode,
    const std::string& phase,
    std::size_t bytes,
    double warmup_sec,
    double steady_sec,
    CopyFn copy) {
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

  run_for_seconds(stream, copy, bytes, warmup_sec);

  EventPair events;
  std::uint64_t iterations = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(steady_sec);

  check(cudaEventRecord(events.start(), stream), "cudaEventRecord(start)");
  while (std::chrono::steady_clock::now() < deadline) {
    for (int index = 0; index < kBatchCopies; ++index) {
      copy(stream, bytes);
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
  const double total_bytes = static_cast<double>(bytes) * static_cast<double>(iterations);
  return Result{
      mode,
      phase,
      bytes,
      steady_sec,
      static_cast<double>(elapsed_ms),
      iterations,
      total_bytes / elapsed_s / 1.0e9,
  };
}

void print_json(const Result& result) {
  std::cout << "{"
            << "\"mode\":\"" << result.mode << "\","
            << "\"phase\":\"" << result.phase << "\","
            << "\"bytes\":" << result.bytes << ","
            << "\"steady_sec\":" << result.steady_sec << ","
            << "\"elapsed_ms\":" << result.elapsed_ms << ","
            << "\"iterations\":" << result.iterations << ","
            << "\"bandwidth_gbps\":" << result.bandwidth_gbps
            << "}" << std::endl;
}

char* h2d_host = nullptr;
char* h2d_device = nullptr;
char* d2d_src = nullptr;
char* d2d_dst = nullptr;

void copy_h2d(cudaStream_t stream, std::size_t bytes) {
  check(cudaMemcpyAsync(
            h2d_device, h2d_host, bytes, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(H2D)");
}

void copy_d2d(cudaStream_t stream, std::size_t bytes) {
  check(cudaMemcpyAsync(d2d_dst, d2d_src, bytes, cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync(D2D)");
}

std::size_t max_bytes(const Options& options) {
  std::size_t maximum = options.bytes;
  for (std::size_t value : options.presweep_bytes) {
    maximum = std::max(maximum, value);
  }
  return maximum;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_args(argc, argv);
    check(cudaSetDevice(options.device), "cudaSetDevice");
    const std::size_t allocation_bytes = max_bytes(options);

    if (options.mode == "h2d") {
      check(cudaHostAlloc(
                reinterpret_cast<void**>(&h2d_host),
                allocation_bytes,
                cudaHostAllocDefault),
            "cudaHostAlloc");
      check(cudaMalloc(reinterpret_cast<void**>(&h2d_device), allocation_bytes),
            "cudaMalloc(h2d_device)");
      std::memset(h2d_host, 0x5a, allocation_bytes);
    } else {
      check(cudaMalloc(reinterpret_cast<void**>(&d2d_src), allocation_bytes),
            "cudaMalloc(d2d_src)");
      check(cudaMalloc(reinterpret_cast<void**>(&d2d_dst), allocation_bytes),
            "cudaMalloc(d2d_dst)");
      check(cudaMemset(d2d_src, 0x5a, allocation_bytes), "cudaMemset(d2d_src)");
      check(cudaMemset(d2d_dst, 0x00, allocation_bytes), "cudaMemset(d2d_dst)");
    }

    CopyFn copy = options.mode == "h2d" ? copy_h2d : copy_d2d;
    for (std::size_t bytes : options.presweep_bytes) {
      print_json(measure_copy(
          options.mode,
          "presweep",
          bytes,
          std::min(options.warmup_sec, 1.0),
          options.presweep_steady_sec,
          copy));
    }
    print_json(measure_copy(
        options.mode,
        "steady",
        options.bytes,
        options.warmup_sec,
        options.steady_sec,
        copy));

    if (h2d_host) {
      cudaFreeHost(h2d_host);
    }
    if (h2d_device) {
      cudaFree(h2d_device);
    }
    if (d2d_src) {
      cudaFree(d2d_src);
    }
    if (d2d_dst) {
      cudaFree(d2d_dst);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"error\":\"" << error.what() << "\"}" << std::endl;
    return 1;
  }
}
