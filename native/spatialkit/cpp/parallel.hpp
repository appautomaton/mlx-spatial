#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace mlx_spatialkit {

inline size_t native_worker_count(size_t item_count, size_t minimum_items_per_worker = 2048) {
  if (item_count == 0) {
    return 1;
  }
  const size_t hardware_workers = std::max<size_t>(1, std::thread::hardware_concurrency());
  const size_t useful_workers = std::max<size_t>(1, item_count / minimum_items_per_worker);
  return std::min(hardware_workers, useful_workers);
}

template <typename Function>
void parallel_for(
    size_t item_count,
    Function &&function,
    size_t minimum_items_per_worker = 2048,
    size_t chunk_size = 256) {
  const size_t worker_count = native_worker_count(item_count, minimum_items_per_worker);
  if (worker_count <= 1) {
    function(0, item_count);
    return;
  }

  std::atomic<size_t> next{0};
  std::atomic<bool> stopped{false};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      try {
        while (!stopped.load(std::memory_order_relaxed)) {
          const size_t begin = next.fetch_add(chunk_size, std::memory_order_relaxed);
          if (begin >= item_count) {
            break;
          }
          function(begin, std::min(item_count, begin + chunk_size));
        }
      } catch (...) {
        stopped.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(failure_mutex);
        if (failure == nullptr) {
          failure = std::current_exception();
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

}  // namespace mlx_spatialkit
