#pragma once
// Load-aware width governor for pipelined SSD search.
// PIPEANN_QSTAR (Q*): device queue depth at which throughput saturates. When set,
// the in-flight width of every active query is capped at max(PIPEANN_WMIN, Q*/n_active),
// so active queries share the device queue instead of each keeping up to W reads
// outstanding. Unset = stock PipeANN behaviour.
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace pipeann {
  struct IOGovernor {
    alignas(64) std::atomic<int64_t> active{0};
    int64_t qstar = 0, wmin = 2;
    IOGovernor() {
      if (const char *q = std::getenv("PIPEANN_QSTAR")) qstar = std::atoll(q);
      if (const char *w = std::getenv("PIPEANN_WMIN")) wmin = std::atoll(w);
    }
    int64_t cap(int64_t w) const {
      if (qstar <= 0) return w;
      int64_t n = active.load(std::memory_order_relaxed);
      if (n < 1) n = 1;
      int64_t c = qstar / n;
      if (c < wmin) c = wmin;
      return c < w ? c : w;
    }
  };
  inline IOGovernor io_gov;
}  // namespace pipeann
