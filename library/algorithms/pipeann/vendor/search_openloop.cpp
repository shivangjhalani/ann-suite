// Open-loop (Poisson arrival) driver for SSD index search.
// Queries arrive at rate lambda; a pool of T workers serves them FIFO (M/G/T).
// Latency = completion - arrival (includes queueing).
#include <omp.h>
#include <ssd_index.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>
#include <algorithm>
#include "utils/log.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"

using clk = std::chrono::steady_clock;

static uint64_t g_rd_ticks = 0;
static void read_disk_stat(uint64_t &rd_ios, uint64_t &io_ticks) {
  std::ifstream f("/sys/block/nvme0n1/stat");
  uint64_t v[11] = {0};
  for (int i = 0; i < 11; i++) f >> v[i];
  rd_ios = v[0];
  g_rd_ticks = v[3];
  io_ticks = v[9];
}

static void read_cpu(uint64_t &busy, uint64_t &total) {
  std::ifstream f("/proc/stat");
  std::string c;
  uint64_t v[10] = {0};
  f >> c;
  for (int i = 0; i < 10; i++) f >> v[i];
  total = 0;
  for (int i = 0; i < 8; i++) total += v[i];
  busy = total - v[3] - v[4];
}

template<typename T>
int run(int argc, char **argv) {
  int a = 2;
  std::string prefix(argv[a++]);
  uint32_t nthreads = atoi(argv[a++]);
  uint32_t beamwidth = atoi(argv[a++]);
  std::string query_bin(argv[a++]);
  std::string gt_bin(argv[a++]);
  uint64_t K = atoi(argv[a++]);
  std::string metric(argv[a++]);
  std::string nbr_type(argv[a++]);
  int mode = atoi(argv[a++]);
  uint32_t mem_L = atoi(argv[a++]);
  uint64_t L = atoi(argv[a++]);
  double lambda = atof(argv[a++]);     // arrivals per second; 0 = closed loop (all arrive at t=0)
  uint64_t nq_total = atoll(argv[a++]); // number of queries to issue
  std::string out_file = argc > a ? argv[a++] : "";

  T *query = nullptr;
  size_t qn, qd;
  pipeann::load_bin<T>(query_bin, query, qn, qd);
  unsigned *gt_ids = nullptr;
  float *gt_d = nullptr;
  uint32_t *tags = nullptr;
  size_t gtn = 0, gtd = 0;
  bool has_gt = file_exists(gt_bin);
  if (has_gt) pipeann::load_truthset(gt_bin, gt_ids, gt_d, gtn, gtd, &tags);

  pipeann::Metric m = pipeann::get_metric(metric);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  auto *nbr = pipeann::get_nbr_handler<T>(m, nbr_type);
  pipeann::IndexBuildParameters p;
  p.max_nthreads = std::max<uint32_t>(16, 2 * nthreads);
  std::unique_ptr<pipeann::SSDIndex<T>> idx(new pipeann::SSDIndex<T>(m, reader, nbr, true, &p));
  if (idx->load(prefix.c_str(), false) != 0) return -1;
  if (mem_L) idx->load_mem_index(prefix + "_mem.index");

  // arrival schedule
  std::vector<double> arr(nq_total, 0.0);
  std::mt19937_64 rng(12345);
  if (lambda > 0) {
    std::exponential_distribution<double> ex(lambda);
    double t = 0;
    for (auto &x : arr) { t += ex(rng); x = t; }
  }
  std::vector<uint32_t> qid(nq_total);
  for (uint64_t i = 0; i < nq_total; i++) qid[i] = i % qn;

  std::vector<double> lat(nq_total), svc(nq_total), ios(nq_total);
  std::vector<uint32_t> res(nq_total * K);
  std::vector<float> dists(nq_total * K);
  std::atomic<uint64_t> next{0};

  uint64_t rd0, tk0, rd1, tk1;
  read_disk_stat(rd0, tk0);
  uint64_t rt0 = g_rd_ticks;
  uint64_t cb0, ct0, cb1, ct1;
  read_cpu(cb0, ct0);
  auto t0 = clk::now();
  auto worker = [&]() {
    while (true) {
      uint64_t i = next.fetch_add(1);
      if (i >= nq_total) break;
      auto at = t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(arr[i]));
      if (clk::now() < at) std::this_thread::sleep_until(at);
      auto s = clk::now();
      pipeann::QueryStats st;
      const T *q = query + (uint64_t) qid[i] * qd;
      if (mode == 2)
        idx->pipe_search(q, K, mem_L, L, res.data() + i * K, dists.data() + i * K, beamwidth, &st);
      else if (mode == 0)
        idx->beam_search(q, K, mem_L, L, res.data() + i * K, dists.data() + i * K, beamwidth, &st);
      else if (mode == 1)
        idx->page_search(q, K, mem_L, L, res.data() + i * K, dists.data() + i * K, beamwidth, &st);
      auto e = clk::now();
      lat[i] = std::chrono::duration<double, std::micro>(e - at).count();
      svc[i] = std::chrono::duration<double, std::micro>(e - s).count();
      ios[i] = st.n_ios;
    }
  };
  std::vector<std::thread> th;
  for (uint32_t t = 0; t < nthreads; t++) th.emplace_back(worker);
  for (auto &x : th) x.join();
  double wall = std::chrono::duration<double>(clk::now() - t0).count();
  read_disk_stat(rd1, tk1);
  double dev_lat_us = 1000.0 * (double) (g_rd_ticks - rt0) / (double) std::max<uint64_t>(1, rd1 - rd0);
  read_cpu(cb1, ct1);
  double cpu_util = (double) (cb1 - cb0) / (double) (ct1 - ct0);

  // recall over all issued queries
  double recall = 0;
  if (has_gt) {
    for (uint64_t i = 0; i < nq_total; i++) {
      uint32_t q = qid[i];
      std::set<uint32_t> g(gt_ids + (uint64_t) q * gtd, gt_ids + (uint64_t) q * gtd + K);
      uint32_t hit = 0;
      for (uint64_t j = 0; j < K; j++) hit += g.count(res[i * K + j]);
      recall += (double) hit / K;
    }
    recall /= nq_total;
  }
  // stats on the steady-state window: drop first 10%
  uint64_t w0 = nq_total / 10;
  std::vector<double> l(lat.begin() + w0, lat.end()), s(svc.begin() + w0, svc.end());
  std::sort(l.begin(), l.end());
  std::sort(s.begin(), s.end());
  auto pct = [](std::vector<double> &v, double p) { return v[std::min(v.size() - 1, (size_t) (p * v.size()))]; };
  double mean_l = 0, mean_s = 0, mean_io = 0;
  for (auto x : l) mean_l += x;
  for (auto x : s) mean_s += x;
  for (uint64_t i = w0; i < nq_total; i++) mean_io += ios[i];
  mean_l /= l.size();
  mean_s /= s.size();
  mean_io /= (nq_total - w0);
  double achieved = nq_total / wall;
  double dev_util = (tk1 - tk0) / (wall * 1000.0);
  double dev_iops = (rd1 - rd0) / wall;
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "RESULT mode=" << mode << " T=" << nthreads << " W=" << beamwidth << " L=" << L << " lambda=" << lambda
            << " achieved_qps=" << achieved << " recall=" << std::setprecision(4) << recall << std::setprecision(1)
            << " lat_mean=" << mean_l << " p50=" << pct(l, 0.5) << " p90=" << pct(l, 0.9) << " p99=" << pct(l, 0.99)
            << " p999=" << pct(l, 0.999) << " svc_mean=" << mean_s << " svc_p99=" << pct(s, 0.99)
            << " ios=" << mean_io << " dev_iops=" << dev_iops << " dev_util=" << std::setprecision(3) << dev_util << " cpu=" << cpu_util << " dev_lat=" << std::setprecision(1) << dev_lat_us
            << std::endl;
  if (!out_file.empty()) {
    std::ofstream o(out_file);
    for (uint64_t i = 0; i < nq_total; i++) o << arr[i] << " " << lat[i] << " " << svc[i] << " " << ios[i] << "\n";
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 15) {
    std::cout << "usage: type prefix T W query gt K metric nbr mode mem_L L lambda nq [out]" << std::endl;
    return -1;
  }
  std::string t(argv[1]);
  if (t == "float") return run<float>(argc, argv);
  if (t == "uint8") return run<uint8_t>(argc, argv);
  if (t == "int8") return run<int8_t>(argc, argv);
}
