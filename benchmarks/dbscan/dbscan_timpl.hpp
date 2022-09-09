/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DBSCAN.hpp>
#include <ArborX_DBSCANVerification.hpp>
#include <ArborX_MinimumSpanningTree.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <fstream>

#include "dbscan.hpp"

using ArborX::ExperimentalHyperGeometry::Point;

template <int DIM>
std::vector<Point<DIM>> sampleData(std::vector<Point<DIM>> const &data,
                                   int num_samples)
{
  std::vector<Point<DIM>> sampled_data(num_samples);

  std::srand(1337);

  // Knuth algorithm
  auto const N = (int)data.size();
  auto const M = num_samples;
  for (int in = 0, im = 0; in < N && im < M; ++in)
  {
    int rn = N - in;
    int rm = M - im;
    if (std::rand() % rn < rm)
      sampled_data[im++] = data[in];
  }
  return sampled_data;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

template <int DIM>
std::vector<Point<DIM>> loadData(std::string const &filename,
                                 bool binary = true, int max_num_points = -1,
                                 int num_samples = -1)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  std::vector<Point<DIM>> v;

  int num_points = 0;
  int dim = 0;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }

  ARBORX_ASSERT(dim == DIM);

  if (max_num_points > 0 && max_num_points < num_points)
    num_points = max_num_points;

  if (!binary)
  {
    v.reserve(num_points);

    auto it = std::istream_iterator<float>(input);
    for (int i = 0; i < num_points; ++i)
      for (int d = 0; d < DIM; ++d)
        v[i][d] = *it++;
  }
  else
  {
    // Directly read into a point
    v.resize(num_points);
    input.read(reinterpret_cast<char *>(v.data()),
               num_points * sizeof(Point<DIM>));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  if (num_samples > 0 && num_samples < (int)v.size())
    v = sampleData(v, num_samples);

  return v;
}

template <typename MemorySpace>
void writeLabelsData(std::string const &filename,
                     Kokkos::View<int *, MemorySpace> labels)
{
  std::ofstream out(filename, std::ofstream::binary);
  ARBORX_ASSERT(out.good());

  auto labels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);

  int n = labels_host.size();
  out.write((char *)&n, sizeof(int));
  out.write((char *)labels_host.data(), sizeof(int) * n);
}

template <typename ExecutionSpace, typename LabelsView,
          typename ClusterIndicesView, typename ClusterOffsetView>
void sortAndFilterClusters(ExecutionSpace const &exec_space,
                           LabelsView const &labels,
                           ClusterIndicesView &cluster_indices,
                           ClusterOffsetView &cluster_offset,
                           int cluster_min_size = 1)
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::sortAndFilterClusters");

  static_assert(Kokkos::is_view<LabelsView>{});
  static_assert(Kokkos::is_view<ClusterIndicesView>{});
  static_assert(Kokkos::is_view<ClusterOffsetView>{});

  using MemorySpace = typename LabelsView::memory_space;

  static_assert(std::is_same<typename LabelsView::value_type, int>{});
  static_assert(std::is_same<typename ClusterIndicesView::value_type, int>{});
  static_assert(std::is_same<typename ClusterOffsetView::value_type, int>{});

  static_assert(std::is_same<typename LabelsView::memory_space, MemorySpace>{});
  static_assert(
      std::is_same<typename ClusterIndicesView::memory_space, MemorySpace>{});
  static_assert(
      std::is_same<typename ClusterOffsetView::memory_space, MemorySpace>{});

  ARBORX_ASSERT(cluster_min_size >= 1);

  int const n = labels.extent_int(0);

  Kokkos::View<int *, MemorySpace> cluster_sizes(
      "ArborX::DBSCAN::cluster_sizes", n);
  Kokkos::parallel_for(
      "ArborX::DBSCAN::compute_cluster_sizes",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        // Ignore noise points
        if (labels(i) < 0)
          return;

        Kokkos::atomic_increment(&cluster_sizes(labels(i)));
      });

  // This kernel serves dual purpose:
  // - it constructs an offset array through exclusive prefix sum, with a
  //   caveat that small clusters (of size < cluster_min_size) are filtered out
  // - it creates a mapping from a cluster index into the cluster's position in
  //   the offset array
  // We reuse the cluster_sizes array for the second, creating a new alias for
  // it for clarity.
  auto &map_cluster_to_offset_position = cluster_sizes;
  constexpr int IGNORED_CLUSTER = -1;
  int num_clusters;
  KokkosExt::reallocWithoutInitializing(exec_space, cluster_offset, n + 1);
  Kokkos::parallel_scan(
      "ArborX::DBSCAN::compute_cluster_offset_with_filter",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i, int &update, bool final_pass) {
        bool is_cluster_too_small = (cluster_sizes(i) < cluster_min_size);
        if (!is_cluster_too_small)
        {
          if (final_pass)
          {
            cluster_offset(update) = cluster_sizes(i);
            map_cluster_to_offset_position(i) = update;
          }
          ++update;
        }
        else
        {
          if (final_pass)
            map_cluster_to_offset_position(i) = IGNORED_CLUSTER;
        }
      },
      num_clusters);
  Kokkos::resize(Kokkos::WithoutInitializing, cluster_offset, num_clusters + 1);
  ArborX::exclusivePrefixSum(exec_space, cluster_offset);

  auto cluster_starts = KokkosExt::clone(exec_space, cluster_offset);
  KokkosExt::reallocWithoutInitializing(
      exec_space, cluster_indices,
      KokkosExt::lastElement(exec_space, cluster_offset));
  Kokkos::parallel_for(
      "ArborX::DBSCAN::compute_cluster_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        // Ignore noise points
        if (labels(i) < 0)
          return;

        auto offset_pos = map_cluster_to_offset_position(labels(i));
        if (offset_pos != IGNORED_CLUSTER)
        {
          auto position =
              Kokkos::atomic_fetch_add(&cluster_starts(offset_pos), 1);
          cluster_indices(position) = i;
        }
      });

  Kokkos::Profiling::popRegion();
}

template <int DIM>
bool ArborXBenchmark::run(ArborXBenchmark::Parameters const &params)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  auto data = loadData<DIM>(params.filename, params.binary,
                            params.max_num_points, params.num_samples);

  auto const primitives = vec2view<MemorySpace>(data, "primitives");

  using Primitives = decltype(primitives);

  ExecutionSpace exec_space;

  Kokkos::Timer timer_total;
  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  bool const verbose = params.print_dbscan_timers;
  auto timer_start = [&exec_space, verbose](Kokkos::Timer &timer) {
    if (verbose)
      exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space, verbose](Kokkos::Timer const &timer) {
    if (verbose)
      exec_space.fence();
    return timer.seconds();
  };

  timer_start(timer_total);

  Kokkos::View<int *, MemorySpace> labels("Example::labels", 0);
  bool success = true;
  if (params.algorithm == "dbscan")
  {
    ArborX::DBSCAN::Parameters dbscan_params;
    dbscan_params.setPrintTimers(params.print_dbscan_timers)
        .setImplementation(params.implementation);

    labels = ArborX::dbscan<ExecutionSpace, Primitives>(
        exec_space, primitives, params.eps, params.core_min_size,
        dbscan_params);

    timer_start(timer);
    Kokkos::View<int *, MemorySpace> cluster_indices("Testing::cluster_indices",
                                                     0);
    Kokkos::View<int *, MemorySpace> cluster_offset("Testing::cluster_offset",
                                                    0);
    sortAndFilterClusters(exec_space, labels, cluster_indices, cluster_offset,
                          params.cluster_min_size);
    elapsed["cluster"] = timer_seconds(timer);
    elapsed["total"] = timer_seconds(timer_total);

    printf("-- postprocess      : %10.3f\n", elapsed["cluster"]);
    printf("total time          : %10.3f\n", elapsed["total"]);

    int num_points = primitives.extent_int(0);
    int num_clusters = cluster_offset.size() - 1;
    int num_cluster_points = cluster_indices.size();
    printf("\n#clusters       : %d\n", num_clusters);
    printf("#cluster points : %d [%.2f%%]\n", num_cluster_points,
           (100.f * num_cluster_points / num_points));
    int num_noise_points = num_points - num_cluster_points;
    printf("#noise   points : %d [%.2f%%]\n", num_noise_points,
           (100.f * num_noise_points / num_points));

    if (params.verify)
    {
      success = ArborX::Details::verifyDBSCAN(
          exec_space, primitives, params.eps, params.core_min_size, labels);
      printf("Verification %s\n", (success ? "passed" : "failed"));
    }
  }
  else if (params.algorithm == "mst")
  {
    ArborX::Details::MinimumSpanningTree<MemorySpace> mst(
        exec_space, primitives, params.core_min_size);
  }

  if (success && !params.filename_labels.empty())
    writeLabelsData(params.filename_labels, labels);

  return success;
}
