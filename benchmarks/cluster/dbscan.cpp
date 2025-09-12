/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_DBSCANVerification.hpp"
#include <ArborX_DBSCAN.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "data.hpp"
#include "parameters.hpp"
#include "print_timers.hpp"

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

  namespace KokkosExt = ArborX::Details::KokkosExt;

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
      Kokkos::RangePolicy(exec_space, 0, n), KOKKOS_LAMBDA(int const i) {
        // Ignore noise points
        if (labels(i) < 0)
          return;

        Kokkos::atomic_inc(&cluster_sizes(labels(i)));
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
      Kokkos::RangePolicy(exec_space, 0, n),
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
  KokkosExt::exclusive_scan(exec_space, cluster_offset, cluster_offset, 0);

  auto cluster_starts = KokkosExt::clone(exec_space, cluster_offset);
  KokkosExt::reallocWithoutInitializing(
      exec_space, cluster_indices,
      KokkosExt::lastElement(exec_space, cluster_offset));
  Kokkos::parallel_for(
      "ArborX::DBSCAN::compute_cluster_indices",
      Kokkos::RangePolicy(exec_space, 0, n), KOKKOS_LAMBDA(int const i) {
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

template <typename ExecutionSpace, typename Primitives>
bool run_dbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
                ArborXBenchmark::Parameters const &params)
{
  using MemorySpace = typename Primitives::memory_space;

  if (params.verbose)
  {
    Kokkos::Profiling::Experimental::set_push_region_callback(
        ArborXBenchmark::push_region);
    Kokkos::Profiling::Experimental::set_pop_region_callback(
        ArborXBenchmark::pop_region);
  }

  using ArborX::DBSCAN::Implementation;
  Implementation implementation = Implementation::FDBSCAN;
  if (params.implementation == "fdbscan-densebox")
    implementation = Implementation::FDBSCAN_DenseBox;

  ArborX::DBSCAN::Parameters dbscan_params;
  dbscan_params.setVerbosity(params.verbose).setImplementation(implementation);

  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::total");

  Kokkos::View<int *, MemorySpace> labels("Example::labels", 0);
  labels = ArborX::dbscan<ExecutionSpace>(exec_space, primitives, params.eps,
                                          params.core_min_size, dbscan_params);

  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::postprocess");
  Kokkos::View<int *, MemorySpace> cluster_indices("Testing::cluster_indices",
                                                   0);
  Kokkos::View<int *, MemorySpace> cluster_offset("Testing::cluster_offset", 0);
  sortAndFilterClusters(exec_space, labels, cluster_indices, cluster_offset,
                        params.cluster_min_size);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  if (params.verbose)
  {
    bool const is_special_case = (params.core_min_size == 2);

    if (implementation == ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox)
      printf("-- dense cells      : %10.3f\n",
             ArborXBenchmark::get_time("ArborX::DBSCAN::dense_cells"));
    printf("-- construction     : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::DBSCAN::tree_construction"));
    printf("-- query+cluster    : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::DBSCAN::clusters"));
    if (!is_special_case)
    {
      printf("---- neigh          : %10.3f\n",
             ArborXBenchmark::get_time("ArborX::DBSCAN::clusters::num_neigh"));
      printf("---- query          : %10.3f\n",
             ArborXBenchmark::get_time("ArborX::DBSCAN::clusters::query"));
    }
    printf("-- postprocess      : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::DBSCAN::postprocess"));
    printf("total time          : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::DBSCAN::total"));
  }

  int num_points = primitives.extent_int(0);
  int num_clusters = cluster_offset.size() - 1;
  int num_cluster_points = cluster_indices.size();
  printf("\n#clusters       : %d\n", num_clusters);
  printf("#cluster points : %d [%.2f%%]\n", num_cluster_points,
         (100.f * num_cluster_points / num_points));
  int num_noise_points = num_points - num_cluster_points;
  printf("#noise   points : %d [%.2f%%]\n", num_noise_points,
         (100.f * num_noise_points / num_points));

  bool success = true;
  if (params.verify)
  {
    success = ArborX::Details::verifyDBSCAN(exec_space, primitives, params.eps,
                                            params.core_min_size, labels);
    printf("Verification %s\n", (success ? "passed" : "failed"));
  }

  if (success && !params.filename_labels.empty())
    writeLabelsData(params.filename_labels, labels);

  return success;
}

template <typename T>
std::string vec2string(std::vector<T> const &s, std::string const &delim = ", ")
{
  assert(s.size() > 1);

  std::ostringstream ss;
  std::copy(s.begin(), s.end(),
            std::ostream_iterator<std::string>{ss, delim.c_str()});
  auto delimited_items = ss.str().erase(ss.str().length() - delim.size());
  return "(" + delimited_items + ")";
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
            << std::endl;

  namespace bpo = boost::program_options;
  using namespace ArborXBenchmark;

  Parameters params;

  std::vector<std::string> allowed_impls = {"fdbscan", "fdbscan-densebox"};

  bpo::options_description desc("Allowed options");
  bool ascii;
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "ascii", bpo::bool_switch(&ascii), "ascii file indicator")
      ( "cluster-min-size", bpo::value<int>(&params.cluster_min_size)->default_value(1), "minimum cluster size")
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "dimension", bpo::value<int>(&params.dim)->default_value(-1), "dimension of points to generate" )
      ( "eps", bpo::value<float>(&params.eps), "DBSCAN eps" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "impl", bpo::value<std::string>(&params.implementation)->default_value("fdbscan"), ("implementation " + vec2string(allowed_impls, " | ")).c_str() )
      ( "labels", bpo::value<std::string>(&params.filename_labels)->default_value(""), "clutering results output" )
      ( "max-num-points", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
      ( "n", bpo::value<int>(&params.n)->default_value(10), "number of points to generate" )
      ( "samples", bpo::value<int>(&params.num_samples)->default_value(-1), "number of samples" )
      ( "variable-density", bpo::bool_switch(&params.variable_density), "type of cluster density to generate" )
      ( "verbose", bpo::bool_switch(&params.verbose), "verbose")
      ( "verify", bpo::bool_switch(&params.verify), "verify connected components")
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  params.binary = !ascii;

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    std::cout << "[Generator Help]\n"
                 "If using generator, the recommended DBSCAN parameters are:\n"
                 "- core-min-size = 10\n"
                 "- eps = 60 (2D constant), 100 (2D variable), 200 (3D "
                 "constant), 400 (3D variable)"
              << std::endl;
    return 1;
  }

  auto found = [](auto const &v, auto x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };

  if (!found(allowed_impls, params.implementation))
  {
    std::cerr << "Implementation must be one of " << vec2string(allowed_impls)
              << "\n";
    return 2;
  }

  // Print out the runtime parameters
  std::stringstream ss;
  ss << params.implementation;
  printf("eps               : %f\n", params.eps);
  printf("minpts            : %d\n", params.core_min_size);
  printf("cluster min size  : %d\n", params.cluster_min_size);
  if (!params.filename_labels.empty())
    printf("filename [labels] : %s [binary]\n", params.filename_labels.c_str());
  printf("implementation    : %s\n", ss.str().c_str());
  printf("verify            : %s\n", (params.verify ? "true" : "false"));
  printf("verbose           : %s\n", (params.verbose ? "true" : "false"));

  ExecutionSpace exec_space;

  int dim =
      (params.filename.empty()
           ? params.dim
           : ArborXBenchmark::getDataDimension(params.filename, params.binary));
#define SWITCH_DIM(DIM)                                                        \
  case DIM:                                                                    \
    success = run_dbscan(exec_space,                                           \
                         ArborXBenchmark::loadData<DIM, MemorySpace>(params),  \
                         params);                                              \
    break;
  bool success = true;
  switch (dim)
  {
    SWITCH_DIM(2)
    SWITCH_DIM(3)
    SWITCH_DIM(4)
    SWITCH_DIM(5)
    SWITCH_DIM(6)
  default:
    std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
  }
#undef SWITCH_DIM

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
