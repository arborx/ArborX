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

template <typename ExecutionSpace, typename Labels>
std::pair<int, int> countClusters(ExecutionSpace const &space,
                                  Labels const &labels)
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::DBSCAN::countClusters");

  namespace KokkosExt = ArborX::Details::KokkosExt;

  static_assert(Kokkos::is_view_v<Labels>);
  static_assert(std::is_same_v<typename Labels::value_type, int>);

  using MemorySpace = typename Labels::memory_space;

  auto const n = labels.extent(0);
  Kokkos::View<int *, MemorySpace> cluster_mask("ArborX::DBSCAN::cluster_mask",
                                                n);
  int num_clusters;
  int num_cluster_points;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::compute_cluster_sizes", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int const i, int &update_num_clusters,
                    int &update_num_cluster_points) {
        // Ignore noise points
        if (labels(i) < 0)
          return;

        auto cluster_already_counted =
            Kokkos::atomic_compare_exchange(&cluster_mask(labels(i)), 0, 1);
        if (!cluster_already_counted)
          ++update_num_clusters;
        ++update_num_cluster_points;
      },
      num_clusters, num_cluster_points);

  return std::make_pair(num_clusters, num_cluster_points);
}

template <typename ExecutionSpace, typename Primitives>
bool run_dbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
                ArborXBenchmark::Parameters const &params)
{
  using MemorySpace = typename Primitives::memory_space;

  bool timer_hooks_set_up = false;
  if (params.verbose)
  {
    timer_hooks_set_up = ArborXBenchmark::try_set_timer_hooks();
    if (!timer_hooks_set_up)
      std::cerr << "\n\n\t*** Warning: Kokkos profiling tools are already "
                   "active, ignoring the the --verbose argument. ***\n\n";
  }

  using ArborX::DBSCAN::Implementation;
  Implementation implementation = Implementation::FDBSCAN;
  if (params.implementation == "fdbscan-densebox")
    implementation = Implementation::FDBSCAN_DenseBox;

  using ArborX::DBSCAN::Algorithm;
  Algorithm algorithm = Algorithm::DBSCAN;
  if (params.algorithm == "dbscan*")
    algorithm = Algorithm::DBSCAN_STAR;

  ArborX::DBSCAN::Parameters dbscan_params;
  dbscan_params.setVerbosity(params.verbose)
      .setImplementation(implementation)
      .setAlgorithm(algorithm);

  exec_space.fence();
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::total");

  Kokkos::View<int *, MemorySpace> labels("Example::labels", 0);
  ArborX::dbscan<ExecutionSpace>(exec_space, primitives, params.eps,
                                 params.core_min_size, labels, dbscan_params);

  exec_space.fence();
  Kokkos::Profiling::popRegion();

  auto [num_clusters, num_cluster_points] = countClusters(exec_space, labels);

  if (timer_hooks_set_up)
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
    printf("total time          : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::DBSCAN::total"));
  }

  int num_points = primitives.extent_int(0);
  printf("\n#clusters       : %d\n", num_clusters);
  printf("#cluster points : %d [%.2f%%]\n", num_cluster_points,
         (100.f * num_cluster_points / num_points));
  int num_noise_points = num_points - num_cluster_points;
  printf("#noise   points : %d [%.2f%%]\n", num_noise_points,
         (100.f * num_noise_points / num_points));

  bool success = true;
  if (params.verify)
  {
    success =
        ArborX::Details::verifyDBSCAN(exec_space, primitives, params.eps,
                                      params.core_min_size, labels, algorithm);
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
  std::vector<std::string> allowed_algorithms = {"dbscan", "dbscan*"};
  std::vector<std::string> allowed_precisions = {"float", "double"};

  bpo::options_description desc("Allowed options");
  bool ascii;
  std::string precision;
  // clang-format off
  desc.add_options()
      ( "help,h", "help message" )
      ( "algorithm,a", bpo::value<std::string>(&params.algorithm)->default_value("dbscan"), ("DBSCAN algorithm " + vec2string(allowed_algorithms, " | ")).c_str() )
      ( "ascii", bpo::bool_switch(&ascii), "ascii file indicator")
      ( "core-min-size,m", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "dimension,d", bpo::value<int>(&params.dim)->default_value(-1), "dimension of points to generate" )
      ( "eps,e", bpo::value<double>(&params.eps), "DBSCAN eps" )
      ( "filename,f", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "impl,i", bpo::value<std::string>(&params.implementation)->default_value("fdbscan"), ("implementation " + vec2string(allowed_impls, " | ")).c_str() )
      ( "labels", bpo::value<std::string>(&params.filename_labels)->default_value(""), "clustering results output" )
      ( "max-num-points,N", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
      ( "n,n", bpo::value<int>(&params.n)->default_value(10), "number of points to generate" )
      ( "precision,p", bpo::value<std::string>(&precision)->default_value("float"), "data precision" )
      ( "samples,s", bpo::value<int>(&params.num_samples)->default_value(-1), "number of samples" )
      ( "variable-density", bpo::bool_switch(&params.variable_density), "type of cluster density to generate" )
      ( "verbose,v", bpo::bool_switch(&params.verbose), "verbose")
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
  if (!found(allowed_algorithms, params.algorithm))
  {
    std::cerr << "Algorithm must be one of " << vec2string(allowed_algorithms)
              << "\n";
    return 3;
  }
  if (!found(allowed_precisions, precision))
  {
    std::cerr << "Precision must be one of " << vec2string(allowed_precisions)
              << "\n";
    return 4;
  }

  // Print out the runtime parameters
  printf("eps               : %f\n", params.eps);
  printf("minpts            : %d\n", params.core_min_size);
  if (!params.filename_labels.empty())
    printf("filename [labels] : %s [binary]\n", params.filename_labels.c_str());
  printf("algorithm         : %s\n", params.algorithm.c_str());
  printf("implementation    : %s\n", params.implementation.c_str());
  printf("precision         : %s\n", precision.c_str());
  printf("verify            : %s\n", (params.verify ? "true" : "false"));
  printf("verbose           : %s\n", (params.verbose ? "true" : "false"));

  ExecutionSpace exec_space;

  int dim =
      (params.filename.empty()
           ? params.dim
           : ArborXBenchmark::getDataDimension(params.filename, params.binary));
#define SWITCH_DIM(DIM, TYPE)                                                  \
  case DIM:                                                                    \
    success = run_dbscan(                                                      \
        exec_space, ArborXBenchmark::loadData<DIM, TYPE, MemorySpace>(params), \
        params);                                                               \
    break
  bool success = true;
  if (precision == "float")
  {
    switch (dim)
    {
      SWITCH_DIM(2, float);
      SWITCH_DIM(3, float);
      SWITCH_DIM(4, float);
      SWITCH_DIM(5, float);
      SWITCH_DIM(6, float);
    default:
      std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    }
  }
  else if (precision == "double")
  {
    switch (dim)
    {
      SWITCH_DIM(2, double);
      SWITCH_DIM(3, double);
      SWITCH_DIM(4, double);
      SWITCH_DIM(5, double);
      SWITCH_DIM(6, double);
    default:
      std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    }
  }
#undef SWITCH_DIM

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
