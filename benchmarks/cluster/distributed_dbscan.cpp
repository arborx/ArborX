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
#include <ArborX_DistributedDBSCAN.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "data.hpp"
#include "distributed_data.hpp"
#include "parameters.hpp"
#include "print_timers.hpp"
#include <mpi.h>

template <typename ExecutionSpace, typename Primitives>
bool run_dist_dbscan(MPI_Comm comm, ExecutionSpace const &exec_space,
                     Primitives const &primitives,
                     ArborXBenchmark::Parameters const &params)
{
  using MemorySpace = typename Primitives::memory_space;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

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

  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<
      typename Primitives::value_type>;

  Kokkos::Profiling::pushRegion("ArborX::DistributedDBSCAN::total");
  Kokkos::View<long long *, MemorySpace> labels("Example::labels", 0);
  ArborX::Experimental::dbscan(comm, exec_space, primitives,
                               (Coordinate)params.eps, params.core_min_size,
                               labels, dbscan_params);
  Kokkos::Profiling::popRegion();

  if (params.verbose && comm_rank == 0)
  {
    printf("total time          : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::DistributedDBSCAN::total"));
  }

  bool success = true;
  if (params.verify)
  {
    success = ArborX::Details::verifyDBSCAN(comm, exec_space, primitives,
                                            params.eps, params.core_min_size,
                                            labels, params.verbose);
    if (comm_rank == 0)
      printf("Verification %s\n", (success ? "passed" : "failed"));
  }

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
  MPI_Init(&argc, &argv);

  MPI_Comm const comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  if (comm_rank == 0)
  {
    std::cout << "ArborX version    : " << ArborX::version() << std::endl;
    std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
    std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
              << std::endl;
    std::cout << "#MPI ranks         : " << comm_size << std::endl;
  }

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we
  // are not on MPI rank 0 to prevent Kokkos from printing the help message
  // multiply.
  auto *help_it = std::find_if(argv, argv + argc, [](std::string const &x) {
    return x == "--help" || x == "--kokkos-help";
  });
  bool is_help_present = (help_it != argv + argc);
  if (is_help_present && comm_rank != 0)
  {
    std::swap(*help_it, *(argv + argc - 1));
    --argc;
  }

  Kokkos::initialize(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  namespace bpo = boost::program_options;
  using namespace ArborXBenchmark;

  Parameters params;
  params.binary = true;
  params.num_samples = -1;

  std::vector<std::string> allowed_impls = {"fdbscan", "fdbscan-densebox"};
  std::vector<std::string> allowed_precisions = {"float", "double"};

  bpo::options_description desc("Allowed options");
  std::string precision;
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "dimension", bpo::value<int>(&params.dim)->default_value(-1), "dimension of points to generate" )
      ( "eps", bpo::value<float>(&params.eps), "DBSCAN eps" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "impl", bpo::value<std::string>(&params.implementation)->default_value("fdbscan"), ("implementation " + vec2string(allowed_impls, " | ")).c_str() )
      ( "max-num-points", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
      ( "n", bpo::value<int>(&params.n)->default_value(10), "number of points to generate per rank" )
      ( "num-seq", bpo::value<int>(&params.n_seq)->default_value(3), "number of points in sequence" )
      ( "precision", bpo::value<std::string>(&precision)->default_value("float"), "data precision" )
      ( "spacing", bpo::value<int>(&params.spacing)->default_value(2), "spacing size" )
      ( "verbose", bpo::bool_switch(&params.verbose), "verbose")
      ( "verify", bpo::bool_switch(&params.verify), "verify connected components")
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  ExecutionSpace exec_space;

  if (is_help_present)
  {
    if (comm_rank == 0)
    {
      std::cout << desc << '\n';
      std::cout << "[Generator Help]\n"
                   "If using generator, the distance between closest points\n"
                   "is 1. Use eps accordingly. If eps is larger than spacing,\n"
                   "all the clusters will be merged together.\n"
                << std::endl;
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
  }

  auto found = [](auto const &v, auto x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };

  if (!found(allowed_impls, params.implementation))
  {
    if (comm_rank == 0)
      std::cerr << "Implementation must be one of " << vec2string(allowed_impls)
                << "\n";
    Kokkos::finalize();
    MPI_Finalize();
    return 2;
  }
  if (!found(allowed_precisions, precision))
  {
    if (comm_rank == 0)
      std::cerr << "Precision must be one of " << vec2string(allowed_precisions)
                << "\n";
    Kokkos::finalize();
    MPI_Finalize();
    return 2;
  }
  if (!params.filename.empty() && precision != "float")
  {
    if (comm_rank == 0)
      std::cerr << "Data loading only supports \"float\"\n";
    Kokkos::finalize();
    MPI_Finalize();
    return 3;
  }

  int dim = (params.filename.empty()
                 ? params.dim
                 : getDataDimension(params.filename, params.binary));
  if (dim != 2 && dim != 3)
  {
    if (comm_rank == 0)
      std::cerr << "Error: dimension " << dim << " not allowed\n" << std::endl;
    Kokkos::finalize();
    MPI_Finalize();
    return 4;
  }

  if (comm_rank == 0)
  {
    // Print out the runtime parameters
    std::stringstream ss;
    ss << params.implementation;
    printf("eps               : %f\n", params.eps);
    printf("minpts            : %d\n", params.core_min_size);
    printf("implementation    : %s\n", ss.str().c_str());
    if (params.filename.empty())
    {
      printf("n                 : %d\n", params.n);
      printf("n_seq             : %d\n", params.n_seq);
      printf("precision         : %s\n", precision.c_str());
      printf("spacing           : %d\n", params.spacing);
    }
    printf("verify            : %s\n", (params.verify ? "true" : "false"));
    printf("verbose           : %s\n", (params.verbose ? "true" : "false"));
  }

  MPI_Barrier(comm);

  bool success = true;
  if (!params.filename.empty())
  {
#define SWITCH_DIM(DIM)                                                        \
  case DIM:                                                                    \
    success = run_dist_dbscan(                                                 \
        comm, exec_space, loadData<DIM, MemorySpace>(comm, params), params);   \
    break;

    switch (dim)
    {
      SWITCH_DIM(2)
      SWITCH_DIM(3)
    }
#undef SWITCH_DIM
  }
  else
  {
    int dim = params.dim;
#define SWITCH_DIM(DIM, TYPE)                                                  \
  case DIM:                                                                    \
    success = run_dist_dbscan(                                                 \
        comm, exec_space,                                                      \
        generateDistributedData<DIM, TYPE, MemorySpace>(comm, params),         \
        params);                                                               \
    break;

    if (precision == "float")
    {
      switch (dim)
      {
        SWITCH_DIM(2, float)
        SWITCH_DIM(3, float)
      }
    }
    else if (precision == "double")
    {
      switch (dim)
      {
        SWITCH_DIM(2, double)
        SWITCH_DIM(3, double)
      }
    }
#undef SWITCH_DIM
  }

  Kokkos::finalize();
  MPI_Finalize();

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
