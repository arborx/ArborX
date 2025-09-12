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
#include <ArborX_MinimumSpanningTree.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <iostream>
#include <sstream>
#include <vector>

#include "data.hpp"
#include "parameters.hpp"
#include "print_timers.hpp"

template <typename ExecutionSpace, typename Primitives>
void run_mst(ExecutionSpace const &exec_space, Primitives const &primitives,
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

  Kokkos::Profiling::pushRegion("ArborX::MST::total");
  ArborX::Experimental::MinimumSpanningTree<MemorySpace> mst(
      exec_space, primitives, params.core_min_size);
  Kokkos::Profiling::popRegion();

  if (!params.verbose)
    return;

  printf("-- construction     : %10.3f\n",
         ArborXBenchmark::get_time("ArborX::MST::construction"));
  if (params.core_min_size > 1)
    printf("-- core distances   : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::MST::compute_core_distances"));
  printf("-- boruvka          : %10.3f\n",
         ArborXBenchmark::get_time("ArborX::MST::boruvka"));
  printf("total time          : %10.3f\n",
         ArborXBenchmark::get_time("ArborX::MST::total"));
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

  std::vector<std::string> allowed_dendrograms = {"boruvka", "union-find"};

  bpo::options_description desc("Allowed options");
  bool ascii;
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "ascii", bpo::bool_switch(&ascii), "ascii file indicator")
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "dimension", bpo::value<int>(&params.dim)->default_value(-1), "dimension of points to generate" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "max-num-points", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
      ( "n", bpo::value<int>(&params.n)->default_value(10), "number of points to generate" )
      ( "samples", bpo::value<int>(&params.num_samples)->default_value(-1), "number of samples" )
      ( "variable-density", bpo::bool_switch(&params.variable_density), "type of cluster density to generate" )
      ( "verbose", bpo::bool_switch(&params.verbose), "verbose")
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

  // Print out the runtime parameters
  printf("minpts            : %d\n", params.core_min_size);
  printf("verbose           : %s\n", (params.verbose ? "true" : "false"));

  ExecutionSpace exec_space;

  int dim =
      (params.filename.empty()
           ? params.dim
           : ArborXBenchmark::getDataDimension(params.filename, params.binary));
#define SWITCH_DIM(DIM)                                                        \
  case DIM:                                                                    \
    run_mst(exec_space, ArborXBenchmark::loadData<DIM, MemorySpace>(params),   \
            params);                                                           \
    break;
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

  return 0;
}
