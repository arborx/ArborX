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
#include <ArborX_HDBSCAN.hpp>
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
void run_hdbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
                 ArborXBenchmark::Parameters const &params)
{
  if (params.verbose)
  {
    Kokkos::Profiling::Experimental::set_push_region_callback(
        ArborXBenchmark::push_region);
    Kokkos::Profiling::Experimental::set_pop_region_callback(
        ArborXBenchmark::pop_region);
  }

  using ArborX::Experimental::DendrogramImplementation;
  DendrogramImplementation dendrogram_impl;
  if (params.dendrogram == "union-find")
    dendrogram_impl = DendrogramImplementation::UNION_FIND;
  else if (params.dendrogram == "boruvka")
    dendrogram_impl = DendrogramImplementation::BORUVKA;
  else
  {
    auto error_string = "Unknown dendogram: \"" + params.dendrogram + "\"";
    Kokkos::abort(error_string.c_str());
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::total");
  auto dendrogram = ArborX::Experimental::hdbscan(
      exec_space, primitives, params.core_min_size, dendrogram_impl);
  Kokkos::Profiling::popRegion();

  if (!params.verbose)
    return;

  if (params.dendrogram == "boruvka")
  {
    printf("-- construction     : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::MST::construction"));
    if (params.core_min_size > 1)
      printf("-- core distances   : %10.3f\n",
             ArborXBenchmark::get_time("ArborX::MST::compute_core_distances"));
    printf("-- boruvka          : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::MST::boruvka"));
    printf("---- sided parents  : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::MST::update_sided_parents"));
    printf("---- vertex parents : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::MST::compute_vertex_parents"));
    printf("-- edge parents     : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::MST::compute_edge_parents"));
  }
  else
  {
    printf("-- mst              : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::HDBSCAN::mst"));
    printf("-- dendrogram       : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::HDBSCAN::dendrogram"));
    printf("---- edge sort      : %10.3f\n",
           ArborXBenchmark::get_time("ArborX::Dendrogram::sort_edges"));
  }
  printf("total time          : %10.3f\n",
         ArborXBenchmark::get_time("ArborX::HDBSCAN::total"));
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

  std::vector<std::string> allowed_dendrograms = {"boruvka", "union-find"};

  bpo::options_description desc("Allowed options");
  bool ascii;
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "ascii", bpo::bool_switch(&ascii), "ascii file indicator")
      ( "core-min-size", bpo::value<int>(&params.core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "dendrogram", bpo::value<std::string>(&params.dendrogram)->default_value("boruvka"), ("dendrogram " + vec2string(allowed_dendrograms, " | ")).c_str() )
      ( "dimension", bpo::value<int>(&params.dim)->default_value(-1), "dimension of points to generate" )
      ( "filename", bpo::value<std::string>(&params.filename), "filename containing data" )
      ( "max-num-points", bpo::value<int>(&params.max_num_points)->default_value(-1), "max number of points to read in")
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

  auto found = [](auto const &v, auto x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };

  if (!found(allowed_dendrograms, params.dendrogram))
  {
    std::cerr << "Dendrogram must be one of " << vec2string(allowed_dendrograms)
              << "\n";
    return 4;
  }

  // Print out the runtime parameters
  printf("dendrogram        : %s\n", params.dendrogram.c_str());
  printf("minpts            : %d\n", params.core_min_size);
  printf("verbose           : %s\n", (params.verbose ? "true" : "false"));

  ExecutionSpace exec_space;

  int dim =
      (params.filename.empty()
           ? params.dim
           : ArborXBenchmark::getDataDimension(params.filename, params.binary));
#define SWITCH_DIM(DIM)                                                        \
  case DIM:                                                                    \
    run_hdbscan(exec_space,                                                    \
                ArborXBenchmark::loadData<DIM, MemorySpace>(params), params);  \
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
