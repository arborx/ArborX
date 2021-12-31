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

#include <ArborX_BoostRTreeHelpers.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cstdlib>

#include "benchmark_registration.hpp"

#ifdef ARBORX_PERFORMANCE_TESTING
#include <mpi.h>
#endif

#include <benchmark/benchmark.h>

template <typename ExecutionSpace, typename TreeType>
struct BenchmarkRegistration
{
  BenchmarkRegistration(Spec const &, std::string const &) {}
};

template <typename ExecutionSpace, typename MemorySpace>
struct BenchmarkRegistration<ExecutionSpace, ArborX::BVH<MemorySpace>>
{
  using TreeType = ArborX::BVH<MemorySpace>;
  BenchmarkRegistration(Spec const &spec, std::string const &description)
  {
    register_benchmark_construction<ExecutionSpace, TreeType>(spec,
                                                              description);
    register_benchmark_spatial_query_no_callback<ExecutionSpace, TreeType>(
        spec, description);
    register_benchmark_spatial_query_callback<ExecutionSpace, TreeType>(
        spec, description);
    register_benchmark_nearest_query_no_callback<ExecutionSpace, TreeType>(
        spec, description);
    register_benchmark_nearest_query_callback<ExecutionSpace, TreeType>(
        spec, description);
  }
};

template <typename ExecutionSpace>
struct BenchmarkRegistration<ExecutionSpace, BoostExt::RTree<ArborX::Point>>
{
  using TreeType = BoostExt::RTree<ArborX::Point>;
  BenchmarkRegistration(Spec const &spec, std::string const &description)
  {
    register_benchmark_construction<ExecutionSpace, TreeType>(spec,
                                                              description);
    register_benchmark_spatial_query_no_callback<ExecutionSpace, TreeType>(
        spec, description);
    register_benchmark_nearest_query_no_callback<ExecutionSpace, TreeType>(
        spec, description);
  }
};

template <typename ExecutionSpace>
using BVHBenchmarkRegistration =
    BenchmarkRegistration<ExecutionSpace,
                          ArborX::BVH<typename ExecutionSpace::memory_space>>;
void register_bvh_benchmarks(Spec const &spec)
{
#ifdef KOKKOS_ENABLE_SERIAL
  if (spec.backends == "all" || spec.backends == "serial")
    BVHBenchmarkRegistration<Kokkos::Serial>(spec, "ArborX::BVH<Serial>");
#else
  if (spec.backends == "serial")
    throw std::runtime_error("Serial backend not available!");
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  if (spec.backends == "all" || spec.backends == "openmp")
    BVHBenchmarkRegistration<Kokkos::OpenMP>(spec, "ArborX::BVH<OpenMP>");
#else
  if (spec.backends == "openmp")
    throw std::runtime_error("OpenMP backend not available!");
#endif

#ifdef KOKKOS_ENABLE_THREADS
  if (spec.backends == "all" || spec.backends == "threads")
    BVHBenchmarkRegistration<Kokkos::Threads>(spec, "ArborX::BVH<Threads>");
#else
  if (spec.backends == "threads")
    throw std::runtime_error("Threads backend not available!");
#endif

#ifdef KOKKOS_ENABLE_CUDA
  if (spec.backends == "all" || spec.backends == "cuda")
    BVHBenchmarkRegistration<Kokkos::Cuda>(spec, "ArborX::BVH<Cuda>");
#else
  if (spec.backends == "cuda")
    throw std::runtime_error("CUDA backend not available!");
#endif

#ifdef KOKKOS_ENABLE_HIP
  if (spec.backends == "all" || spec.backends == "hip")
    BVHBenchmarkRegistration<Kokkos::Experimental::HIP>(spec,
                                                        "ArborX::BVH<HIP>");
#else
  if (spec.backends == "hip")
    throw std::runtime_error("HIP backend not available!");
#endif

#ifdef KOKKOS_ENABLE_OPENMPTARGET
  if (spec.backends == "all" || spec.backends == "openmptarget")
    BVHBenchmarkRegistration<Kokkos::Experimental::OpenMPTarget>(
        spec, "ArborX::BVH<OpenMPTarget>");
#else
  if (spec.backends == "openmptarget")
    throw std::runtime_error("OpenMPTarget backend not available!");
#endif

#ifdef KOKKOS_ENABLE_SYCL
  if (spec.backends == "all" || spec.backends == "sycl")
    BVHBenchmarkRegistration<Kokkos::Experimental::SYCL>(spec,
                                                         "ArborX::BVH<SYCL>");
#else
  if (spec.backends == "sycl")
    throw std::runtime_error("SYCL backend not available!");
#endif
}

void register_boostrtree_benchmarks(Spec const &spec)
{
#ifdef KOKKOS_ENABLE_SERIAL
  if (spec.backends == "all" || spec.backends == "rtree")
    BenchmarkRegistration<Kokkos::Serial, BoostExt::RTree<ArborX::Point>>(
        spec, "BoostRTree");
#else
  std::ignore = spec;
#endif
}

// NOTE Motivation for this class that stores the argument count and values is
// I could not figure out how to make the parser consume arguments with
// Boost.Program_options
// Benchmark removes its own arguments from the command line arguments. This
// means, that by virtue of returning references to internal data members in
// argc() and argv() function, it will necessarily modify the members. It will
// decrease _argc, and "reduce" _argv data. Hence, we must keep a copy of _argv
// that is not modified from the outside to release memory in the destructor
// correctly.
class CmdLineArgs
{
private:
  int _argc;
  std::vector<char *> _argv;
  std::vector<char *> _owner_ptrs;

public:
  CmdLineArgs(std::vector<std::string> const &args, char const *exe)
      : _argc(args.size() + 1)
      , _owner_ptrs{new char[std::strlen(exe) + 1]}
  {
    std::strcpy(_owner_ptrs[0], exe);
    _owner_ptrs.reserve(_argc);
    for (auto const &s : args)
    {
      _owner_ptrs.push_back(new char[s.size() + 1]);
      std::strcpy(_owner_ptrs.back(), s.c_str());
    }
    _argv = _owner_ptrs;
  }

  ~CmdLineArgs()
  {
    for (auto *p : _owner_ptrs)
    {
      delete[] p;
    }
  }

  int &argc() { return _argc; }

  char **argv() { return _argv.data(); }
};

int main(int argc, char *argv[])
{
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  Spec single_spec;
  std::string source_pt_cloud;
  std::string target_pt_cloud;
  std::vector<std::string> exact_specs;
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&single_spec.n_values)->default_value(50000), "number of indexable values (source)" )
        ( "queries", bpo::value<int>(&single_spec.n_queries)->default_value(20000), "number of queries (target)" )
        ( "predicate-sort", bpo::value<bool>(&single_spec.sort_predicates)->default_value(true), "sort predicates" )
        ( "neighbors", bpo::value<int>(&single_spec.n_neighbors)->default_value(10), "desired number of results per query" )
        ( "buffer", bpo::value<int>(&single_spec.buffer_size)->default_value(0), "size for buffer optimization in radius search" )
        ( "source-point-cloud-type", bpo::value<std::string>(&source_pt_cloud)->default_value("filled_box"), "shape of the source point cloud"  )
        ( "target-point-cloud-type", bpo::value<std::string>(&target_pt_cloud)->default_value("filled_box"), "shape of the target point cloud"  )
        ( "exact-spec", bpo::value<std::vector<std::string>>(&exact_specs)->multitoken(), "exact specification (can be specified multiple times for batch)" )
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                   .options(desc)
                                   .allow_unregistered()
                                   .run();
  bpo::store(parsed, vm);
  CmdLineArgs pass_further{
      bpo::collect_unrecognized(parsed.options, bpo::include_positional),
      argv[0]};
  bpo::notify(vm);

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version: " << KokkosExt::version() << std::endl;

  if (vm.count("help") > 0)
  {
    // Full list of options consists of Kokkos + Boost.Program_options +
    // Google Benchmark and we still need to call benchmark::Initialize() to
    // get those printed to the standard output.
    std::cout << desc << "\n";
    int ac = 2;
    char *av[] = {(char *)"ignored", (char *)"--help"};
    // benchmark::Initialize() calls exit(0) when `--help` so register
    // Kokkos::finalize() to be called on normal program termination.
    std::atexit(Kokkos::finalize);
    benchmark::Initialize(&ac, av);
    return 1;
  }

  if (vm.count("exact-spec") > 0)
  {
    for (std::string option :
         {"values", "queries", "predicate-sort", "neighbors", "buffer",
          "source-point-cloud-type", "target-point-cloud-type"})
    {
      if (!vm[option].defaulted())
      {
        std::cout << "Conflicting options: 'exact-spec' and '" << option
                  << "', exiting..." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  benchmark::Initialize(&pass_further.argc(), pass_further.argv());
  // Throw if some of the arguments have not been recognized.
  std::ignore =
      bpo::command_line_parser(pass_further.argc(), pass_further.argv())
          .options(bpo::options_description(""))
          .run();

  std::vector<Spec> specs;
  specs.reserve(exact_specs.size());
  for (auto const &spec_string : exact_specs)
    specs.emplace_back(spec_string);

  if (vm.count("exact-spec") == 0)
  {
    single_spec.backends = "all";
    single_spec.source_point_cloud_type = to_point_cloud_enum(source_pt_cloud);
    single_spec.target_point_cloud_type = to_point_cloud_enum(target_pt_cloud);
    specs.push_back(single_spec);
  }

  for (auto const &spec : specs)
  {
    register_bvh_benchmarks(spec);
    register_boostrtree_benchmarks(spec);
  }

  benchmark::RunSpecifiedBenchmarks();

  Kokkos::finalize();
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
