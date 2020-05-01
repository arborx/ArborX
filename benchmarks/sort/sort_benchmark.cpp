/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <random>

#include "sort_benchmark_helpers.hpp"
#include <benchmark/benchmark.h>

template <typename ViewType>
void buildRandomData(ViewType data)
{
  using ValueType = typename ViewType::value_type;
  std::conditional_t<std::is_integral<ValueType>::value,
                     std::uniform_int_distribution<ValueType>,
                     std::uniform_real_distribution<ValueType>>
      distribution(0, 10000);
  std::default_random_engine generator;
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  auto data_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, data);
  unsigned int const n = data.extent(0);
  for (unsigned int i = 0; i < n; ++i)
    data_host(i) = random();
  Kokkos::deep_copy(data, data_host);
}

template <class SortAlgorithm>
void sort(benchmark::State &state)
{
  using MemorySpace = typename SortAlgorithm::memory_space;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, MemorySpace> data("data", n);
  Kokkos::View<ValueType *, MemorySpace> data_copy("data_copy", n);
  buildRandomData(data_copy);

  for (auto _ : state)
  {
    Kokkos::deep_copy(data, data_copy);
    auto const start = std::chrono::high_resolution_clock::now();
    SortAlgorithm::sort(data);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class SortAlgorithm>
void sort_and_compute_permutation(benchmark::State &state)
{
  using MemorySpace = typename SortAlgorithm::memory_space;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, MemorySpace> data("data", n);
  Kokkos::View<ValueType *, MemorySpace> data_copy("data_copy", n);
  buildRandomData(data_copy);

  for (auto _ : state)
  {
    Kokkos::deep_copy(data, data_copy);
    auto const start = std::chrono::high_resolution_clock::now();
    auto permute = SortAlgorithm::sortAndComputePermutation(data);
    std::ignore = permute;
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class SortAlgorithm>
void compute_permutation(benchmark::State &state)
{
  using MemorySpace = typename SortAlgorithm::memory_space;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, MemorySpace> data("data", n);
  Kokkos::View<ValueType *, MemorySpace> data_copy("data_copy", n);
  buildRandomData(data_copy);

  for (auto _ : state)
  {
    Kokkos::deep_copy(data, data_copy);
    auto const start = std::chrono::high_resolution_clock::now();
    auto permute = SortAlgorithm::computePermutation(data);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}
template <class SortAlgorithm>
void apply_permutation(benchmark::State &state)
{
  using MemorySpace = typename SortAlgorithm::memory_space;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, MemorySpace> data("data", n);
  Kokkos::View<ValueType *, MemorySpace> data_copy("data_copy", n);
  buildRandomData(data_copy);

  auto permute = SortAlgorithm::sortAndComputePermutation(data_copy);

  for (auto _ : state)
  {
    auto const start = std::chrono::high_resolution_clock::now();
    SortAlgorithm::applyPermutation(permute, data_copy, data);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class SortAlgorithm>
void sort_compute_and_apply_permutation(benchmark::State &state)
{
  using MemorySpace = typename SortAlgorithm::memory_space;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, MemorySpace> data_orig("data", n);
  Kokkos::View<ValueType *, MemorySpace> data("data", n);
  Kokkos::View<ValueType *, MemorySpace> data_copy("data_copy", n);
  buildRandomData(data_orig);

  for (auto _ : state)
  {
    Kokkos::deep_copy(data, data_orig);
    auto const start = std::chrono::high_resolution_clock::now();
    auto permute = SortAlgorithm::sortAndComputePermutation(data);
    SortAlgorithm::applyPermutation(permute, data, data_copy);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

#define REGISTER_SORT_BENCHMARK(SortAlgorithm)                                 \
  BENCHMARK_TEMPLATE(sort, SortAlgorithm)                                      \
      ->Args({n})                                                              \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);

#define REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(SortAlgorithm)         \
  BENCHMARK_TEMPLATE(sort_and_compute_permutation, SortAlgorithm)              \
      ->Args({n})                                                              \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);

#define REGISTER_COMPUTE_PERMUTATION_BENCHMARK(SortAlgorithm)                  \
  BENCHMARK_TEMPLATE(compute_permutation, SortAlgorithm)                       \
      ->Args({n})                                                              \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);

#define REGISTER_APPLY_PERMUTATION_BENCHMARK(SortAlgorithm)                    \
  BENCHMARK_TEMPLATE(apply_permutation, SortAlgorithm)                         \
      ->Args({n})                                                              \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);

#define REGISTER_SORT_COMPUTE_AND_APPLY_PERMUTATION_BENCHMARK(SortAlgorithm)   \
  BENCHMARK_TEMPLATE(sort_compute_and_apply_permutation, SortAlgorithm)        \
      ->Args({n})                                                              \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);

// NOTE Motivation for this class that stores the argument count and values
// is I could not figure out how to make the parser consume arguments with
// Boost.Program_options
// Benchmark removes its own arguments from the command line arguments. This
// means, that by virtue of returning references to internal data members in
// argc() and argv() function, it will necessarily modify the members. It
// will decrease _argc, and "reduce" _argv data. Hence, we must keep a copy
// of _argv that is not modified from the outside to release memory in the
// destructor correctly.
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
    for (auto p : _owner_ptrs)
    {
      delete[] p;
    }
  }

  int &argc() { return _argc; }

  char **argv() { return _argv.data(); }
};

template <typename ValueType, typename SizeType>
void register_benchmarks(int const n)
{
  using Host = typename Kokkos::HostSpace::execution_space;
#if defined(KOKKOS_ENABLE_SERIAL)
  using Serial = Kokkos::Serial;
  using Kokkos_Serial =
      KokkosHelper<ValueType, Serial, typename Serial::memory_space, SizeType>;
  using StdSort_Serial = StdSortHelper<ValueType, SizeType>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  using OpenMP = Kokkos::OpenMP;
  using Kokkos_OpenMP =
      KokkosHelper<ValueType, OpenMP, typename OpenMP::memory_space, SizeType>;
  using PSS_OpenMP = PSSHelper<ValueType, SizeType>;
#ifdef ENABLE_GNU_PARALLEL
  using GnuParallel_OpenMP = SortGnuParallel<ValueType, SizeType>;
#endif
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  using Cuda = Kokkos::Cuda;
  using CudaSpace = Kokkos::CudaSpace;
  using Kokkos_Cuda = KokkosHelper<ValueType, Cuda, CudaSpace, SizeType>;
  using Kokkos_Cuda_Host = KokkosHelper<ValueType, Host, CudaSpace, SizeType>;
#if defined(KOKKOS_ENABLE_SERIAL)
  using Kokkos_Cuda_Serial =
      KokkosHelper<ValueType, Serial, CudaSpace, SizeType>;
#endif
  using Thrust_Cuda = ThrustHelper<ValueType, SizeType>;
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Serial);
  REGISTER_COMPUTE_PERMUTATION_BENCHMARK(StdSort_Serial);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Serial);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(StdSort_Serial);
  REGISTER_SORT_BENCHMARK(Kokkos_Serial);
  REGISTER_SORT_BENCHMARK(StdSort_Serial);
  REGISTER_SORT_COMPUTE_AND_APPLY_PERMUTATION_BENCHMARK(Kokkos_Serial);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_OpenMP);
  REGISTER_COMPUTE_PERMUTATION_BENCHMARK(PSS_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(PSS_OpenMP);
  REGISTER_SORT_BENCHMARK(Kokkos_OpenMP);
  REGISTER_SORT_BENCHMARK(PSS_OpenMP);
  REGISTER_SORT_COMPUTE_AND_APPLY_PERMUTATION_BENCHMARK(Kokkos_OpenMP);
#ifdef ENABLE_GNU_PARALLEL
  REGISTER_COMPUTE_PERMUTATION_BENCHMARK(GnuParallel_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(GnuParallel_OpenMP);
  REGISTER_SORT_BENCHMARK(GnuParallel_OpenMP);
#endif
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda);
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda_Host);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Cuda);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Cuda_Host);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Thrust_Cuda);
  REGISTER_SORT_BENCHMARK(Kokkos_Cuda);
  REGISTER_SORT_BENCHMARK(Kokkos_Cuda_Host);
  REGISTER_SORT_BENCHMARK(Thrust_Cuda);
  REGISTER_SORT_COMPUTE_AND_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda);
  REGISTER_SORT_COMPUTE_AND_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda_Host);
#if defined(KOKKOS_ENABLE_SERIAL)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda_Serial);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Cuda_Serial);
  REGISTER_SORT_BENCHMARK(Kokkos_Cuda_Serial);
  REGISTER_SORT_COMPUTE_AND_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda_Serial);
#endif
#endif
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  int n;
  std::string value_type, size_type;
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "num-values,n", bpo::value<int>(&n)->default_value(1000), "size" )
        ( "value-type", bpo::value<std::string>(&value_type)->default_value("float"), "value type" )
        ( "size-type", bpo::value<std::string>(&size_type)->default_value("unsigned int"), "size type" )
        ( "no-header", bpo::bool_switch(), "do not print version and hash" )
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

  if (!vm["no-header"].as<bool>())
  {
    std::cout << "value type    : " << value_type << std::endl;
    std::cout << "size type     : " << size_type << std::endl;
  }

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

  benchmark::Initialize(&pass_further.argc(), pass_further.argv());
  // Throw if some of the arguments have not been recognized.
  std::ignore =
      bpo::command_line_parser(pass_further.argc(), pass_further.argv())
          .options(bpo::options_description(""))
          .run();

  benchmark::Initialize(&argc, argv);

  // clang-format off
  if      (value_type == "float"  && size_type == "unsigned int") register_benchmarks<float, unsigned int>(n);
  else if (value_type == "float"  && size_type == "size_t")       register_benchmarks<float, size_t>(n);
  else if (value_type == "double" && size_type == "unsigned int") register_benchmarks<double, unsigned int>(n);
  else if (value_type == "double" && size_type == "size_t")       register_benchmarks<double, size_t>(n);
  else if (value_type == "int"    && size_type == "unsigned int") register_benchmarks<int, unsigned int>(n);
  else if (value_type == "int"    && size_type == "size_t")       register_benchmarks<int, size_t>(n);
  // clang-format on

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
