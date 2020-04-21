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
#include <Kokkos_Sort.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <cstdlib>
#include <random>

#include <benchmark/benchmark.h>

#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
#define ENABLE_GNU_PARALLEL
#endif

#ifdef ENABLE_GNU_PARALLEL
#include <parallel/algorithm> // __gnu_parallel::sort, __gnu_parallel::transform
#endif

// clang-format off
#if defined(KOKKOS_ENABLE_CUDA)
#  if defined(KOKKOS_COMPILER_CLANG) && KOKKOS_COMPILER_CLANG < 900
// Clang of version less than 9.0 cannot compile Thrust, failing with errors
// like this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// Defining _CubLog here allows us to avoid that code path, however disabling
// some debugging diagnostics.
//
#    define _CubLog
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#  else // #if (KOKKOS_COMPILER_CLANG < 900)
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#  endif // #if (KOKKOS_COMPILER_CLANG < 900)
#endif   // #if defined(KOKKOS_ENABLE_CUDA)
// clang-format on

template <typename ValueType, typename DeviceType, typename SizeType>
struct SortKokkos
{
  using value_type = ValueType;
  using device_type = DeviceType;

  static Kokkos::View<SizeType *, DeviceType>
  sortAndComputePermutation(Kokkos::View<ValueType *, DeviceType> view)
  {
    using ViewType = Kokkos::View<ValueType *, DeviceType>;
    using ExecutionSpace = typename ViewType::execution_space;
    using CompType = Kokkos::BinOp1D<ViewType>;

    int const n = view.extent(0);

    Kokkos::MinMaxScalar<ValueType> result;
    Kokkos::MinMax<ValueType> reducer(result);
    Kokkos::parallel_reduce(
        "find_min_max_view", Kokkos::RangePolicy<ExecutionSpace>(0, n),
        Kokkos::Impl::min_max_functor<ViewType>(view), reducer);

    Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type,
                    SizeType>
        bin_sort(view, CompType(n / 2, result.min_val, result.max_val), true);
    bin_sort.create_permute_vector();
    bin_sort.sort(view);

    return bin_sort.get_permute_vector();
  }

  static void sort(Kokkos::View<ValueType *, DeviceType> view)
  {
    auto permute = sortAndComputePermutation(view);
    std::ignore = permute;
  }

  static void applyPermutation(Kokkos::View<SizeType *, DeviceType> permute,
                               Kokkos::View<ValueType *, DeviceType> in,
                               Kokkos::View<ValueType *, DeviceType> &out)
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    int const n = in.extent(0);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n),
        KOKKOS_LAMBDA(int const i) { out(permute(i)) = in(i); });
  }
};

#if defined(KOKKOS_ENABLE_SERIAL)
template <typename ValueType, typename DeviceType>
struct SortStd
{
  using value_type = ValueType;
  using device_type = DeviceType;

  static void sort(Kokkos::View<ValueType *, DeviceType> view)
  {
    using ViewType = Kokkos::View<ValueType *, DeviceType>;
    using ExecutionSpace = typename ViewType::execution_space;

    static_assert(std::is_same<Kokkos::Serial,
                               typename DeviceType::execution_space>::value,
                  "");

    std::sort(view.data(), view.data() + view.extent(0));
  }
};
#endif

#if defined(KOKKOS_ENABLE_OPENMP) && defined(ENABLE_GNU_PARALLEL)
template <typename ValueType, typename DeviceType, typename SizeType>
struct SortGnuParallel
{
  using value_type = ValueType;
  using device_type = DeviceType;

  static void sort(Kokkos::View<ValueType *, DeviceType> view)
  {
    static_assert(std::is_same<Kokkos::OpenMP,
                               typename DeviceType::execution_space>::value,
                  "");

#if !defined(__CUDA_ARCH__)
    int const n = view.extent(0);
    __gnu_parallel::sort(view.data(), view.data() + n);
#endif
  }

  static Kokkos::View<SizeType *, DeviceType>
  sortAndComputePermutation(Kokkos::View<ValueType *, DeviceType> view)
  {
    using ViewType = Kokkos::View<ValueType *, DeviceType>;
    using ExecutionSpace = typename ViewType::execution_space;

    static_assert(std::is_same<Kokkos::OpenMP,
                               typename DeviceType::execution_space>::value,
                  "");

    int const n = view.extent(0);

    Kokkos::View<SizeType *, DeviceType> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    Kokkos::parallel_for("iota", Kokkos::RangePolicy<ExecutionSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { permute(i) = i; });

#if !defined(__CUDA_ARCH__)
    __gnu_parallel::sort(permute.data(), permute.data() + n,
                         [&view](size_t const &a, size_t const &b) {
                           return view(a) < view(b);
                         });
#endif

    Kokkos::View<int *, DeviceType> view_copy("view_copy", n);
    Kokkos::deep_copy(view_copy, view);
    Kokkos::parallel_for(
        "apply_permutation", Kokkos::RangePolicy<ExecutionSpace>(0, n),
        KOKKOS_LAMBDA(int i) { view(i) = view_copy(permute(i)); });

    return permute;
  }
};
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <typename ValueType, typename DeviceType, typename SizeType>
struct SortThrust
{
  using value_type = ValueType;
  using device_type = DeviceType;

  static void sort(Kokkos::View<ValueType *, DeviceType> view)
  {
    static_assert(
        std::is_same<Kokkos::Cuda, typename DeviceType::execution_space>::value,
        "");

    int const n = view.extent(0);

    auto begin_ptr = thrust::device_ptr<ValueType>(view.data());
    auto end_ptr = thrust::device_ptr<ValueType>(view.data() + n);
    thrust::sort(begin_ptr, end_ptr);
  }

  static Kokkos::View<SizeType *, DeviceType>
  sortAndComputePermutation(Kokkos::View<ValueType *, DeviceType> view)
  {
    using ViewType = Kokkos::View<ValueType *, DeviceType>;
    using ExecutionSpace = typename ViewType::execution_space;

    static_assert(
        std::is_same<Kokkos::Cuda, typename DeviceType::execution_space>::value,
        "");

    int const n = view.extent(0);

    Kokkos::View<SizeType *, DeviceType> permute(
        Kokkos::ViewAllocateWithoutInitializing("permutation"), n);

    Kokkos::parallel_for("iota", Kokkos::RangePolicy<ExecutionSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { permute(i) = i; });

    auto permute_ptr = thrust::device_ptr<SizeType>(permute.data());
    auto begin_ptr = thrust::device_ptr<ValueType>(view.data());
    auto end_ptr = thrust::device_ptr<ValueType>(view.data() + n);
    thrust::sort_by_key(begin_ptr, end_ptr, permute_ptr);

    return permute;
  }
};
#endif

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
  using DeviceType = typename SortAlgorithm::device_type;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, DeviceType> data("data", n);
  buildRandomData(data);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
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
  using DeviceType = typename SortAlgorithm::device_type;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, DeviceType> data("data", n);
  buildRandomData(data);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    auto permute = SortAlgorithm::sortAndComputePermutation(data);
    std::ignore = permute;
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class SortAlgorithm>
void apply_permutation(benchmark::State &state)
{
  using DeviceType = typename SortAlgorithm::device_type;
  using ValueType = typename SortAlgorithm::value_type;

  int const n = state.range(0);

  // Construct random points
  Kokkos::View<ValueType *, DeviceType> data("data", n);
  buildRandomData(data);

  auto permute = SortAlgorithm::sortAndComputePermutation(data);

  Kokkos::View<ValueType *, DeviceType> data_copy("data_copy", n);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
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

#define REGISTER_APPLY_PERMUTATION_BENCHMARK(SortAlgorithm)                    \
  BENCHMARK_TEMPLATE(apply_permutation, SortAlgorithm)                         \
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
#if defined(KOKKOS_ENABLE_SERIAL)
  using Serial = Kokkos::Serial::device_type;
  using Kokkos_Serial = SortKokkos<ValueType, Serial, SizeType>;
  REGISTER_SORT_BENCHMARK(Kokkos_Serial);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Serial);
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Serial);
#endif

  using StdSort_Serial = SortStd<ValueType, Serial>;
  REGISTER_SORT_BENCHMARK(StdSort_Serial);

#if defined(KOKKOS_ENABLE_OPENMP)
  using OpenMP = Kokkos::OpenMP::device_type;
  using Kokkos_OpenMP = SortKokkos<ValueType, OpenMP, SizeType>;
  REGISTER_SORT_BENCHMARK(Kokkos_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_OpenMP);
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_OpenMP);

#ifdef ENABLE_GNU_PARALLEL
  using GnuParallel_OpenMP = SortGnuParallel<ValueType, OpenMP, SizeType>;
  REGISTER_SORT_BENCHMARK(GnuParallel_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(GnuParallel_OpenMP);
#endif

#endif

#if defined(KOKKOS_ENABLE_CUDA)
  using Cuda = Kokkos::Cuda::device_type;
  using Kokkos_Cuda = SortKokkos<ValueType, Cuda, SizeType>;
  REGISTER_SORT_BENCHMARK(Kokkos_Cuda);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Cuda);
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda);

  using Thrust_Cuda = SortThrust<ValueType, Cuda, SizeType>;
  REGISTER_SORT_BENCHMARK(Thrust_Cuda);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Thrust_Cuda);
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
