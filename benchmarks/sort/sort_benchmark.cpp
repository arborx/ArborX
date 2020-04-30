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

#if defined(KOKKOS_ENABLE_OPENMP)
#include "pss_parallel_stable_sort.hpp"
#endif
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

template <typename MemorySpace, typename ExecutionSpace, typename = void>
struct is_accessible_from : std::false_type
{
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value, "");
};

template <typename MemorySpace, typename ExecutionSpace>
struct is_accessible_from<MemorySpace, ExecutionSpace,
                          typename std::enable_if<Kokkos::SpaceAccessibility<
                              ExecutionSpace, MemorySpace>::accessible>::type>
    : std::true_type
{
};

template <typename ExecutionSpace, typename ViewType>
void iota(ExecutionSpace exec_space, ViewType view)
{
  auto const n = view.extent(0);
  Kokkos::parallel_for("iota",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int i) { view(i) = i; });
}

template <typename ValueType, typename ExecutionSpace, typename MemorySpace,
          typename SizeType, typename Enable = void>
struct KokkosHelper;

template <typename ValueType, typename ExecutionSpace, typename MemorySpace,
          typename SizeType>
struct KokkosHelper<
    ValueType, ExecutionSpace, MemorySpace, SizeType,
    std::enable_if_t<is_accessible_from<MemorySpace, ExecutionSpace>::value>>
{
  using value_type = ValueType;
  using execution_space = ExecutionSpace;
  using memory_space = MemorySpace;

  static auto
  sortAndComputePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    using ViewType =
        Kokkos::View<ValueType *,
                     Kokkos::Device<execution_space, memory_space>>;
    using CompType = Kokkos::BinOp1D<ViewType>;

    Kokkos::MinMaxScalar<ValueType> result;
    Kokkos::MinMax<ValueType> reducer(result);
    Kokkos::parallel_reduce(
        "min_max", Kokkos::RangePolicy<ExecutionSpace>(execution_space{}, 0, n),
        Kokkos::Impl::min_max_functor<ViewType>(view), reducer);

    Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type,
                    SizeType>
        bin_sort(view, CompType(n / 2, result.min_val, result.max_val), true);
    bin_sort.create_permute_vector();
    bin_sort.sort(view);

    return bin_sort.get_permute_vector();
  }

  static void sort(Kokkos::View<ValueType *, MemorySpace> view)
  {
    auto permute = sortAndComputePermutation(view);
    std::ignore = permute;
  }

  static void applyPermutation(Kokkos::View<SizeType *, MemorySpace> permute,
                               Kokkos::View<ValueType *, MemorySpace> in,
                               Kokkos::View<ValueType *, MemorySpace> &out)
  {
    int const n = in.extent(0);

    Kokkos::parallel_for(
        "apply_permutation",
        Kokkos::RangePolicy<execution_space>(execution_space{}, 0, n),
        KOKKOS_LAMBDA(int const i) { out(permute(i)) = in(i); });
  }
};

template <typename ValueType, typename ExecutionSpace, typename MemorySpace,
          typename SizeType>
struct KokkosHelper<
    ValueType, ExecutionSpace, MemorySpace, SizeType,
    std::enable_if_t<!is_accessible_from<MemorySpace, ExecutionSpace>::value>>
{
  using value_type = ValueType;
  using execution_space = ExecutionSpace;
  using memory_space = MemorySpace;

  static auto
  sortAndComputePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    auto view_mirror =
        Kokkos::create_mirror_view_and_copy(execution_space{}, view);

    using ViewType = decltype(view_mirror);
    using CompType = Kokkos::BinOp1D<ViewType>;

    Kokkos::MinMaxScalar<ValueType> result;
    Kokkos::MinMax<ValueType> reducer(result);
    Kokkos::parallel_reduce(
        "min_max", Kokkos::RangePolicy<ExecutionSpace>(execution_space{}, 0, n),
        Kokkos::Impl::min_max_functor<ViewType>(view_mirror), reducer);

    Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type,
                    SizeType>
        bin_sort(view_mirror, CompType(n / 2, result.min_val, result.max_val),
                 true);
    bin_sort.create_permute_vector();
    bin_sort.sort(view_mirror);

    Kokkos::deep_copy(view, view_mirror);

    return Kokkos::create_mirror_view_and_copy(
        typename memory_space::execution_space{},
        bin_sort.get_permute_vector());
  }

  static void sort(Kokkos::View<ValueType *, MemorySpace> view)
  {
    auto permute = sortAndComputePermutation(view);
    std::ignore = permute;
  }

  static void applyPermutation(Kokkos::View<SizeType *, MemorySpace> permute,
                               Kokkos::View<ValueType *, MemorySpace> in,
                               Kokkos::View<ValueType *, MemorySpace> &out)
  {
    int const n = in.extent(0);

    execution_space exec_space;

    auto permute_mirror =
        Kokkos::create_mirror_view_and_copy(exec_space, permute);
    auto in_mirror = Kokkos::create_mirror_view_and_copy(exec_space, in);
    auto out_mirror = Kokkos::create_mirror_view(exec_space, out);

    Kokkos::parallel_for("apply_permutation",
                         Kokkos::RangePolicy<execution_space>(exec_space, 0, n),
                         KOKKOS_LAMBDA(int const i) {
                           out_mirror(permute_mirror(i)) = in_mirror(i);
                         });

    Kokkos::deep_copy(out, out_mirror);
  }
};

#if defined(KOKKOS_ENABLE_SERIAL)
template <typename ValueType, typename SizeType>
struct StdSortHelper
{
  using value_type = ValueType;
  using execution_space = Kokkos::Serial;
  using memory_space = Kokkos::Serial::memory_space;

  static void sort(Kokkos::View<ValueType *, memory_space> view)
  {
    std::sort(view.data(), view.data() + view.extent(0));
  }

  static auto computePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    Kokkos::View<SizeType *, memory_space> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    for (int i = 0; i < n; ++i)
      permute(i) = i;

    std::sort(permute.data(), permute.data() + n,
              [&view](size_t const &a, size_t const &b) {
                return view(a) < view(b);
              });

    return permute;
  }

  static Kokkos::View<SizeType *, memory_space>
  sortAndComputePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    auto permute = computePermutation(view);

    std::vector<ValueType> view_copy(n);
    memcpy(view_copy.data(), view.data(), n * sizeof(ValueType));
    for (int i = 0; i < n; ++i)
      view(permute(i)) = view_copy[i];

    return permute;
  }
};
#endif

#if defined(KOKKOS_ENABLE_OPENMP) && defined(ENABLE_GNU_PARALLEL)
template <typename ValueType, typename SizeType>
struct SortGnuParallel
{
  using value_type = ValueType;
  using execution_space = Kokkos::OpenMP;
  using memory_space = Kokkos::HostSpace;

  static void sort(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);
    __gnu_parallel::sort(view.data(), view.data() + n);
  }

  static auto computePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    Kokkos::View<SizeType *, memory_space> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    iota(execution_space{}, permute);

    __gnu_parallel::sort(permute.data(), permute.data() + n,
                         [&view](size_t const &a, size_t const &b) {
                           return view(a) < view(b);
                         });

    return permute;
  }

  static Kokkos::View<SizeType *, memory_space>
  sortAndComputePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    auto permute = computePermutation(view);

    Kokkos::View<int *, memory_space> view_copy("view_copy", n);
    Kokkos::deep_copy(view_copy, view);
    Kokkos::parallel_for(
        "apply_permutation",
        Kokkos::RangePolicy<execution_space>(execution_space{}, 0, n),
        KOKKOS_LAMBDA(int i) { view(permute(i)) = view_copy(i); });

    return permute;
  }
};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <typename ValueType, typename SizeType>
struct PSSHelper
{
  using value_type = ValueType;
  using execution_space = Kokkos::OpenMP;
  using memory_space = Kokkos::HostSpace;

  static void sort(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);
    pss::parallel_stable_sort(view.data(), view.data() + n,
                              std::less<ValueType>{});
  }

  static auto computePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    Kokkos::View<SizeType *, memory_space> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    iota(execution_space{}, permute);

    pss::parallel_stable_sort(permute.data(), permute.data() + n,
                              [&view](size_t const &a, size_t const &b) {
                                return view(a) < view(b);
                              });

    return permute;
  }

  static Kokkos::View<SizeType *, memory_space>
  sortAndComputePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    auto permute = computePermutation(view);

    Kokkos::View<int *, memory_space> view_copy("view_copy", n);
    Kokkos::deep_copy(view_copy, view);
    Kokkos::parallel_for(
        "apply_permutation",
        Kokkos::RangePolicy<execution_space>(execution_space{}, 0, n),
        KOKKOS_LAMBDA(int i) { view(permute(i)) = view_copy(i); });

    return permute;
  }
};
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <typename ValueType, typename SizeType>
struct ThrustHelper
{
  using value_type = ValueType;
  using execution_space = Kokkos::Cuda;
  using memory_space = Kokkos::CudaSpace;

  static void sort(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    auto begin_ptr = thrust::device_ptr<ValueType>(view.data());
    auto end_ptr = thrust::device_ptr<ValueType>(view.data() + n);
    thrust::sort(begin_ptr, end_ptr);
  }

  static auto
  sortAndComputePermutation(Kokkos::View<ValueType *, memory_space> view)
  {
    int const n = view.extent(0);

    Kokkos::View<SizeType *, memory_space> permute(
        Kokkos::ViewAllocateWithoutInitializing("permutation"), n);

    auto permute_begin = thrust::device_ptr<SizeType>(permute.data());
    auto permute_end = thrust::device_ptr<SizeType>(permute.data() + n);
    auto view_begin = thrust::device_ptr<ValueType>(view.data());
    auto view_end = thrust::device_ptr<ValueType>(view.data() + n);

    thrust::sequence(permute_begin, permute_end, 0);
    thrust::sort_by_key(view_begin, view_end, permute_begin);

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
  using Kokkos_Host = KokkosHelper<ValueType, Host, CudaSpace, SizeType>;
  using Thrust_Cuda = ThrustHelper<ValueType, SizeType>;
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Serial);
  REGISTER_COMPUTE_PERMUTATION_BENCHMARK(StdSort_Serial);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Serial);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(StdSort_Serial);
  REGISTER_SORT_BENCHMARK(Kokkos_Serial);
  REGISTER_SORT_BENCHMARK(StdSort_Serial);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_OpenMP);
  REGISTER_COMPUTE_PERMUTATION_BENCHMARK(PSS_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(PSS_OpenMP);
  REGISTER_SORT_BENCHMARK(Kokkos_OpenMP);
  REGISTER_SORT_BENCHMARK(PSS_OpenMP);
#ifdef ENABLE_GNU_PARALLEL
  REGISTER_COMPUTE_PERMUTATION_BENCHMARK(GnuParallel_OpenMP);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(GnuParallel_OpenMP);
  REGISTER_SORT_BENCHMARK(GnuParallel_OpenMP);
#endif
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Cuda);
  REGISTER_APPLY_PERMUTATION_BENCHMARK(Kokkos_Host);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Cuda);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Kokkos_Host);
  REGISTER_SORT_AND_COMPUTE_PERMUTATION_BENCHMARK(Thrust_Cuda);
  REGISTER_SORT_BENCHMARK(Kokkos_Cuda);
  REGISTER_SORT_BENCHMARK(Kokkos_Host);
  REGISTER_SORT_BENCHMARK(Thrust_Cuda);
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
