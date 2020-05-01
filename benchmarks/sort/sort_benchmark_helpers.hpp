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

#ifndef SORT_BENCHMARK_HELPERS_HPP
#define SORT_BENCHMARK_HELPERS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <boost/program_options.hpp>

#include <algorithm>

#if defined(KOKKOS_ENABLE_OPENMP)
#include "pss_parallel_stable_sort.hpp"
#endif

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

#if 1
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

    return bin_sort.get_permute_vector();
#else
    auto permute =
        StdSortHelper<ValueType, SizeType>::sortAndComputePermutation(
            view_mirror);
    Kokkos::deep_copy(view, view_mirror);
    return permute;
#endif
  }

  static void sort(Kokkos::View<ValueType *, MemorySpace> view)
  {
    auto permute = sortAndComputePermutation(view);
    std::ignore = permute;
  }

  template <typename... PermuteViewProperties>
  static void
  applyPermutation(Kokkos::View<SizeType *, PermuteViewProperties...> permute,
                   Kokkos::View<ValueType *, MemorySpace> in,
                   Kokkos::View<ValueType *, MemorySpace> &out)
  {
    int const n = in.extent(0);

    execution_space exec_space;

    auto in_mirror = Kokkos::create_mirror_view_and_copy(exec_space, in);
    auto out_mirror = Kokkos::create_mirror_view(exec_space, out);

    Kokkos::parallel_for(
        "apply_permutation",
        Kokkos::RangePolicy<execution_space>(exec_space, 0, n),
        KOKKOS_LAMBDA(int const i) { out_mirror(permute(i)) = in_mirror(i); });

    Kokkos::deep_copy(out, out_mirror);
  }
};

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

#endif
