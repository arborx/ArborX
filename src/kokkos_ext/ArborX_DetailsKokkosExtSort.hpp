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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_SORT_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_SORT_HPP

#include <ArborX_Config.hpp> // ARBORX_ENABLE_THRUST

#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsUtils.hpp> // minMax

#include <Kokkos_Sort.hpp>

// clang-format off
#ifdef ARBORX_ENABLE_THRUST
#  if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_COMPILER_CLANG)

// Older Thrust (or CUB to be more precise) versions use __shfl instead of
// __shfl_sync for clang which was removed in PTX ISA version 6.4, also see
// https://github.com/NVIDIA/cub/pull/170.
#    include <cub/version.cuh>
#    if defined(CUB_VERSION) && (CUB_VERSION < 101100) && !defined(CUB_USE_COOPERATIVE_GROUPS)
#      define CUB_USE_COOPERATIVE_GROUPS
#    endif

// Some versions of Clang fail to compile Thrust, failing with errors like
// this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// The exact combination of versions for Clang and Thrust (or CUDA) for this
// failure was not investigated, however even very recent version combination
// (Clang 14.0.0 and Cuda 11.0) demonstrated failure.
//
// Defining _CubLog here allows us to avoid that code path, however disabling
// some debugging diagnostics
#    pragma push_macro("_CubLog")
#    undef _CubLog
#    define _CubLog
#  endif

#  ifdef KOKKOS_ENABLE_OPENMP
// hipcc does not define _OPENMP when compiling with "-fopenmp"
// (https://github.com/ROCm-Developer-Tools/HIP/blob/develop/docs/markdown/hip_faq.md#why-_openmp-is-undefined-when-compiling-with--fopenmp).
// So we have to explicitly define it here (which is safe to do due to being
// inside KOKKOS_ENABLE_OPENMP). OpenMP specification actually requires the
// macro to be in the form of yyyymm, but Thrust libraries only check whether
// it's defined.
#    pragma push_macro("_OPENMP")
#    undef _OPENMP
#    define _OPENMP
#    include "thrust/system/omp/execution_policy.h"
#    pragma pop_macro("_OPENMP")
#  endif
#  include <thrust/device_ptr.h>
#  include <thrust/sort.h>

#  if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_COMPILER_CLANG)
#    pragma pop_macro("_CubLog")
#  endif
#endif
// clang-format on

#if defined(KOKKOS_ENABLE_SYCL) && defined(ARBORX_ENABLE_ONEDPL)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

namespace KokkosExt
{

#ifdef ARBORX_ENABLE_THRUST
template <typename ExecutionSpace>
auto getThrustExecutionPolicy(ExecutionSpace const &space)
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
    return thrust::cuda::par.on(space.cuda_stream());
  else
#endif
#ifdef KOKKOS_ENABLE_HIP
      if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Experimental::HIP>)
    return thrust::hip::par.on(space.hip_stream());
  else
#endif
#ifdef KOKKOS_ENABLE_OPENMP
      if constexpr (std::is_same_v<ExecutionSpace, Kokkos::OpenMP>)
    return thrust::omp::par;
  else
#endif
#ifdef KOKKOS_ENABLE_SERIAL
      if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Serial>)
    return thrust::seq;
  else
#endif
    return;
}
#endif

template <typename ExecutionSpace>
struct is_thrust_available_for_space
    : std::conditional_t<std::is_same_v<decltype(getThrustExecutionPolicy(
                                            std::declval<ExecutionSpace>())),
                                        void>,
                         std::false_type, std::true_type>
{};

template <typename ExecutionSpace, typename Keys, typename Values>
void sortByKey(ExecutionSpace const &space, Keys &keys, Values &values)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::KokkosExt::sortByKey::Kokkos");

  static_assert(Kokkos::is_view<Keys>::value);
  static_assert(Kokkos::is_view<Values>::value);
  static_assert(Keys::rank == 1);
  static_assert(Values::rank == 1);
  static_assert(KokkosExt::is_accessible_from<typename Keys::memory_space,
                                              ExecutionSpace>::value);
  static_assert(KokkosExt::is_accessible_from<typename Values::memory_space,
                                              ExecutionSpace>::value);
  auto const n = keys.size();
  ARBORX_ASSERT(values.size() == n);

  if (n == 0)
    return;

#ifdef ARBORX_ENABLE_THRUST
  if constexpr (is_thrust_available_for_space<ExecutionSpace>{})
  {
    // Using non-declared symbols is not valid even in constexpr branch, so we
    // have to protect the call.
    thrust::sort_by_key(getThrustExecutionPolicy(space), keys.data(),
                        keys.data() + keys.size(), values.data());
  }
  else
#endif
  {
    // Use Kokkos::BinSort
    auto [min_val, max_val] = ArborX::minMax(space, keys);
    if (min_val == max_val)
      return;

    using SizeType = unsigned int;
    using CompType = Kokkos::BinOp1D<Keys>;

#if KOKKOS_VERSION >= 30700
    Kokkos::BinSort<Keys, CompType, typename Keys::device_type, SizeType>
        bin_sort(space, keys, CompType(n / 2, min_val, max_val), true);
    bin_sort.create_permute_vector(space);
    bin_sort.sort(space, keys);
    bin_sort.sort(space, values);
#else
    Kokkos::BinSort<Keys, CompType, typename Keys::device_type, SizeType>
        bin_sort(keys, CompType(n / 2, min_val, max_val), true);
    bin_sort.create_permute_vector();
    bin_sort.sort(keys);
    bin_sort.sort(values);
#endif
  }
}

#if defined(ARBORX_ENABLE_THRUST)

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <typename Keys, typename Values>
void sortByKey(
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::Cuda const &space,
#else
    Kokkos::Experimental::HIP const &space,
#endif
    Keys &keys, Values &values)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::KokkosExt::sortByKey::Thrust");

  using ExecutionSpace = std::decay_t<decltype(space)>;
  static_assert(Kokkos::is_view<Keys>::value);
  static_assert(Kokkos::is_view<Values>::value);
  static_assert(Keys::rank == 1);
  static_assert(Values::rank == 1);
  static_assert(KokkosExt::is_accessible_from<typename Keys::memory_space,
                                              ExecutionSpace>::value);
  static_assert(KokkosExt::is_accessible_from<typename Values::memory_space,
                                              ExecutionSpace>::value);
  auto const n = keys.size();
  ARBORX_ASSERT(values.size() == n);

  if (n == 0)
    return;

#if defined(KOKKOS_ENABLE_CUDA)
  auto const execution_policy = thrust::cuda::par.on(space.cuda_stream());
#else
  auto const execution_policy = thrust::hip::par.on(space.hip_stream());
#endif

  thrust::sort_by_key(execution_policy, keys.data(), keys.data() + n,
                      values.data());
}
#endif

#endif

#if defined(KOKKOS_ENABLE_SYCL) && defined(ARBORX_ENABLE_ONEDPL)
template <typename Keys, typename Values>
void sortByKey(Kokkos::Experimental::SYCL const &space, Keys &keys,
               Values &values)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::KokkosExt::sortByKey::OneDPL");

  using ExecutionSpace = std::decay_t<decltype(space)>;
  static_assert(Kokkos::is_view<Keys>::value);
  static_assert(Kokkos::is_view<Values>::value);
  static_assert(Keys::rank == 1);
  static_assert(Values::rank == 1);
  static_assert(KokkosExt::is_accessible_from<typename Keys::memory_space,
                                              ExecutionSpace>::value);
  static_assert(KokkosExt::is_accessible_from<typename Values::memory_space,
                                              ExecutionSpace>::value);
  auto const n = keys.size();
  ARBORX_ASSERT(values.size() == n);

  if (n == 0)
    return;

  auto zipped_begin =
      oneapi::dpl::make_zip_iterator(keys.data(), values.data());
  oneapi::dpl::execution::device_policy policy(
      *space.impl_internal_space_instance()->m_queue);
  oneapi::dpl::sort(
      policy, zipped_begin, zipped_begin + n,
      [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
}
#endif

} // namespace KokkosExt

#endif
