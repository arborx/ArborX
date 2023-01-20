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

#include <ArborX_Config.hpp> // ARBORX_ENABLE_ROCTHRUST

#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsUtils.hpp> // minMax

#include <Kokkos_Sort.hpp>

// clang-format off
#if defined(KOKKOS_ENABLE_CUDA)
#  if defined(KOKKOS_COMPILER_CLANG)

// Older Thrust (or CUB to be more precise) versions use __shfl instead of
// __shfl_sync for clang which was removed in PTX ISA version 6.4, also see
// https://github.com/NVIDIA/cub/pull/170.
#include <cub/version.cuh>
#if defined(CUB_VERSION) && (CUB_VERSION < 101100) && !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

// Some versions of Clang fail to compile Thrust, failing with errors like
// this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// The exact combination of versions for Clang and Thrust (or CUDA) for this
// failure was not investigated, however even very recent version combination
// (Clang 10.0.0 and Cuda 10.0) demonstrated failure.
//
// Defining _CubLog here allows us to avoid that code path, however disabling
// some debugging diagnostics
//
// If _CubLog is already defined, we save it into ARBORX_CubLog_save, and
// restore it at the end
#    ifdef _CubLog
#      define ARBORX_CubLog_save _CubLog
#    endif
#    define _CubLog
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#    undef _CubLog
#    ifdef ARBORX_CubLog_save
#      define _CubLog ARBORX_CubLog_save
#      undef ARBORX_CubLog_save
#    endif
#  else // #if defined(KOKKOS_COMPILER_CLANG)
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#  endif // #if defined(KOKKOS_COMPILER_CLANG)
#endif   // #if defined(KOKKOS_ENABLE_CUDA)
// clang-format on

#if defined(KOKKOS_ENABLE_HIP) && defined(ARBORX_ENABLE_ROCTHRUST)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#if defined(KOKKOS_ENABLE_SYCL) && defined(ARBORX_ENABLE_ONEDPL)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

namespace KokkosExt
{

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

  auto [min_val, max_val] = ArborX::minMax(space, keys);
  if (min_val == max_val)
    return;

  using SizeType = unsigned int;
  using CompType = Kokkos::BinOp1D<Keys>;

  Kokkos::BinSort<Keys, CompType, typename Keys::device_type, SizeType>
      bin_sort(space, keys, CompType(n / 2, min_val, max_val), true);
  bin_sort.create_permute_vector(space);
  bin_sort.sort(space, keys);
  bin_sort.sort(space, values);
}

#if defined(KOKKOS_ENABLE_CUDA) ||                                             \
    (defined(KOKKOS_ENABLE_HIP) && defined(ARBORX_ENABLE_ROCTHRUST))
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
