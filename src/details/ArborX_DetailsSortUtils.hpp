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

#ifndef ARBORX_DETAILS_SORT_UTILS_HPP
#define ARBORX_DETAILS_SORT_UTILS_HPP

#include <ArborX_Config.hpp> // ARBORX_ENABLE_ROCTHRUST

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp> // is_accessible_from
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>         // clone
#include <ArborX_DetailsUtils.hpp>                        // iota
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>
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

namespace ArborX
{

namespace Details
{

// NOTE returns the permutation indices **and** sorts the input view
template <typename ExecutionSpace, typename ViewType,
          class SizeType = unsigned int>
Kokkos::View<SizeType *, typename ViewType::device_type>
sortObjects(ExecutionSpace const &space, ViewType &view)
{
  int const n = view.extent(0);

  if (n == 0)
  {
    return Kokkos::View<SizeType *, typename ViewType::device_type>(
        "ArborX::Sorting::permute", 0);
  }

  using ValueType = typename ViewType::value_type;
  using CompType = Kokkos::BinOp1D<ViewType>;

  ValueType min_val;
  ValueType max_val;
  std::tie(min_val, max_val) = ArborX::minMax(space, view);
  if (min_val == max_val)
  {
    Kokkos::View<SizeType *, typename ViewType::device_type> permute(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::Sorting::permute"),
        n);
    iota(space, permute);
    return permute;
  }

  Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type, SizeType>
      bin_sort(view, CompType(n / 2, min_val, max_val), true);
  bin_sort.create_permute_vector();
  bin_sort.sort(view);
  // FIXME Kokkos::BinSort is currently missing overloads that an execution
  // space as argument

  return bin_sort.get_permute_vector();
}

#if defined(KOKKOS_ENABLE_CUDA) ||                                             \
    (defined(KOKKOS_ENABLE_HIP) && defined(ARBORX_ENABLE_ROCTHRUST))
// NOTE returns the permutation indices **and** sorts the input view
template <typename ViewType, class SizeType = unsigned int>
Kokkos::View<SizeType *, typename ViewType::device_type> sortObjects(
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::Cuda const &space,
#else
    Kokkos::Experimental::HIP const &space,
#endif
    ViewType &view)
{
  int const n = view.extent(0);

  using ValueType = typename ViewType::value_type;
  static_assert(std::is_same<std::decay_t<decltype(space)>,
                             typename ViewType::execution_space>::value,
                "");

  Kokkos::View<SizeType *, typename ViewType::device_type> permute(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::Sorting::permutation"),
      n);
  ArborX::iota(space, permute);

#if defined(KOKKOS_ENABLE_CUDA)
  auto const execution_policy = thrust::cuda::par.on(space.cuda_stream());
#else
  auto const execution_policy = thrust::hip::par.on(space.hip_stream());
#endif

  auto permute_ptr = thrust::device_ptr<SizeType>(permute.data());
  auto begin_ptr = thrust::device_ptr<ValueType>(view.data());
  auto end_ptr = thrust::device_ptr<ValueType>(view.data() + n);
  thrust::sort_by_key(execution_policy, begin_ptr, end_ptr, permute_ptr);

  return permute;
}
#endif

#if defined(KOKKOS_ENABLE_SYCL) && defined(ARBORX_ENABLE_ONEDPL)
// NOTE returns the permutation indices **and** sorts the input view
template <typename ViewType, class SizeType = unsigned int>
Kokkos::View<SizeType *, typename ViewType::device_type>
sortObjects(Kokkos::Experimental::SYCL const &space, ViewType &view)
{
  int const n = view.extent(0);

  static_assert(
      KokkosExt::is_accessible_from<typename ViewType::memory_space,
                                    Kokkos::Experimental::SYCL>::value,
      "");

  Kokkos::View<SizeType *, typename ViewType::device_type> permute(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::Sorting::permutation"),
      n);
  ArborX::iota(space, permute);

  auto zipped_begin =
      oneapi::dpl::make_zip_iterator(view.data(), permute.data());
  oneapi::dpl::execution::device_policy policy(
      *space.impl_internal_space_instance()->m_queue);
  oneapi::dpl::sort(
      policy, zipped_begin, zipped_begin + n,
      [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

  return permute;
}
#endif

// Helper functions and structs for applyPermutations
namespace PermuteHelper
{
template <class DstViewType, class SrcViewType, int Rank = DstViewType::Rank>
struct CopyOp;

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 1>
{
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const &dst, size_t i_dst, SrcViewType const &src,
                   size_t i_src)
  {
    dst(i_dst) = src(i_src);
  }
};

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 2>
{
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const &dst, size_t i_dst, SrcViewType const &src,
                   size_t i_src)
  {
    for (unsigned int j = 0; j < dst.extent(1); j++)
      dst(i_dst, j) = src(i_src, j);
  }
};

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 3>
{
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const &dst, size_t i_dst, SrcViewType const &src,
                   size_t i_src)
  {
    for (unsigned int j = 0; j < dst.extent(1); j++)
      for (unsigned int k = 0; k < dst.extent(2); k++)
        dst(i_dst, j, k) = src(i_src, j, k);
  }
};
} // namespace PermuteHelper

template <typename ExecutionSpace, typename PermutationView, typename InputView,
          typename OutputView>
void applyInversePermutation(ExecutionSpace const &space,
                             PermutationView const &permutation,
                             InputView const &input_view,
                             OutputView const &output_view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  ARBORX_ASSERT(permutation.extent(0) == input_view.extent(0));
  ARBORX_ASSERT(output_view.extent(0) == input_view.extent(0));

  Kokkos::parallel_for(
      "ArborX::Sorting::inverse_permute",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, input_view.extent(0)),
      KOKKOS_LAMBDA(int i) {
        PermuteHelper::CopyOp<OutputView, InputView>::copy(
            output_view, permutation(i), input_view, i);
      });
}

template <typename ExecutionSpace, typename PermutationView, typename InputView,
          typename OutputView>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation,
                      InputView const &input_view,
                      OutputView const &output_view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  ARBORX_ASSERT(permutation.extent(0) == input_view.extent(0));
  ARBORX_ASSERT(output_view.extent(0) == input_view.extent(0));

  Kokkos::parallel_for(
      "ArborX::Sorting::permute",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, input_view.extent(0)),
      KOKKOS_LAMBDA(int i) {
        PermuteHelper::CopyOp<OutputView, InputView>::copy(
            output_view, i, input_view, permutation(i));
      });
}

template <typename ExecutionSpace, typename PermutationView, typename View>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation, View &view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  auto scratch_view = KokkosExt::clone(space, view);
  applyPermutation(space, permutation, scratch_view, view);
}

} // namespace Details

} // namespace ArborX

#endif
