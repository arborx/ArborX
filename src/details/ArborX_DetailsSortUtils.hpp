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

#ifndef ARBORX_DETAILS_SORT_UTILS_HPP
#define ARBORX_DETAILS_SORT_UTILS_HPP

#include <ArborX_DetailsUtils.hpp> // iota
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp> // min_max_functor

// clang-format off
#if defined(KOKKOS_ENABLE_CUDA)
#  if defined(KOKKOS_COMPILER_CLANG)
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

  using ValueType = typename ViewType::value_type;
  using CompType = Kokkos::BinOp1D<ViewType>;

  Kokkos::MinMaxScalar<ValueType> result;
  Kokkos::MinMax<ValueType> reducer(result);
  parallel_reduce(ARBORX_MARK_REGION("find_min_max_view"),
                  Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                  Kokkos::Impl::min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val)
  {
    Kokkos::View<SizeType *, typename ViewType::device_type> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    iota(space, permute);
    return permute;
  }

  Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type, SizeType>
      bin_sort(view, CompType(n / 2, result.min_val, result.max_val), true);
  bin_sort.create_permute_vector();
  bin_sort.sort(view);
  // FIXME Kokkos::BinSort is currently missing overloads that an execution
  // space as argument

  return bin_sort.get_permute_vector();
}

#if defined(KOKKOS_ENABLE_CUDA)
// NOTE returns the permutation indices **and** sorts the input view
template <typename ViewType, class SizeType = unsigned int>
Kokkos::View<SizeType *, typename ViewType::device_type>
sortObjects(Kokkos::Cuda const &space, ViewType &view)
{
  int const n = view.extent(0);

  using ValueType = typename ViewType::value_type;
  static_assert(
      std::is_same<Kokkos::Cuda, typename ViewType::execution_space>::value,
      "");

  Kokkos::View<SizeType *, typename ViewType::device_type> permute(
      Kokkos::ViewAllocateWithoutInitializing("permutation"), n);
  ArborX::iota(space, permute);

  auto const execution_policy = thrust::cuda::par.on(space.cuda_stream());

  auto permute_ptr = thrust::device_ptr<SizeType>(permute.data());
  auto begin_ptr = thrust::device_ptr<ValueType>(view.data());
  auto end_ptr = thrust::device_ptr<ValueType>(view.data() + n);
  thrust::sort_by_key(execution_policy, begin_ptr, end_ptr, permute_ptr);

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

template <typename ExecutionSpace, typename PermutationView, typename View>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation, View &view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  ARBORX_ASSERT(permutation.extent(0) == view.extent(0));
  auto scratch_view = clone(space, view);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("permute"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, view.extent(0)),
      KOKKOS_LAMBDA(int i) {
        PermuteHelper::CopyOp<View, View>::copy(scratch_view, i, view,
                                                permutation(i));
      });
  Kokkos::deep_copy(space, view, scratch_view);
}

} // namespace Details

} // namespace ArborX

#endif
