/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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

#include <Kokkos_Sort.hpp> // min_max_functor
#include <Kokkos_View.hpp>

// clang-format off
#if defined(KOKKOS_ENABLE_CUDA)
#  if defined(KOKKOS_COMPILER_CLANG) && KOKKOS_COMPILER_CLANG < 900
// Clang of version less than 9.0 cannot compile Thrust, failing with errors
// like this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
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
#  else // #if (KOKKOS_COMPILER_CLANG < 900)
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#  endif // #if (KOKKOS_COMPILER_CLANG < 900)
#endif   // #if defined(KOKKOS_ENABLE_CUDA)
// clang-format on

namespace ArborX
{

namespace Details
{

// NOTE returns the permutation indices **and** sorts the input view
template <typename ViewType>
Kokkos::View<size_t *, typename ViewType::device_type>
sortObjects(ViewType &view)
{
  using ExecutionSpace = typename ViewType::execution_space;

  int const n = view.extent(0);

  using ValueType = typename ViewType::value_type;
  using CompType = Kokkos::BinOp1D<ViewType>;

  Kokkos::MinMaxScalar<ValueType> result;
  Kokkos::MinMax<ValueType> reducer(result);
  parallel_reduce(ARBORX_MARK_REGION("find_min_max_view"),
                  Kokkos::RangePolicy<ExecutionSpace>(0, n),
                  Kokkos::Impl::min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val)
  {
    Kokkos::View<size_t *, typename ViewType::device_type> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    iota(permute);
    return permute;
  }

  // Passing the SizeType template argument to Kokkos::BinSort because it
  // defaults to the memory space size type which is different on the host and
  // on cuda (size_t versus unsigned int respectively).  size_t feels like a
  // better choice here because its size is guaranteed to coincide with the
  // pointer size which is a good thing for converting with reinterpret_cast
  // (when leaf indices are encoded into the pointer to one of their children)
  Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type, size_t>
      bin_sort(view, CompType(n / 2, result.min_val, result.max_val), true);
  bin_sort.create_permute_vector();
  bin_sort.sort(view);

  return bin_sort.get_permute_vector();
}

#if defined(KOKKOS_ENABLE_CUDA)
// NOTE returns the permutation indices **and** sorts the input view
template <typename ValueType, typename MemorySpace>
Kokkos::View<size_t *, Kokkos::Device<Kokkos::Cuda, MemorySpace>> sortObjects(
    Kokkos::View<ValueType *, Kokkos::Device<Kokkos::Cuda, MemorySpace>> view)
{
  int const n = view.extent(0);

  Kokkos::View<size_t *, Kokkos::Device<Kokkos::Cuda, MemorySpace>> permute(
      Kokkos::ViewAllocateWithoutInitializing("permutation"), n);
  ArborX::iota(permute);

  auto permute_ptr = thrust::device_ptr<size_t>(permute.data());
  auto begin_ptr = thrust::device_ptr<ValueType>(view.data());
  auto end_ptr = thrust::device_ptr<ValueType>(view.data() + n);
  thrust::sort_by_key(begin_ptr, end_ptr, permute_ptr);

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
    for (int j = 0; j < (int)dst.extent(1); j++)
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
    for (int j = 0; j < dst.extent(1); j++)
      for (int k = 0; k < dst.extent(2); k++)
        dst(i_dst, j, k) = src(i_src, j, k);
  }
};
} // namespace PermuteHelper

template <typename PermutationView, typename View>
void applyPermutations(PermutationView const &permutation, View &view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  ARBORX_ASSERT(permutation.extent(0) == view.extent(0));
  auto scratch_view = clone(view);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("permute"),
      Kokkos::RangePolicy<typename View::execution_space>(0, view.extent(0)),
      KOKKOS_LAMBDA(int i) {
        PermuteHelper::CopyOp<View, View>::copy(scratch_view, i, view,
                                                permutation(i));
      });
  Kokkos::deep_copy(view, scratch_view);
}

} // namespace Details

} // namespace ArborX

#endif
