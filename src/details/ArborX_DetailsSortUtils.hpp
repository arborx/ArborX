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

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp> // is_accessible_from
#include <ArborX_DetailsKokkosExtSort.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp> // clone
#include <ArborX_DetailsUtils.hpp>                // iota
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

// NOTE returns the permutation indices **and** sorts the input view
template <typename ExecutionSpace, typename ViewType,
          class SizeType = unsigned int>
auto sortObjects(ExecutionSpace const &space, ViewType &view)
{
  Kokkos::Profiling::pushRegion("ArborX::Sorting");

  Kokkos::View<SizeType *, typename ViewType::device_type> permute(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::Sorting::permute"),
      view.extent(0));
  ArborX::iota(space, permute);

  KokkosExt::sortByKey(space, view, permute);

  Kokkos::Profiling::popRegion();

  return permute;
}

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
  static_assert(std::is_integral<typename PermutationView::value_type>::value);
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
  static_assert(std::is_integral<typename PermutationView::value_type>::value);
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
  static_assert(std::is_integral<typename PermutationView::value_type>::value);
  auto scratch_view = KokkosExt::clone(space, view);
  applyPermutation(space, permutation, scratch_view, view);
}

} // namespace ArborX::Details

#endif
