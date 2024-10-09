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

#ifndef ARBORX_SORT_UTILS_HPP
#define ARBORX_SORT_UTILS_HPP

#include <kokkos_ext/ArborX_KokkosExtSort.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp> // iota
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>   // clone
#include <misc/ArborX_Exception.hpp>

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
  KokkosExt::iota(space, permute);

  KokkosExt::sortByKey(space, view, permute);

  Kokkos::Profiling::popRegion();

  return permute;
}

template <typename ExecutionSpace, typename PermutationView, typename InputView,
          typename OutputView>
void applyInversePermutation(ExecutionSpace const &space,
                             PermutationView const &permutation,
                             InputView const &input_view,
                             OutputView const &output_view)
{
  static_assert(Kokkos::is_view_v<InputView>);
  static_assert(Kokkos::is_view_v<OutputView>);
  static_assert(InputView::rank() == 1);
  static_assert(OutputView::rank() == 1);
  static_assert(std::is_integral_v<typename PermutationView::value_type>);

  auto const n = input_view.extent(0);
  ARBORX_ASSERT(permutation.extent(0) == n);
  ARBORX_ASSERT(output_view.extent(0) == n);

  Kokkos::parallel_for(
      "ArborX::Sorting::inverse_permute", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i) { output_view(permutation(i)) = input_view(i); });
}

template <typename ExecutionSpace, typename PermutationView, typename InputView,
          typename OutputView>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation,
                      InputView const &input_view,
                      OutputView const &output_view)
{
  static_assert(Kokkos::is_view_v<InputView>);
  static_assert(Kokkos::is_view_v<OutputView>);
  static_assert(InputView::rank() == 1);
  static_assert(OutputView::rank() == 1);
  static_assert(std::is_integral_v<typename PermutationView::value_type>);

  auto const n = input_view.extent(0);
  ARBORX_ASSERT(permutation.extent(0) == n);
  ARBORX_ASSERT(output_view.extent(0) == n);

  Kokkos::parallel_for(
      "ArborX::Sorting::permute", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i) { output_view(i) = input_view(permutation(i)); });
}

template <typename ExecutionSpace, typename PermutationView, typename View>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation, View &view)
{
  auto scratch_view = KokkosExt::clone(space, view);
  applyPermutation(space, permutation, scratch_view, view);
}

} // namespace ArborX::Details

#endif
