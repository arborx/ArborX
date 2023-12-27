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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_STD_ALGORITHMS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_STD_ALGORITHMS_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details::KokkosExt
{

template <typename ExecutionSpace, typename SrcView, typename DstView,
          typename InitValueType = typename DstView::value_type>
void exclusive_scan(ExecutionSpace const &space, SrcView const &src,
                    DstView const &dst, InitValueType init = 0)
{
  static_assert(
      Kokkos::is_execution_space<std::decay_t<ExecutionSpace>>::value);
  static_assert(Kokkos::is_view<SrcView>::value);
  static_assert(Kokkos::is_view<DstView>::value);
  static_assert(
      is_accessible_from<typename SrcView::memory_space, ExecutionSpace>::value,
      "Source view must be accessible from the execution space");
  static_assert(
      is_accessible_from<typename DstView::memory_space, ExecutionSpace>::value,
      "Destination view must be accessible from the execution "
      "space");
  static_assert(std::is_same<typename SrcView::value_type,
                             typename DstView::non_const_value_type>::value,
                "exclusive_scan requires non-const destination type");
  static_assert(unsigned(DstView::rank) == unsigned(SrcView::rank) &&
                    unsigned(DstView::rank) == unsigned(1),
                "exclusive_scan requires Views of rank 1");

  using ValueType = typename DstView::value_type;

  auto const n = src.extent(0);
  ARBORX_ASSERT(n == dst.extent(0));
  Kokkos::parallel_scan(
      "ArborX::Algorithms::exclusive_scan",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i, ValueType &update, bool final_pass) {
        auto const tmp = src(i);
        if (final_pass)
          dst(i) = update + init;
        update += tmp;
      });
}

template <typename ExecutionSpace, typename ViewType>
typename ViewType::non_const_value_type
reduce(ExecutionSpace const &space, ViewType const &v,
       typename ViewType::non_const_value_type init)
{
  static_assert(
      Kokkos::is_execution_space<std::decay_t<ExecutionSpace>>::value);
  static_assert(Kokkos::is_view<ViewType>::value);
  static_assert(is_accessible_from<typename ViewType::memory_space,
                                   ExecutionSpace>::value,
                "Source view must be accessible from the execution space");
  static_assert(ViewType::rank == 1, "accumulate requires a View of rank 1");

  // NOTE: Passing the argument init directly to the parallel_reduce() while
  // using a lambda does not yield the expected result because Kokkos will
  // supply a default init method that sets the reduction result to zero.
  // Rather than going through the hassle of defining a custom functor for
  // the reduction, introduce here a temporary variable and add it to init
  // before returning.
  typename ViewType::non_const_value_type tmp;
  Kokkos::parallel_reduce(
      "ArborX::Algorithms::accumulate",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, v.extent(0)),
      KOKKOS_LAMBDA(int i, typename ViewType::non_const_value_type &update) {
        update += v(i);
      },
      tmp);
  init += tmp;
  return init;
}

template <typename ExecutionSpace, typename SrcView, typename DstView>
void adjacent_difference(ExecutionSpace const &space, SrcView const &src,
                         DstView const &dst)
{
  static_assert(
      Kokkos::is_execution_space<std::decay_t<ExecutionSpace>>::value);
  static_assert(Kokkos::is_view<SrcView>::value);
  static_assert(Kokkos::is_view<DstView>::value);
  static_assert(
      is_accessible_from<typename SrcView::memory_space, ExecutionSpace>::value,
      "Source view must be accessible from the execution space");
  static_assert(
      is_accessible_from<typename DstView::memory_space, ExecutionSpace>::value,
      "Destination view must be accessible from the execution space");
  static_assert(SrcView::rank == 1 && DstView::rank == 1,
                "adjacent_difference operates on rank-1 views");
  static_assert(
      std::is_same<typename DstView::value_type,
                   typename DstView::non_const_value_type>::value,
      "adjacent_difference requires non-const destination value type");
  static_assert(std::is_same<typename SrcView::non_const_value_type,
                             typename DstView::value_type>::value,
                "adjacent_difference requires same value type for source and "
                "destination");

  auto const n = src.extent(0);
  ARBORX_ASSERT(n == dst.extent(0));
  ARBORX_ASSERT(src != dst);
  Kokkos::parallel_for(
      "ArborX::Algorithms::adjacent_difference",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        if (i > 0)
          dst(i) = src(i) - src(i - 1);
        else
          dst(i) = src(i);
      });
}

template <typename ExecutionSpace, typename ViewType>
void iota(ExecutionSpace const &space, ViewType const &v,
          typename ViewType::value_type value = 0)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_view<ViewType>::value);
  static_assert(unsigned(ViewType::rank) == unsigned(1),
                "iota requires a View of rank 1");

  using ValueType = typename ViewType::value_type;
  static_assert(std::is_arithmetic<ValueType>::value,
                "iota requires a View with an arithmetic value type");
  static_assert(
      std::is_same<ValueType, typename ViewType::non_const_value_type>::value,
      "iota requires a View with non-const value type");

  Kokkos::parallel_for(
      "ArborX::Algorithms::iota",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, v.extent(0)),
      KOKKOS_LAMBDA(int i) { v(i) = value + (ValueType)i; });
}

} // namespace ArborX::Details::KokkosExt

#endif
