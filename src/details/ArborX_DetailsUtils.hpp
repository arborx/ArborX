/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_UTILS_HPP
#define ARBORX_DETAILS_UTILS_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxReduce.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename ExecutionSpace, typename ST, typename... SP, typename DT,
          typename... DP>
[[deprecated]] void exclusivePrefixSum(ExecutionSpace &&space,
                                       Kokkos::View<ST, SP...> const &src,
                                       Kokkos::View<DT, DP...> const &dst)
{
  Details::KokkosExt::exclusive_scan(std::forward<ExecutionSpace>(space), src,
                                     dst, 0);
}

template <typename ExecutionSpace, typename T, typename... P>
[[deprecated]] inline std::enable_if_t<
    Kokkos::is_execution_space<std::remove_reference_t<ExecutionSpace>>::value>
exclusivePrefixSum(ExecutionSpace &&space, Kokkos::View<T, P...> const &v)
{
  exclusivePrefixSum(std::forward<ExecutionSpace>(space), v, v);
}

template <typename ST, typename... SP, typename DT, typename... DP>
[[deprecated]] inline void
exclusivePrefixSum(Kokkos::View<ST, SP...> const &src,
                   Kokkos::View<DT, DP...> const &dst)
{
  using ExecutionSpace = typename Kokkos::View<DT, DP...>::execution_space;
  exclusivePrefixSum(ExecutionSpace{}, src, dst);
}

template <typename T, typename... P>
[[deprecated]] inline void exclusivePrefixSum(Kokkos::View<T, P...> const &v)
{
  using ExecutionSpace = typename Kokkos::View<T, P...>::execution_space;
  exclusivePrefixSum(ExecutionSpace{}, v);
}

/** \brief Get a copy of the last element.
 *
 *  Returns a copy of the last element in the view on the host.  Note that it
 *  may require communication between host and device (e.g. if the view passed
 *  as an argument lives on the device).
 *
 *  \pre \c v is of rank 1 and not empty.
 */
template <typename T, typename... P>
[[deprecated]] typename Kokkos::ViewTraits<T, P...>::non_const_value_type
lastElement(Kokkos::View<T, P...> const &v)
{
  using ExecutionSpace = typename Kokkos::View<T, P...>::execution_space;
  return Details::KokkosExt::lastElement(ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename T, typename... P>
[[deprecated]] void
iota(ExecutionSpace &&space, Kokkos::View<T, P...> const &v,
     typename Kokkos::ViewTraits<T, P...>::value_type value = 0)
{
  Details::KokkosExt::iota(std::forward<ExecutionSpace>(space), v, value);
}

template <typename T, typename... P>
[[deprecated]] inline void
iota(Kokkos::View<T, P...> const &v,
     typename Kokkos::ViewTraits<T, P...>::value_type value = 0)
{
  using ExecutionSpace = typename Kokkos::ViewTraits<T, P...>::execution_space;
  iota(ExecutionSpace{}, v, value);
}

template <typename ExecutionSpace, typename ViewType>
[[deprecated]] std::pair<typename ViewType::non_const_value_type,
                         typename ViewType::non_const_value_type>
minMax(ExecutionSpace &&space, ViewType const &v)
{
  return Details::KokkosExt::minmax_reduce(std::forward<ExecutionSpace>(space),
                                           v);
}

template <typename ViewType>
[[deprecated]] inline std::pair<typename ViewType::non_const_value_type,
                                typename ViewType::non_const_value_type>
minMax(ViewType const &v)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return minMax(ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename ViewType>
[[deprecated]] typename ViewType::non_const_value_type
min(ExecutionSpace &&space, ViewType const &v)
{
  return Details::KokkosExt::min_reduce(std::forward<ExecutionSpace>(space), v);
}

template <typename ViewType>
[[deprecated]] inline typename ViewType::non_const_value_type
min(ViewType const &v)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return min(ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename ViewType>
[[deprecated]] typename ViewType::non_const_value_type
max(ExecutionSpace &&space, ViewType const &v)
{
  return Details::KokkosExt::max_reduce(std::forward<ExecutionSpace>(space), v);
}

template <typename ViewType>
[[deprecated]] inline typename ViewType::non_const_value_type
max(ViewType const &v)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return max(ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename ViewType>
[[deprecated]] typename ViewType::non_const_value_type
accumulate(ExecutionSpace &&space, ViewType const &v,
           typename ViewType::non_const_value_type init)
{
  Details::KokkosExt::reduce(std::forward<ExecutionSpace>(space), v, init);
}

template <typename ViewType>
[[deprecated]] inline typename ViewType::non_const_value_type
accumulate(ViewType const &v, typename ViewType::non_const_value_type init)
{
  using ExecutionSpace = typename ViewType::execution_space;
  return accumulate(ExecutionSpace{}, v, init);
}

// FIXME shameless forward declaration
template <typename View>
typename View::non_const_type clone(View &v);

template <typename ExecutionSpace, typename SrcViewType, typename DstViewType>
[[deprecated]] void adjacentDifference(ExecutionSpace &&space,
                                       SrcViewType const &src,
                                       DstViewType const &dst)
{
  Details::KokkosExt::adjacent_difference(std::forward<ExecutionSpace>(space),
                                          src, dst);
}

template <typename SrcViewType, typename DstViewType>
[[deprecated]] inline void adjacentDifference(SrcViewType const &src,
                                              DstViewType const &dst)
{
  using ExecutionSpace = typename DstViewType::execution_space;
  adjacentDifference(ExecutionSpace{}, src, dst);
}

// FIXME split this into one for STL-like algorithms and another one for view
// utility helpers

// NOTE: not possible to avoid initialization with Kokkos::realloc()
template <typename View>
[[deprecated]] void
reallocWithoutInitializing(View &v, size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                           size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
{
  using ExecutionSpace = typename View::execution_space;
  Details::KokkosExt::reallocWithoutInitializing(ExecutionSpace{}, v, n0, n1,
                                                 n2, n3, n4, n5, n6, n7);
}

template <typename View>
[[deprecated]] void
reallocWithoutInitializing(View &v, typename View::array_layout const &layout)
{
  using ExecutionSpace = typename View::execution_space;
  Details::KokkosExt::reallocWithoutInitializing(ExecutionSpace{}, v, layout);
}

template <typename View>
[[deprecated]] typename View::non_const_type
cloneWithoutInitializingNorCopying(View &v)
{
  using ExecutionSpace = typename View::execution_space;
  return Details::KokkosExt::cloneWithoutInitializingNorCopying(
      ExecutionSpace{}, v);
}

template <typename ExecutionSpace, typename View>
[[deprecated]] typename View::non_const_type clone(ExecutionSpace const &space,
                                                   View &v)
{
  return Details::KokkosExt::clone(space, v);
}

template <typename View>
[[deprecated]] inline typename View::non_const_type clone(View &v)
{
  using ExecutionSpace = typename View::execution_space;
  return Details::KokkosExt::clone(ExecutionSpace{}, v);
}

namespace Details
{
template <typename ExecutionSpace, typename View, typename Offset>
void computeOffsetsInOrderedView(ExecutionSpace const &exec_space, View view,
                                 Offset &offsets)
{
  static_assert(KokkosExt::is_accessible_from<typename View::memory_space,
                                              ExecutionSpace>::value);
  static_assert(KokkosExt::is_accessible_from<typename Offset::memory_space,
                                              ExecutionSpace>::value);

  auto const n = view.extent_int(0);

  int num_offsets;
  KokkosExt::reallocWithoutInitializing(exec_space, offsets, n + 1);
  Kokkos::parallel_scan(
      "ArborX::Algorithms::compute_offsets_in_sorted_view",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n + 1),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cell_first_index =
            (i == 0 || i == n || view(i) != view(i - 1));
        if (is_cell_first_index)
        {
          if (final_pass)
            offsets(update) = i;
          ++update;
        }
      },
      num_offsets);
  Kokkos::resize(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing),
                 offsets, num_offsets);
}

} // namespace Details

} // namespace ArborX

#endif
