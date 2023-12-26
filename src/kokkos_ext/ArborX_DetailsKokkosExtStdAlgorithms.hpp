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

#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details::KokkosExt
{

// NOTE: This functor is used in exclusivePrefixSum( src, dst ).  We were
// getting a compile error on CUDA when using a KOKKOS_LAMBDA.
template <typename T, typename DeviceType>
class ExclusiveScanFunctor
{
public:
  ExclusiveScanFunctor(Kokkos::View<T *, DeviceType> const &in,
                       Kokkos::View<T *, DeviceType> const &out)
      : _in(in)
      , _out(out)
  {}
  KOKKOS_INLINE_FUNCTION void operator()(int i, T &update,
                                         bool final_pass) const
  {
    T const in_i = _in(i);
    if (final_pass)
      _out(i) = update;
    update += in_i;
  }

private:
  Kokkos::View<T *, DeviceType> _in;
  Kokkos::View<T *, DeviceType> _out;
};

template <typename ExecutionSpace, typename ST, typename... SP, typename DT,
          typename... DP>
void exclusive_scan(ExecutionSpace &&space, Kokkos::View<ST, SP...> const &src,
                    Kokkos::View<DT, DP...> const &dst)
{
  static_assert(
      std::is_same<
          typename Kokkos::ViewTraits<DT, DP...>::value_type,
          typename Kokkos::ViewTraits<DT, DP...>::non_const_value_type>::value,
      "exclusivePrefixSum requires non-const destination type");

  static_assert(
      (unsigned(Kokkos::ViewTraits<DT, DP...>::rank) ==
       unsigned(Kokkos::ViewTraits<ST, SP...>::rank)) &&
          (unsigned(Kokkos::ViewTraits<DT, DP...>::rank) == unsigned(1)),
      "exclusivePrefixSum requires Views of rank 1");

  using ValueType = typename Kokkos::ViewTraits<DT, DP...>::value_type;
  using DeviceType = typename Kokkos::ViewTraits<DT, DP...>::device_type;

  auto const n = src.extent(0);
  ARBORX_ASSERT(n == dst.extent(0));
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_scan("ArborX::Algorithms::exclusive_scan", policy,
                        ExclusiveScanFunctor<ValueType, DeviceType>(src, dst));
}

template <typename ExecutionSpace, typename T, typename... P>
inline std::enable_if_t<
    Kokkos::is_execution_space<std::remove_reference_t<ExecutionSpace>>::value>
exclusive_scan(ExecutionSpace &&space, Kokkos::View<T, P...> const &v)
{
  exclusive_scan(std::forward<ExecutionSpace>(space), v, v);
}

template <typename ExecutionSpace, typename ViewType>
typename ViewType::non_const_value_type
reduce(ExecutionSpace &&space, ViewType const &v,
       typename ViewType::non_const_value_type init)
{
  static_assert(ViewType::rank == 1, "accumulate requires a View of rank 1");
  auto const n = v.extent(0);
  // NOTE: Passing the argument init directly to the parallel_reduce() while
  // using a lambda does not yield the expected result because Kokkos will
  // supply a default init method that sets the reduction result to zero.
  // Rather than going through the hassle of defining a custom functor for
  // the reduction, introduce here a temporary variable and add it to init
  // before returning.
  typename ViewType::non_const_value_type tmp;
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_reduce(
      "ArborX::Algorithms::accumulate", policy,
      KOKKOS_LAMBDA(int i, typename ViewType::non_const_value_type &update) {
        update += v(i);
      },
      tmp);
  init += tmp;
  return init;
}

template <typename ExecutionSpace, typename SrcViewType, typename DstViewType>
void adjacent_difference(ExecutionSpace &&space, SrcViewType const &src,
                         DstViewType const &dst)
{
  static_assert(SrcViewType::rank == 1 && DstViewType::rank == 1,
                "adjacentDifference operates on rank-1 views");
  static_assert(std::is_same<typename DstViewType::value_type,
                             typename DstViewType::non_const_value_type>::value,
                "adjacentDifference requires non-const destination value type");
  static_assert(std::is_same<typename SrcViewType::non_const_value_type,
                             typename DstViewType::value_type>::value,
                "adjacentDifference requires same value type for source and "
                "destination");
  // QUESTION Should we assert anything about the memory spaces?
  auto const n = src.extent(0);
  ARBORX_ASSERT(n == dst.extent(0));
  ARBORX_ASSERT(src != dst);
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_for(
      "ArborX::Algorithms::adjacent_difference", policy, KOKKOS_LAMBDA(int i) {
        if (i > 0)
          dst(i) = src(i) - src(i - 1);
        else
          dst(i) = src(i);
      });
}

} // namespace ArborX::Details::KokkosExt

#endif
