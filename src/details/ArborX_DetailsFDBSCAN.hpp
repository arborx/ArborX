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

#ifndef ARBORX_DETAILSFDBSCAN_HPP
#define ARBORX_DETAILSFDBSCAN_HPP

#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

template <typename MemorySpace>
struct CountUpToN
{
  Kokkos::View<int *, MemorySpace> _counts;
  int _n;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int) const
  {
    auto i = getData(query);
    Kokkos::atomic_increment(&_counts(i));

    if (_counts(i) < _n)
      return ArborX::CallbackTreeTraversalControl::normal_continuation;

    // Once count reaches threshold, terminate the traversal.
    return ArborX::CallbackTreeTraversalControl::early_exit;
  }
};

template <typename MemorySpace, typename CorePointsType>
struct FDBSCANCallback
{
  UnionFind<MemorySpace> _union_find;
  CorePointsType _is_core_point;

  FDBSCANCallback(Kokkos::View<int *, MemorySpace> const &view,
                  CorePointsType is_core_point)
      : _union_find(view)
      , _is_core_point(is_core_point)
  {}

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int j) const
  {
    int const i = ArborX::getData(query);

    bool const is_border_point = !_is_core_point(i);
    if (is_border_point)
    {
      // Ignore border points, they will be processed by the
      // connected core points. Theoretically, border points could have been
      // filtered out prior to running the algorithm, but that may be expensive.
      return ArborX::CallbackTreeTraversalControl::early_exit;
    }

    bool const is_neighbor_core_point = _is_core_point(j);
    if (is_neighbor_core_point && i > j)
    {
      // For a core point that is connected to another core point, do the
      // standard CCS algorithm
      _union_find.merge(i, j);
    }
    else if (!is_neighbor_core_point)
    {
      // For a border point that is connected to a core point, set its
      // representative to that of the core point. If it is connected to
      // multiple core points, it will be assigned to the cluster that the last
      // core point was in.
      //
      // NOTE: DO NOT USE merge(i, j) here. This may set this border
      // point as a representative for the whole cluster. This would mean that
      // a) labels_(i) == i still (so it would be processed later, and b) it may
      // be combined with a different cluster later forming a bridge.
      _union_find.merge_into(j, i);
    }

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};
} // namespace Details
} // namespace ArborX

#endif
