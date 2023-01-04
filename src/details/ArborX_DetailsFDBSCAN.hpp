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

template <typename UnionFind, typename CorePointsType>
struct FDBSCANCallback
{
  UnionFind _union_find;
  CorePointsType _is_core_point;

  KOKKOS_FUNCTION auto operator()(int i, int j) const
  {
    bool const is_border_point = !_is_core_point(i);
    bool const neighbor_is_core_point = _is_core_point(j);
    if (is_border_point)
    {
      if (neighbor_is_core_point)
      {
        // For a border point that is connected to a core point, set its
        // representative to that of the core point. If it is connected to
        // multiple core points, it will be assigned to the cluster that the
        // first found core point neighbor was in.
        //
        // NOTE: DO NOT USE merge(i, j) here. This may set this border point as
        // a representative for the whole cluster potentially forming a bridge
        // with a different cluster.
        _union_find.merge_into(i, j);

        // Once a border point is assigned to a cluster, can terminate the
        // associated traversal.
        return ArborX::CallbackTreeTraversalControl::early_exit;
      }
    }
    else
    {
      if (neighbor_is_core_point)
      {
        // For a core point that is connected to another core point, do the
        // standard CCS algorithm
        _union_find.merge(i, j);
      }
      else
      {
        // Merge the neighbor in (see NOTE about border points above).
        _union_find.merge_into(j, i);
      }
    }

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};

} // namespace Details
} // namespace ArborX

#endif
