/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
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

#include <detail/ArborX_Callbacks.hpp>
#include <detail/ArborX_PairValueIndex.hpp>
#include <detail/ArborX_Predicates.hpp>
#include <detail/ArborX_UnionFind.hpp>

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

  template <typename Query, typename Value>
  KOKKOS_FUNCTION auto operator()(Query const &query, Value const &) const
  {
    int const i = getData(query);
    int &count = _counts(i);
    if (Kokkos::atomic_inc_fetch(&count) >= _n)
      return ArborX::CallbackTreeTraversalControl::early_exit;

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};

template <typename UnionFind, typename CorePointsType, bool DbscanStar = false>
struct FDBSCANCallback
{
  UnionFind _union_find;
  CorePointsType _is_core_point;

  template <typename Value, typename Index>
  KOKKOS_FUNCTION auto
  operator()(PairValueIndex<Value, Index> const &value1,
             PairValueIndex<Value, Index> const &value2) const
  {
    int i = value1.index;
    int j = value2.index;

    bool const is_border_point = !_is_core_point(i);
    bool const neighbor_is_core_point = _is_core_point(j);

    if constexpr (DbscanStar == true)
    {
      // Border points do not participate in merging in DBSCAN*
      if (is_border_point || !neighbor_is_core_point)
        return ArborX::CallbackTreeTraversalControl::early_exit;
    }

    if (is_border_point)
    {
      if (neighbor_is_core_point)
      {
        // For a border point that is connected to a core point, set its
        // representative to that of the core point. If it is connected to
        // multiple core points, it will be assigned to the cluster that the
        // first found core point neighbor was in.
        //
        // NOTE: DO NOT USE merge(i, j) here. This may set this border point
        // as a representative for the whole cluster potentially forming a
        // bridge with a different cluster.
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
