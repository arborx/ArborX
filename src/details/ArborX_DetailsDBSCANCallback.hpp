/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILSDBSCANCALLBACK_HPP
#define ARBORX_DETAILSDBSCANCALLBACK_HPP

#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

template <typename MemorySpace, typename CorePointsType>
struct DBSCANCallback
{
  UnionFind<MemorySpace> union_find_;
  CorePointsType is_core_point_;

  DBSCANCallback(Kokkos::View<int *, MemorySpace> const &view,
                 CorePointsType is_core_point)
      : union_find_(view)
      , is_core_point_(is_core_point)
  {
  }

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int j) const
  {
    int const i = ArborX::getData(query);

    // NOTE: for halo finder/ccs algorithm (in which is_core_point(i) is always
    // true), the algorithm below will be simplified to
    //   if (i > j)

    if (!is_core_point_(j))
    {
      // The neighbor is not a core point, do nothing
      return;
    }

    bool is_boundary_point =
        !is_core_point_(i); // is_core_point_(j) is aready true

    if (is_boundary_point && union_find_.representative(i) == i)
    {
      // For a boundary point that was not processed before (labels_(i) == i),
      // set its representative to that of the core point. This way, when
      // another neighbor that is core point appears later, we won't process
      // this point.
      //
      // NOTE: DO NOT USE merge(i, j) here. This may set this boundary
      // point as a representative for the whole cluster. This would mean that
      // a) labels_(i) == i still (so it would be processed later, and b) it may
      // be combined with a different cluster later forming a bridge.
      union_find_.merge_into(i, j);
    }
    else if (!is_boundary_point && i > j)
    {
      // For a core point that is connected to another core point, do the
      // standard CCS algorithm
      union_find_.merge(i, j);
    }
  }
};
} // namespace Details
} // namespace ArborX

#endif
