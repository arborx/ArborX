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

#ifndef ARBORX_WEIGHTED_EDGE_HPP
#define ARBORX_WEIGHTED_EDGE_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>

namespace ArborX::Details
{

struct WeightedEdge
{
  int source;
  int target;
  float weight;

private:
  // performs lexicographical comparison by comparing first the weights and then
  // the unordered pair of vertices
  friend KOKKOS_FUNCTION constexpr bool operator<(WeightedEdge const &lhs,
                                                  WeightedEdge const &rhs)
  {
    if (lhs.weight != rhs.weight)
    {
      return (lhs.weight < rhs.weight);
    }
    using Kokkos::min;
    auto const lhs_min = min(lhs.source, lhs.target);
    auto const rhs_min = min(rhs.source, rhs.target);
    if (lhs_min != rhs_min)
    {
      return (lhs_min < rhs_min);
    }
    using Kokkos::max;
    auto const lhs_max = max(lhs.source, lhs.target);
    auto const rhs_max = max(rhs.source, rhs.target);
    return (lhs_max < rhs_max);
  }
};

} // namespace ArborX::Details

#endif
