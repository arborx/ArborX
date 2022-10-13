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

#ifndef ARBORX_DETAILS_WEIGHTED_EDGE_HPP
#define ARBORX_DETAILS_WEIGHTED_EDGE_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsMutualReachabilityDistance.hpp>
#include <ArborX_DetailsTreeNodeLabeling.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ProfileSection.hpp>

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
    using KokkosExt::min;
    auto const lhs_min = min(lhs.source, lhs.target);
    auto const rhs_min = min(rhs.source, rhs.target);
    if (lhs_min != rhs_min)
    {
      return (lhs_min < rhs_min);
    }
    using KokkosExt::max;
    auto const lhs_max = max(lhs.source, lhs.target);
    auto const rhs_max = max(rhs.source, rhs.target);
    return (lhs_max < rhs_max);
  }
};

} // namespace ArborX::Details

#endif
