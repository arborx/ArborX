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

#ifndef ARBORX_MUTUAL_REACHABILITY_DISTANCE_HPP
#define ARBORX_MUTUAL_REACHABILITY_DISTANCE_HPP

#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_HappyTreeFriends.hpp>
#include <detail/ArborX_Predicates.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>

namespace ArborX
{
namespace Details
{

template <class Primitives, class Distances>
struct MaxDistance
{
  Primitives _primitives;
  Distances _distances;

  using memory_space = typename Primitives::memory_space;
  using size_type = typename memory_space::size_type;

  template <class Predicate, typename Value>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value) const
  {
    size_type const i = value.index;
    size_type const j = getData(predicate);
    using Kokkos::max;
    auto const distance_ij = distance(_primitives(i), _primitives(j));
    // NOTE using knowledge that each nearest predicate traversal is performed
    // by a single thread.  The distance update below would need to be atomic
    // otherwise.
    _distances(j) = max(_distances(j), distance_ij);
  }
};

template <class CoreDistances>
struct MutualReachability
{
  CoreDistances _core_distances;
  using value_type = typename CoreDistances::non_const_value_type;
  using size_type = typename CoreDistances::memory_space::size_type;

  KOKKOS_FUNCTION value_type operator()(size_type i, size_type j,
                                        value_type distance_ij) const
  {
    using Kokkos::max;
    return max({_core_distances(i), _core_distances(j), distance_ij});
  }
};

struct Euclidean
{
  using value_type = float;
  using size_type = int;
  KOKKOS_FUNCTION value_type operator()(size_type /*i*/, size_type /*j*/,
                                        value_type distance_ij) const
  {
    return distance_ij;
  }
};

} // namespace Details

} // namespace ArborX

#endif
