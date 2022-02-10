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

#ifndef ARBORX_DETAILS_MUTUAL_REACHABILITY_DISTANCE_HPP
#define ARBORX_DETAILS_MUTUAL_REACHABILITY_DISTANCE_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_Predicates.hpp>

namespace ArborX
{
namespace Details
{

template <class Primitives, class Distances>
struct MaxDistance
{
  Primitives _primitives;
  Distances _distances;
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using memory_space = typename Access::memory_space;
  using size_type = typename memory_space::size_type;
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, size_type i) const
  {
    size_type const j = getData(predicate);
    using KokkosExt::max;
    auto const distance_ij =
        distance(Access::get(_primitives, i), Access::get(_primitives, j));
    // NOTE using knowledge that each nearest predicate traversal is performed
    // by a single thread.  The distance update below would need to be atomic
    // otherwise.
    _distances(j) = max(_distances(j), distance_ij);
  }
};

template <class Primitives>
struct NearestK
{
  Primitives primitives;
  int k; // including self-collisions
};

} // namespace Details

template <class Primitives>
struct AccessTraits<Details::NearestK<Primitives>, PredicatesTag>
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using memory_space = typename Access::memory_space;
  using size_type = typename memory_space::size_type;
  static KOKKOS_FUNCTION size_type size(Details::NearestK<Primitives> const &x)
  {
    return Access::size(x.primitives);
  }
  static KOKKOS_FUNCTION auto get(Details::NearestK<Primitives> const &x,
                                  size_type i)
  {
    return attach(nearest(Access::get(x.primitives, i), x.k), i);
  }
};

namespace Details
{

template <class CoreDistances>
struct MutualReachability
{
  CoreDistances _core_distances;
  using value_type = typename CoreDistances::non_const_value_type;
  using size_type = typename CoreDistances::memory_space::size_type;

  KOKKOS_FUNCTION value_type operator()(size_type i, size_type j,
                                        value_type distance_ij) const
  {
    using KokkosExt::max;
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
