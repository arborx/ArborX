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

#ifndef ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP
#define ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsTreeConstruction.hpp> // Kokkos::reduction_identity<ArborX::Box>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{
struct BruteForceImpl
{
  template <class ExecutionSpace, class Primitives, class BoundingVolumes,
            class Bounds>
  static void initializeBoundingVolumesAndReduceBoundsOfTheScene(
      ExecutionSpace const &space, Primitives const &primitives,
      BoundingVolumes const &bounding_volumes, Bounds &bounds)
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    int const n = Access::size(primitives);

    Kokkos::parallel_reduce("ArborX::BruteForce::BruteForce::"
                            "initialize_bounding_volumes_and_reduce_bounds",
                            Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                            KOKKOS_LAMBDA(int i, Bounds &update) {
                              using Details::expand;
                              Bounds bounding_volume{};
                              expand(bounding_volume,
                                     Access::get(primitives, i));
                              bounding_volumes(i) = bounding_volume;
                              update += bounding_volume;
                            },
                            bounds);
  }

  template <class ExecutionSpace, class Primitives, class Predicates,
            class Callback>
  static void query(ExecutionSpace const &space, Primitives const &primitives,
                    Predicates const &predicates, Callback const &callback)
  {
    using AccessPrimitives = AccessTraits<Primitives, PrimitivesTag>;
    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;

    int const n_primitives = AccessPrimitives::size(primitives);
    int const n_predicates = AccessPredicates::size(predicates);

    Kokkos::parallel_for(
        "ArborX::BruteForce::query::spatial::"
        "check_all_predicates_against_all_primitives",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {0, 0}, {n_primitives, n_predicates}),
        KOKKOS_LAMBDA(int primitive_index, int predicate_index) {
          auto const &predicate =
              AccessPredicates::get(predicates, predicate_index);
          auto const &primitive =
              AccessPrimitives::get(primitives, primitive_index);
          if (predicate(primitive))
          {
            callback(predicate, primitive_index);
          }
        });
  }
};
} // namespace Details
} // namespace ArborX

#endif
