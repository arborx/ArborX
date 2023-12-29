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

#ifndef ARBORX_DETAILS_BATCHED_QUERIES_HPP
#define ARBORX_DETAILS_BATCHED_QUERIES_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // returnCentroid, translateAndScale
#include <ArborX_DetailsSortUtils.hpp>  // sortObjects
#include <ArborX_HyperBox.hpp>
#include <ArborX_SpaceFillingCurves.hpp>

#include <Kokkos_Core.hpp>

#include <tuple>

namespace ArborX::Details
{
template <typename DeviceType>
struct BatchedQueries
{
public:
  // BatchedQueries defines functions for sorting queries along the Z-order
  // space-filling curve in order to minimize data divergence.  The goal is
  // to increase correlation between traversal decisions made by nearby
  // threads and thereby increase performance.
  //
  // NOTE: sortQueriesAlongZOrderCurve() does not actually apply the sorting
  // order, it returns the permutation indices.  applyPermutation() was added
  // in that purpose.  reversePermutation() is able to restore the initial
  // order on the results that are in "compressed row storage" format.  You
  // may notice it is not used anymore in the code that performs the batched
  // queries.  We found that it was slightly more performant to add a level of
  // indirection when recording results rather than using that function at
  // the end.  We decided to keep reversePermutation around for now.

  template <typename ExecutionSpace, typename Predicates, typename Box,
            typename SpaceFillingCurve>
  static Kokkos::View<unsigned int *, DeviceType>
  sortPredicatesAlongSpaceFillingCurve(ExecutionSpace const &space,
                                       SpaceFillingCurve const &curve,
                                       Box const &scene_bounding_box,
                                       Predicates const &predicates)
  {
    auto const n_queries = predicates.size();

    using Point =
        std::decay_t<decltype(returnCentroid(getGeometry(predicates(0))))>;
    using LinearOrderingValueType =
        std::invoke_result_t<SpaceFillingCurve, Box, Point>;
    Kokkos::View<LinearOrderingValueType *, DeviceType> linear_ordering_indices(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::BVH::query::linear_ordering"),
        n_queries);
    Kokkos::parallel_for(
        "ArborX::BatchedQueries::project_predicates_onto_space_filling_curve",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          linear_ordering_indices(i) = curve(
              scene_bounding_box, returnCentroid(getGeometry(predicates(i))));
        });

    return sortObjects(space, linear_ordering_indices);
  }

  // NOTE  trailing return type seems required :(
  // error: The enclosing parent function ("applyPermutation") for an extended
  // __host__ __device__ lambda must not have deduced return type
  template <typename ExecutionSpace, typename Predicates>
  static auto
  applyPermutation(ExecutionSpace const &space,
                   Kokkos::View<unsigned int const *, DeviceType> permute,
                   Predicates const &v)
      -> Kokkos::View<typename Predicates::value_type *, DeviceType>
  {
    auto const n = v.size();
    ARBORX_ASSERT(permute.extent(0) == n);

    Kokkos::View<typename Predicates::value_type *, DeviceType> w(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::permuted_predicates"),
        n);
    Kokkos::parallel_for(
        "ArborX::BatchedQueries::permute_entries",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
        KOKKOS_LAMBDA(int i) { w(i) = v(permute(i)); });

    return w;
  }
};

} // namespace ArborX::Details

#endif
