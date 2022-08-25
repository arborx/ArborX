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

#ifndef ARBORX_DETAILS_BATCHED_QUERIES_HPP
#define ARBORX_DETAILS_BATCHED_QUERIES_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // returnCentroid, translateAndScale
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp> // sortObjects
#include <ArborX_DetailsUtils.hpp>     // exclusivePrefixSum, lastElement
#include <ArborX_HyperBox.hpp>
#include <ArborX_SpaceFillingCurves.hpp>

#include <Kokkos_Core.hpp>

#include <tuple>

namespace ArborX
{

namespace Details
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
    using Access = AccessTraits<Predicates, PredicatesTag>;
    auto const n_queries = Access::size(predicates);

    using Point = std::decay_t<decltype(returnCentroid(
        getGeometry(Access::get(predicates, 0))))>;
    using LinearOrderingValueType =
        Kokkos::detected_t<SpaceFillingCurveProjectionArchetypeExpression,
                           SpaceFillingCurve, Box, Point>;
    Kokkos::View<LinearOrderingValueType *, DeviceType> linear_ordering_indices(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::BVH::query::linear_ordering"),
        n_queries);
    Kokkos::parallel_for(
        "ArborX::BatchedQueries::project_predicates_onto_space_filling_curve",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          linear_ordering_indices(i) =
              curve(scene_bounding_box,
                    returnCentroid(getGeometry(Access::get(predicates, i))));
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
      -> Kokkos::View<typename AccessTraitsHelper<
                          AccessTraits<Predicates, PredicatesTag>>::type *,
                      DeviceType>
  {
    using Access = AccessTraits<Predicates, PredicatesTag>;
    auto const n = Access::size(v);
    ARBORX_ASSERT(permute.extent(0) == n);

    using T = std::decay_t<decltype(Access::get(
        std::declval<Predicates const &>(), std::declval<int>()))>;
    Kokkos::View<T *, DeviceType> w(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::permuted_predicates"),
        n);
    Kokkos::parallel_for(
        "ArborX::BatchedQueries::permute_entries",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
        KOKKOS_LAMBDA(int i) { w(i) = Access::get(v, permute(i)); });

    return w;
  }

  template <typename ExecutionSpace, typename Permute, typename Offset>
  static typename Offset::non_const_type
  permuteOffset(ExecutionSpace const &space, Permute const &permute,
                Offset const &offset)
  {
    auto const n = permute.extent(0);
    ARBORX_ASSERT(offset.extent(0) == n + 1);

    auto tmp_offset =
        KokkosExt::cloneWithoutInitializingNorCopying(space, offset);
    Kokkos::parallel_for(
        "ArborX::BatchedQueries::adjacent_difference_and_permutation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
          tmp_offset(permute(i)) = offset(i + 1) - offset(i);
        });

    exclusivePrefixSum(space, tmp_offset);

    return tmp_offset;
  }

  template <typename ExecutionSpace, typename Permute, typename Values,
            typename Offset, typename Offset2>
  static typename Values::non_const_type
  permuteIndices(ExecutionSpace const &space, Permute const &permute,
                 Values const &indices, Offset const &offset,
                 Offset2 const &tmp_offset)
  {
    auto const n = permute.extent(0);

    ARBORX_ASSERT(offset.extent(0) == n + 1);
    ARBORX_ASSERT(tmp_offset.extent(0) == n + 1);
    using KokkosExt::lastElement;
    ARBORX_ASSERT(lastElement(space, offset) == indices.extent_int(0));
    ARBORX_ASSERT(lastElement(space, tmp_offset) == indices.extent_int(0));

    auto tmp_indices =
        KokkosExt::cloneWithoutInitializingNorCopying(space, indices);
    Kokkos::parallel_for(
        "ArborX::BatchedQueries::permute_indices",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int q) {
          for (int i = 0; i < offset(q + 1) - offset(q); ++i)
          {
            tmp_indices(tmp_offset(permute(q)) + i) = indices(offset(q) + i);
          }
        });
    return tmp_indices;
  }

  template <typename ExecutionSpace, typename T, typename... P>
  static std::tuple<Kokkos::View<int *, DeviceType>, Kokkos::View<T *, P...>>
  reversePermutation(ExecutionSpace const &space,
                     Kokkos::View<unsigned int const *, DeviceType> permute,
                     Kokkos::View<int const *, DeviceType> offset,
                     Kokkos::View<T /*const*/ *, P...> out)
  {
    auto const tmp_offset = permuteOffset(space, permute, offset);

    auto const tmp_out =
        permuteIndices(space, permute, out, offset, tmp_offset);
    return std::make_tuple(tmp_offset, tmp_out);
  }

  template <typename ExecutionSpace>
  static std::tuple<Kokkos::View<int *, DeviceType>,
                    Kokkos::View<int *, DeviceType>,
                    Kokkos::View<float *, DeviceType>>
  reversePermutation(ExecutionSpace const &space,
                     Kokkos::View<unsigned int const *, DeviceType> permute,
                     Kokkos::View<int const *, DeviceType> offset,
                     Kokkos::View<int const *, DeviceType> indices,
                     Kokkos::View<float const *, DeviceType> distances)
  {
    auto const tmp_offset = permuteOffset(permute, offset);

    auto const tmp_indices =
        permuteIndices(space, permute, indices, offset, tmp_offset);

    auto const tmp_distances =
        permuteIndices(space, permute, distances, offset, tmp_offset);

    return std::make_tuple(tmp_offset, tmp_indices, tmp_distances);
  }
};

} // namespace Details
} // namespace ArborX

#endif
